/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace ReverseSequence_Kernel {

// seq_lengths 分 chunk 缓存进 UB，单个 chunk 最多缓存的 batch 个数
constexpr int64_t SEQ_CACHE_CHUNK = 2048;

struct ReverseSequenceArgs {
    GM_ADDR input;
    GM_ADDR seqLengths;
    GM_ADDR output;

    int64_t batchSize;
    int64_t maxSeqLen;
    int64_t dataDim;

    int64_t left;
    int64_t handleTotalCount;

    uint32_t rowsPerTile;
    uint32_t rowAlignCount;
};

template <typename dType, typename tType>
class ReverseSequence {
public:
    __aicore__ inline ReverseSequence(){};

    __aicore__ inline void init(ReverseSequenceArgs& args, TPipe& pipe)
    {
        this->args = &args;

        // 行区间均摊：前 left 个核各多承担 1 行
        int64_t thisId = GetBlockIdx();
        int64_t rowCount = args.handleTotalCount;
        if (thisId < args.left) {
            rowCount += 1;
            rowStart = thisId * rowCount;
        } else {
            rowStart = (args.handleTotalCount + 1) * args.left + (thisId - args.left) * rowCount;
        }
        rowEnd = rowStart + rowCount;

        inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ dType*>(args.input),
                                args.batchSize * args.maxSeqLen * args.dataDim);
        seqLengthsGM.SetGlobalBuffer(reinterpret_cast<__gm__ tType*>(args.seqLengths), args.batchSize);
        outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ dType*>(args.output),
                                 args.batchSize * args.maxSeqLen * args.dataDim);

        // 行 tile buffer：rowsPerTile 行，UB 内每行按 32B 对齐后的 stride（rowAlignCount 个元素）存放
        pipe.InitBuffer(rowBuf, static_cast<uint32_t>(args.rowsPerTile * args.rowAlignCount * sizeof(dType)));
        pipe.InitBuffer(seqBuf, static_cast<uint32_t>(SEQ_CACHE_CHUNK * sizeof(tType)));

        eventMte2Mte3 = pipe.FetchEventID(HardEvent::MTE2_MTE3);
        eventMte3Mte2 = pipe.FetchEventID(HardEvent::MTE3_MTE2);
        eventMte2S = pipe.FetchEventID(HardEvent::MTE2_S);
    }

    __aicore__ inline void Compute()
    {
        if (rowStart >= rowEnd || args->maxSeqLen <= 0) {
            return;
        }
        const int64_t msl = args->maxSeqLen;
        // 本核行区间覆盖的 batch 区间，仅在入口处做两次除法，行内全部用递增计数
        const int64_t batchBegin = rowStart / msl;
        const int64_t batchEnd = (rowEnd - 1) / msl;

        for (int64_t chunkBegin = batchBegin; chunkBegin <= batchEnd; chunkBegin += SEQ_CACHE_CHUNK) {
            int64_t batchCount = batchEnd - chunkBegin + 1;
            if (batchCount > SEQ_CACHE_CHUNK) {
                batchCount = SEQ_CACHE_CHUNK;
            }
            LoadSeqLengths(chunkBegin, batchCount);
            for (int64_t b = chunkBegin; b < chunkBegin + batchCount; ++b) {
                ProcessBatch(b, seqBuf.Get<tType>().GetValue(b - chunkBegin));
            }
        }
    }

private:
    // 将 [batchBegin, batchBegin + batchCount) 的 seq_lengths 一次搬入 UB，
    // 替代逐行 GlobalTensor::GetValue 标量读 GM（效率极低，生产代码禁用）
    __aicore__ inline void LoadSeqLengths(int64_t batchBegin, int64_t batchCount)
    {
        LocalTensor<tType> seqLocal = seqBuf.Get<tType>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(batchCount * sizeof(tType));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPadExtParams<tType> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyPad(seqLocal, seqLengthsGM[batchBegin], params, padParams);
        SetFlag<HardEvent::MTE2_S>(eventMte2S);
        WaitFlag<HardEvent::MTE2_S>(eventMte2S);
    }

    __aicore__ inline void ProcessBatch(int64_t batchIdx, int64_t seqLen)
    {
        const int64_t msl = args->maxSeqLen;
        const int64_t dataDim = args->dataDim;
        // 防御性截断：seq_len 超过 max_seq_len 时按 max_seq_len 处理，避免越界写 GM
        if (seqLen > msl) {
            seqLen = msl;
        }
        const int64_t batchRowBase = batchIdx * msl;
        // 本核需搬运的行区间：行号属于本核行范围且 < seqLen。
        // seqLen<=0 或区间不相交时为空，整段跳过（这些行保持 output 原值，不做任何搬运）
        int64_t sLo = rowStart > batchRowBase ? rowStart : batchRowBase;
        int64_t sHi = batchRowBase + seqLen;
        if (sHi > rowEnd) {
            sHi = rowEnd;
        }
        if (sLo >= sHi) {
            return;
        }

        const uint32_t rowBytes = static_cast<uint32_t>(dataDim * sizeof(dType));
        LocalTensor<dType> rowLocal = rowBuf.Get<dType>();
        for (int64_t tileBegin = sLo; tileBegin < sHi; tileBegin += args->rowsPerTile) {
            const int64_t remain = sHi - tileBegin;
            const uint32_t tileRows =
                remain < static_cast<int64_t>(args->rowsPerTile) ? static_cast<uint32_t>(remain) : args->rowsPerTile;
            // 有效行在 GM 上连续，blockCount 模式一次搬入 tileRows 行；
            // UB 内每行按 32B 对齐 stride 存放，保证后续逐行写出的 UB 起址 32B 对齐
            DataCopyExtParams inParams;
            inParams.blockCount = static_cast<uint16_t>(tileRows);
            inParams.blockLen = rowBytes;
            inParams.srcStride = 0;
            inParams.dstStride = 0;
            inParams.rsv = 0;
            DataCopyPadExtParams<dType> inPadParams;
            inPadParams.isPad = false;
            inPadParams.leftPadding = 0;
            inPadParams.rightPadding = static_cast<uint8_t>(args->rowAlignCount - dataDim);
            inPadParams.paddingValue = 0;
            DataCopyPad(rowLocal, inputGM[tileBegin * dataDim], inParams, inPadParams);
            SetFlag<HardEvent::MTE2_MTE3>(eventMte2Mte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventMte2Mte3);

            for (uint32_t r = 0; r < tileRows; ++r) {
                const int64_t seqIdx = tileBegin + r - batchRowBase;
                const int64_t revRow = batchRowBase + seqLen - 1 - seqIdx;
                DataCopyExtParams outParams;
                outParams.blockCount = 1;
                outParams.blockLen = rowBytes;
                outParams.srcStride = 0;
                outParams.dstStride = 0;
                outParams.rsv = 0;
                DataCopyPad(outputGM[revRow * dataDim], rowLocal[r * args->rowAlignCount], outParams);
            }
            // 单 buffer：本 tile 写出完成后，下一 tile 的搬入才允许覆盖 UB
            SetFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMte3Mte2);
        }
    }

    int64_t rowStart;
    int64_t rowEnd;
    ReverseSequenceArgs* args;

    GlobalTensor<dType> inputGM;
    GlobalTensor<tType> seqLengthsGM;
    GlobalTensor<dType> outputGM;

    TBuf<TPosition::VECCALC> rowBuf;
    TBuf<TPosition::VECCALC> seqBuf;

    TEventID eventMte2Mte3;
    TEventID eventMte3Mte2;
    TEventID eventMte2S;
};
}  // namespace ReverseSequence_Kernel

// call of kernel function
extern "C" __global__ __aicore__ void reverse_sequence_cust(GM_ADDR input, GM_ADDR seq_lengths, GM_ADDR output,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    ReverseSequence_Kernel::ReverseSequenceArgs args{input,
                                                     seq_lengths,
                                                     output,
                                                     tiling_data.batchSize,
                                                     tiling_data.maxSeqLen,
                                                     tiling_data.dataDim,
                                                     tiling_data.left,
                                                     tiling_data.handleTotalCount,
                                                     tiling_data.rowsPerTile,
                                                     tiling_data.rowAlignCount};

    TPipe pipe;
    ReverseSequence_Kernel::ReverseSequence<DTYPE_INPUT, DTYPE_SEQ_LENGTHS> kernel;
    kernel.init(args, pipe);
    kernel.Compute();
}
