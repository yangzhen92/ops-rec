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

#include "kernel_operator.h"

using namespace AscendC;

namespace TokenMixing_Kernel {
constexpr int ALIGN_32 = 32;
struct TokenMixingArgs {
    GM_ADDR x;
    GM_ADDR x_t;
    GM_ADDR gamma;
    GM_ADDR beta;
    GM_ADDR y;
    float epsilon;
    uint32_t coreNum;
    uint64_t xDim0;
    uint64_t xDim1;
    uint64_t xDim2;
    uint64_t xDim2WithPadding;
    uint64_t perCoreComputeRows;
    uint64_t formerCoreRows;
    uint64_t formerLoopCount;
    uint64_t formerRemainRows;
    uint64_t tailCoreRows;
    uint64_t tailLoopCount;
    uint64_t tailRemainRows;
};

template <typename T>
__aicore__ inline void CpGm2Local(const LocalTensor<T>& lt, const GlobalTensor<T>& gt, uint64_t len)
{
    uint64_t alignLen = len * sizeof(T) / ALIGN_32 * ALIGN_32;
    uint64_t unAlignLen = len * sizeof(T) - alignLen;

    GlobalTensor<uint16_t> uint16Gt;
    uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(T) / 2);
    LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

    if (alignLen != 0) {
        DataCopy(uint16Lt, uint16Gt, alignLen / 2);
    }
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, static_cast<uint32_t>(unAlignLen), 0, 0, 0};
        const DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(uint16Lt[alignLen / 2], uint16Gt[alignLen / 2], dataCopyExtParams, dataCopyPadExtParams);
    }
}

template <typename T>
__aicore__ inline void CpLocal2Gm(const GlobalTensor<T>& gt, const LocalTensor<T>& lt, uint64_t len)
{
    uint64_t alignLen = len * sizeof(T) / ALIGN_32 * ALIGN_32;
    uint64_t unAlignLen = len * sizeof(T) - alignLen;

    GlobalTensor<uint16_t> uint16Gt;
    uint16Gt.SetGlobalBuffer((__gm__ uint16_t*)gt.GetPhyAddr(), len * sizeof(T) / 2);
    LocalTensor<uint16_t> uint16Lt = lt.template ReinterpretCast<uint16_t>();

    if (alignLen != 0) {
        DataCopy(uint16Gt, uint16Lt, alignLen / 2);
    }
    if (unAlignLen != 0) {
        const DataCopyExtParams dataCopyExtParams{1, static_cast<uint32_t>(unAlignLen), 0, 0, 0};
        const DataCopyPadExtParams<uint16_t> dataCopyPadExtParams{false, 0, 0, 0};
        DataCopyPad(uint16Gt[alignLen / 2], uint16Lt[alignLen / 2], dataCopyExtParams);
    }
}

template <typename T>
class TokenMixing {
public:
    __aicore__ inline TokenMixing(){};

    // 初始化函数，完成内存初始化相关操作
    __aicore__ inline void Init(TokenMixingArgs* args, TPipe* pipe)
    {
        // 入参指针保存
        this->args = args;
        this->pipe = pipe;
        bshLength_ = args->perCoreComputeRows *
                     (args->xDim2WithPadding);  // 满载情况下，单次搬运的元素总数（行数 * 对齐后的列数）
        bsLength_ = args->perCoreComputeRows;  // 满载情况下，单次搬运的行数

        // 当前核的index
        int64_t blockIdx = GetBlockIdx();                   // 当前ai core 的硬件索引id
        bool isTailCore = (blockIdx == args->coreNum - 1);  // 判断当前核是否为尾核
        // 根据是否是尾核，从 args 中认领当前核需要处理的总行数、满载循环次数、剩余不足满载的行数
        coreRows_ = isTailCore ? args->tailCoreRows : args->formerCoreRows;
        loopCount_ = isTailCore ? args->tailLoopCount : args->formerLoopCount;
        remainRows_ = isTailCore ? args->tailRemainRows : args->formerRemainRows;
        // 计算当前核在全局数据中的起始行偏移量：前面的核数 * 前面核的固定分配行数
        startRow_ = blockIdx * args->formerCoreRows;

        bshLength_ = args->perCoreComputeRows * args->xDim2WithPadding;
        bsLength_ = args->perCoreComputeRows;

        // 将外部传入的 GM 物理地址绑定到 GlobalTensor 对象上，以便后续调用 Ascend C 接口
        xGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->x), bshLength_);
        xTGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->x_t), bshLength_);
        gammaGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->gamma), args->xDim2WithPadding);
        betaGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->beta), args->xDim2WithPadding);
        yGT.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(args->y), bshLength_);
        // 在 UB (Unified Buffer) 中为各个队列分配物理内存，参数为：队列、深度(1)、字节数
        // 申请输入队列
        pipe->InitBuffer(inQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(inTQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(inQueueGamma, 1, (args->xDim2WithPadding) * sizeof(T));
        pipe->InitBuffer(inQueueBeta, 1, (args->xDim2WithPadding) * sizeof(T));
        // 申请输出队列
        pipe->InitBuffer(outQueue, 1, bshLength_ * sizeof(T));
        // 申请计算中间态队列 (用于 LayerNorm 的均值、方差、各种中间张量)
        pipe->InitBuffer(outQueueMean, 1, bsLength_ * sizeof(float));
        pipe->InitBuffer(outQueueRstd, 1, bsLength_ * sizeof(float));
        pipe->InitBuffer(addTmpQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(tmpQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(tmpMulQueue, 1, bshLength_ * sizeof(T));
        pipe->InitBuffer(oneQueue, 1, bsLength_ * sizeof(T));
    }

    __aicore__ inline void Process()  // 流水线控制
    {
        // 如果当前核没有分到任务，直接退出（处理极小数据的情况）
        if (coreRows_ == 0) {
            return;
        }
        // 满载循环处理
        for (uint64_t i = 0; i < loopCount_; ++i) {
            CopyIn(i, args->perCoreComputeRows);   // 搬入满载行数的数据到 UB
            Compute(args->perCoreComputeRows);     // 在 UB 内进行计算
            CopyOut(i, args->perCoreComputeRows);  // 把计算结果从 UB 搬回 GM
        }
        // 处理最后剩下的不足一次满载的剩余行数据
        if (remainRows_ > 0) {
            CopyIn(loopCount_, remainRows_);
            Compute(remainRows_);
            CopyOut(loopCount_, remainRows_);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t iterIdx, uint64_t rowCount)  // 数据搬入
    {
        // 核内偏移
        // 计算当前循环在 Global Memory 中的绝对起步偏移量：
        // 当前核的起步偏移 + 当前循环批次的偏移
        uint64_t offsetInCore = startRow_ * args->xDim2 + iterIdx * args->perCoreComputeRows * args->xDim2;
        uint64_t copyLen = rowCount * args->xDim2;

        // 从队列中分配本地 (UB) 张量空间
        LocalTensor<T> xLocal = inQueue.AllocTensor<T>();
        Duplicate(xLocal, 0.0f, bshLength_);  // 初始化为0
        LocalTensor<T> xTLocal = inTQueue.AllocTensor<T>();
        Duplicate(xTLocal, 0.0f, bshLength_);  // 初始化为0
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        LocalTensor<T> betaLocal = inQueueBeta.AllocTensor<T>();
        // 同步所有核，确保所有核都完成本地张量的分配
        PipeBarrier<PIPE_ALL>();

        // 如果原始通道数本身就是 32 字节对齐
        if (args->xDim2 == args->xDim2WithPadding) {
            // 直接连续搬运大块数据，速度最快
            CpGm2Local(xLocal, xGT[offsetInCore], copyLen);
            CpGm2Local(xTLocal, xTGT[offsetInCore], copyLen);
        } else {
            // 如果不对齐，必须按行搬运，并在每行的末尾补齐 Padding 零
            uint64_t xDataLen = args->xDim2 * sizeof(float);  // 每行的实际字节数
            DataCopyExtParams params{1, static_cast<uint32_t>(xDataLen), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, 0, 0};  // padding
            for (uint64_t i = 0; i < rowCount; ++i) {
                // 逐行搬运：目标地址跳跃 xDim2WithPadding，源地址跳跃原始 xDim2
                DataCopyPad(xLocal[i * args->xDim2WithPadding], xGT[offsetInCore + i * args->xDim2], params, padParams);
                DataCopyPad(xTLocal[i * args->xDim2WithPadding], xTGT[offsetInCore + i * args->xDim2], params,
                            padParams);
            }
        }
        // gamma 和 beta 是 1 维向量，只搬运一次
        CpGm2Local(gammaLocal, gammaGT, args->xDim2);
        CpGm2Local(betaLocal, betaGT, args->xDim2);
        // 将填充好数据的张量推入队列，供 Compute 阶段使用
        inQueue.EnQue(xLocal);
        inTQueue.EnQue(xTLocal);
        inQueueGamma.EnQue(gammaLocal);
        inQueueBeta.EnQue(betaLocal);
    }

    __aicore__ inline void Compute(uint64_t rowCount)  // 计算阶段
    {
        // 输入队列出队， 从队列中获取本地 (UB) 张量
        LocalTensor<T> xLocal = inQueue.DeQue<T>();
        LocalTensor<T> xTLocal = inTQueue.DeQue<T>();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<T> betaLocal = inQueueBeta.DeQue<T>();
        // 分配临时计算张量和输出张量
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        LocalTensor<float> meanLocal = outQueueMean.AllocTensor<float>();
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        LocalTensor<T> addTmpLocal = addTmpQueue.AllocTensor<T>();
        LocalTensor<T> tmpLocal = tmpQueue.AllocTensor<T>();
        LocalTensor<T> tmpMulLocal = tmpMulQueue.AllocTensor<T>();
        LocalTensor<T> oneLocal = oneQueue.AllocTensor<T>();
        Duplicate(oneLocal, 1.0f, rowCount);  // 全为 1 的向量，用于后续 1/sqrt(x) 的被除数

        // 第一步：向量加法 addTmpLocal = x + x_t
        Add(addTmpLocal, xLocal, xTLocal, bshLength_);  // 计算 x + x_t
        // shape定义
        uint32_t meanShape[2] = {static_cast<uint32_t>(rowCount), 1};
#ifdef SUPPORT_950
        uint32_t shape[2] = {static_cast<uint32_t>(rowCount), static_cast<uint32_t>(args->xDim2WithPadding)};
#else
        uint32_t shape[2] = {static_cast<uint32_t>(rowCount), static_cast<uint32_t>(args->xDim2)};
#endif
        // layerNorm拆分
        // 计算平均值mean
        Muls(tmpMulLocal, addTmpLocal, 1.0f / args->xDim2, bshLength_);
        PipeBarrier<PIPE_V>();  // 矢量计算同步屏障
        // 沿行方向进行 ReduceSum，得到每一行的 mean
        ReduceSum<float, Pattern::Reduce::AR, false>(meanLocal, tmpMulLocal, shape, true);
        PipeBarrier<PIPE_V>();
        // broadcast mean[rowCount] -> [rowCount, xDim2WithPadding]
        // 将 [rowCount, 1] 的 mean 广播拉伸为 [rowCount, H_pad] 大小，方便后续做逐元素减法
        uint32_t trueShape[2] = {static_cast<uint32_t>(rowCount), static_cast<uint32_t>(args->xDim2WithPadding)};
        Broadcast<float, 2, 1>(tmpLocal, meanLocal, trueShape, meanShape);
        // 如果补了 padding，需要把 padding 位置的 mean 置为 0，防止影响后续计算
        if (args->xDim2 < args->xDim2WithPadding) {
            for (uint64_t i = 0; i < rowCount; ++i) {
                for (uint64_t j = args->xDim2; j < args->xDim2WithPadding; ++j) {
                    tmpLocal.SetValue(i * args->xDim2WithPadding + j, 0.0f);
                }
            }
        }
        // (x-mean) * (x-mean)
        Sub(tmpLocal, addTmpLocal, tmpLocal, rowCount * args->xDim2WithPadding);  // 计算 x - mean存入tmpLocal
        Mul(yLocal, tmpLocal, tmpLocal, rowCount * args->xDim2WithPadding);  // 计算 (x-mean) * (x-mean)存入yLocal
        // 计算var
        Muls(yLocal, yLocal, 1.0f / args->xDim2, bshLength_);
        ReduceSum<float, Pattern::Reduce::AR, false>(rstdLocal, yLocal, shape, true);
        PipeBarrier<PIPE_V>();
        // 计算rstd=1/sqrt(var+epsilon)  标准差的倒数
        Adds(rstdLocal, rstdLocal, args->epsilon, rowCount);
        Sqrt(rstdLocal, rstdLocal, rowCount);
        Div(rstdLocal, oneLocal, rstdLocal, rowCount);
        Broadcast<float, 2, 1>(yLocal, rstdLocal, trueShape, meanShape);
        // 计算layernorm=(rstd*(x-mean))*gamma+beta
        Mul(yLocal, yLocal, tmpLocal, rowCount * args->xDim2WithPadding);
        // 广播gamma
        uint32_t gammaShape[2] = {1, static_cast<uint32_t>(args->xDim2WithPadding)};
        Broadcast<float, 2, 0>(tmpLocal, gammaLocal, trueShape, gammaShape);
        Mul(yLocal, yLocal, tmpLocal, rowCount * args->xDim2WithPadding);
        // 广播beta
        Broadcast<float, 2, 0>(tmpLocal, betaLocal, trueShape, gammaShape);
        Add(yLocal, yLocal, tmpLocal, rowCount * args->xDim2WithPadding);

        outQueue.EnQue<T>(yLocal);  // 将计算结果推入输出队列，供 CopyOut 阶段使用

        // 释放输入和临时张量的 UB 空间，给下一轮循环复用
        inQueue.FreeTensor(xLocal);
        inTQueue.FreeTensor(xTLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        inQueueBeta.FreeTensor(betaLocal);
        addTmpQueue.FreeTensor(addTmpLocal);
        tmpMulQueue.FreeTensor(tmpMulLocal);
        tmpQueue.FreeTensor(tmpLocal);
        oneQueue.FreeTensor(oneLocal);
        outQueueMean.FreeTensor(meanLocal);
        outQueueRstd.FreeTensor(rstdLocal);
    }

    __aicore__ inline void CopyOut(uint64_t iterIdx, uint64_t rowCount)
    {
        // 核内偏移
        // 计算写入 GM 的起始绝对偏移量
        uint64_t offsetInCore = startRow_ * args->xDim2 + iterIdx * args->perCoreComputeRows * args->xDim2;
        uint64_t copyLen = rowCount * args->xDim2;
        LocalTensor<T> yLocal = outQueue.DeQue<T>();  // 从输出队列中取出包含最终结果的张量
        // 根据是否做了 32 字节补齐，决定连续写回还是跳跃式按行写回
        if (args->xDim2 == args->xDim2WithPadding) {
            // 若对齐，直接整块数据拷贝回 GM
            CpLocal2Gm(yGT[offsetInCore], yLocal, copyLen);
        } else {
            // 若不对齐，按行拷贝回 GM，丢弃 UB 里的 Padding 尾巴数据
            for (int i = 0; i < rowCount; ++i) {
                CpLocal2Gm(yGT[offsetInCore + i * args->xDim2], yLocal[i * args->xDim2WithPadding], args->xDim2);
            }
        }
        // 释放输出队列中的张量，给下一轮循环复用
        outQueue.FreeTensor(yLocal);
    }

    TPipe* pipe;
    GlobalTensor<T> xGT, xTGT, gammaGT, betaGT, yGT;
    GlobalTensor<float> meanGT, rstdGT;
    TQue<TPosition::VECIN, 1> inQueue;
    TQue<TPosition::VECIN, 1> inTQueue;
    TQue<TPosition::VECIN, 1> inQueueGamma;
    TQue<TPosition::VECIN, 1> inQueueBeta;
    TQue<TPosition::VECOUT, 1> outQueue;
    TQue<TPosition::VECCALC, 1> outQueueMean;
    TQue<TPosition::VECCALC, 1> outQueueRstd;
    TQue<TPosition::VECCALC, 1> addTmpQueue;
    TQue<TPosition::VECCALC, 1> tmpQueue;
    TQue<TPosition::VECCALC, 1> tmpMulQueue;
    TQue<TPosition::VECCALC, 1> oneQueue;

    TokenMixingArgs* args;
    uint64_t bshLength_;
    uint64_t bsLength_;
    uint64_t coreRows_;
    uint64_t loopCount_;
    uint64_t remainRows_;
    uint64_t startRow_;
};
}  // namespace TokenMixing_Kernel

// 内核入口函数 参数全部是物理内存地址
// 把 Host 存入共享内存的 Tiling 数据解包，实例化计算类，并启动流水线
extern "C" __global__ __aicore__ void token_mixing(GM_ADDR x, GM_ADDR x_t, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
                                                   GM_ADDR workspace,
                                                   GM_ADDR tiling)  // 运行于 AI Core 上的全局内核函数
{
    // 从 tiling 物理地址中，反序列化提取出 Host 端计算好的排班表数据
    GET_TILING_DATA(tiling_data, tiling);

    // 组装参数结构体，把物理地址和 Tiling 数据塞在一起，方便后续传递
    TokenMixing_Kernel::TokenMixingArgs args{x,
                                             x_t,
                                             gamma,
                                             beta,
                                             y,
                                             tiling_data.epsilon,
                                             tiling_data.coreNum,
                                             tiling_data.xDim0,
                                             tiling_data.xDim1,
                                             tiling_data.xDim2,
                                             tiling_data.xDim2WithPadding,
                                             tiling_data.perCoreComputeRows,
                                             tiling_data.formerCoreRows,
                                             tiling_data.formerLoopCount,
                                             tiling_data.formerRemainRows,
                                             tiling_data.tailCoreRows,
                                             tiling_data.tailLoopCount,
                                             tiling_data.tailRemainRows};
    TPipe pipe;              // 实例化同步管道，用于控制异步数据搬运和计算之间的同步
    if (TILING_KEY_IS(0)) {  // 对应 Host 端设置的 TilingKey（当前只支持 float32，即 0
        // 实例化具体的 TokenMixing 算子类，模板参数为 float
        TokenMixing_Kernel::TokenMixing<float> kernel;
        // 调用初始化接口，传入参数和管道指针
        kernel.Init(&args, &pipe);
        // 调用执行接口，启动流水线计算
        kernel.Process();
    }
}
