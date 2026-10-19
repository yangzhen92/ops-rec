/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
============================================================================== */

/*!
 * \file create_attn_metadata.h
 * \brief
 */

#ifndef hstu_attn_METADATA_H
#define hstu_attn_METADATA_H

#include <cstdint>
#include <cassert>
#include "../../attn_plan/common/attn_metadata_layout.h"
#include "../../attn_plan/common/attn_schedule_policy.h"

namespace optiling {

// Constants
constexpr uint32_t FA_META_SIZE = 1024U;
using FA_METADATA_T = uint32_t;

namespace metadata = ops_rec::attn_plan::metadata;
constexpr uint32_t HEAD_METADATA_STRIDE = metadata::HEAD_STRIDE;
constexpr uint32_t FA_METADATA_STRIDE = metadata::FA_STRIDE;
constexpr uint32_t FD_METADATA_STRIDE = metadata::FD_STRIDE;

// Head Metadata Index Definitions
constexpr uint32_t HEAD_SECTION_NUM_INDEX = metadata::HEAD_SECTION_NUM;
constexpr uint32_t HEAD_IS_FD_INDEX = metadata::HEAD_IS_FD;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = metadata::HEAD_M_BASE_SIZE;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = metadata::HEAD_S2_BASE_SIZE;
constexpr uint32_t HEAD_AIV_CORE_NUM_INDEX = metadata::HEAD_AIV_CORE_NUM;
constexpr uint32_t HEAD_AIC_CORE_NUM_INDEX = metadata::HEAD_AIC_CORE_NUM;
constexpr uint32_t HEAD_DIRECT_MAGIC_INDEX = metadata::HEAD_DIRECT_MAGIC;
constexpr uint32_t HSTU_DIRECT_MAGIC = metadata::DIRECT_MAGIC;
constexpr uint32_t DIRECT_CORE_TASK_BEGIN_INDEX = metadata::DIRECT_TASK_BEGIN;
constexpr uint32_t DIRECT_CORE_TASK_COUNT_INDEX = metadata::DIRECT_TASK_COUNT;

// FA Metadata Index Definitions
constexpr uint32_t FA_BN2_START_INDEX = metadata::FA_BN2_START;
constexpr uint32_t FA_M_START_INDEX = metadata::FA_M_START;
constexpr uint32_t FA_S2_START_INDEX = metadata::FA_S2_START;
constexpr uint32_t FA_BN2_END_INDEX = metadata::FA_BN2_END;
constexpr uint32_t FA_M_END_INDEX = metadata::FA_M_END;
constexpr uint32_t FA_S2_END_INDEX = metadata::FA_S2_END;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = metadata::FA_FIRST_FD_WORKSPACE;

// FD Metadata Index Definitions
constexpr uint32_t FD_BN2_IDX_INDEX = metadata::FD_BN2;
constexpr uint32_t FD_M_IDX_INDEX = metadata::FD_M;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = metadata::FD_WORKSPACE;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = metadata::FD_WORKSPACE_NUM;
constexpr uint32_t FD_M_START_INDEX = metadata::FD_M_START;
constexpr uint32_t FD_M_NUM_INDEX = metadata::FD_M_NUM;

namespace detail {
struct FaMetadata {
    uint32_t sectionNum;
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    FA_METADATA_T* headMetadata;  // [HEAD_METADATA_STRIDE];
    FA_METADATA_T* faMetadata;    // [sectionNum][aicCoreNum][FA_METADATA_STRIDE];
    FA_METADATA_T* fdMetadata;    // [sectionNum][aivCoreNum][FD_METADATA_STRIDE];
    FaMetadata(void* metadataPtr, uint32_t sectionNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
        : sectionNum(sectionNum),
          aicCoreNum(aicCoreNum),
          aivCoreNum(aivCoreNum),
          headMetadata(static_cast<FA_METADATA_T*>(metadataPtr)),
          faMetadata(headMetadata + HEAD_METADATA_STRIDE),
          fdMetadata(faMetadata + sectionNum * aicCoreNum * FA_METADATA_STRIDE)
    {
        headMetadata[0] = sectionNum;
    }

    void Clear(bool clearFdMetadata = true)
    {
        for (size_t i = 0; i < HEAD_METADATA_STRIDE; ++i) {
            headMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aicCoreNum * FA_METADATA_STRIDE; ++i) {
            faMetadata[i] = 0U;
        }
        if (clearFdMetadata) {
            for (size_t i = 0; i < sectionNum * aivCoreNum * FD_METADATA_STRIDE; ++i) {
                fdMetadata[i] = 0U;
            }
        }
    }

    void SetHeadMetadata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < HEAD_METADATA_STRIDE);
        headMetadata[metaIdx] = val;
    }

    uint32_t GetHeadMetadata(uint32_t metaIdx)
    {
        assert(metaIdx < HEAD_METADATA_STRIDE);
        return headMetadata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicCoreNum);
        assert(metaIdx < FA_METADATA_STRIDE);
        faMetadata[sectionIdx * aicCoreNum * FA_METADATA_STRIDE + aicIdx * FA_METADATA_STRIDE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicCoreNum);
        assert(metaIdx < FA_METADATA_STRIDE);
        return faMetadata[aicCoreNum * FA_METADATA_STRIDE * sectionIdx + FA_METADATA_STRIDE * aicIdx + metaIdx];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivCoreNum);
        assert(metaIdx < FD_METADATA_STRIDE);
        fdMetadata[aivCoreNum * FD_METADATA_STRIDE * sectionIdx + FD_METADATA_STRIDE * aivIdx + metaIdx] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivCoreNum);
        assert(metaIdx < FD_METADATA_STRIDE);
        return fdMetadata[aivCoreNum * FD_METADATA_STRIDE * sectionIdx + FD_METADATA_STRIDE * aivIdx + metaIdx];
    }
};
}  // namespace detail

}  // namespace optiling

#endif
