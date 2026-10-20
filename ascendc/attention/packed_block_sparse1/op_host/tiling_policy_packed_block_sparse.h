#ifndef OPS_REC_PACKED_BLOCK_SPARSE_TILING_H
#define OPS_REC_PACKED_BLOCK_SPARSE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PackedBlockSparseTilingData)
TILING_DATA_FIELD_DEF(int32_t, strideH);
TILING_DATA_FIELD_DEF(int32_t, strideQ);
TILING_DATA_FIELD_DEF(int32_t, strideF);
TILING_DATA_FIELD_DEF(int32_t, maxSeqLenQ);
TILING_DATA_FIELD_DEF(int32_t, maxSeqLenKv);
TILING_DATA_FIELD_DEF(int32_t, nFunc);
TILING_DATA_FIELD_DEF(int32_t, qBlockSize);
TILING_DATA_FIELD_DEF(int32_t, kvBlockSize);
TILING_DATA_FIELD_DEF(int32_t, numQBlocks);
TILING_DATA_FIELD_DEF(int32_t, numKvBlocks);
TILING_DATA_FIELD_DEF(int32_t, hNum);
TILING_DATA_FIELD_DEF(int32_t, bNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CreateFwdPackedBlockSparse, PackedBlockSparseTilingData)
REGISTER_TILING_DATA_CLASS(CreateBwdPackedBlockSparse, PackedBlockSparseTilingData)
}  // namespace optiling
#endif
