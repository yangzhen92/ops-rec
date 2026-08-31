#ifndef HSTU_KERNEL_TILING_KEY_JAGGED_H
#define HSTU_KERNEL_TILING_KEY_JAGGED_H

#include "ascendc/host_api/tiling/template_argument.h"

// 本文件与 hstu_kernel_tiling_key_dense.h / hstu_kernel_tiling_key_paged.h 由
// 原 hstu_kernel_tiling_key.h 按 typeTilingKey 拆分而来：每个 kernel 只编译自己
// 实际派发的组合，避免 .o 携带其它 op 的实例化（死代码）。
// DECL 必须与另两个文件逐字一致 —— tiling key 的位宽布局由 DECL 决定，
// 任何差异都会导致 key 编码错位；host 侧 op_host/atlasA2/hstu_kernel_tiling_key.h
// 保留全表，修改需同步。

// maskedType: { 0:mask_tril, 2:mask_none, 3:mask_custom }
// enableBias: { 0:no_bias, 1:bias }
// typeTilingKey: { 0:normal, 1:jagged, 2:paged }
// deterministic: {0:False, 1:true}
ASCENDC_TPL_ARGS_DECL(HstuDenseForward,  // 算子OpType
                      ASCENDC_TPL_UINT_DECL(maskedType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 2, 3),
                      ASCENDC_TPL_BOOL_DECL(enableBias, 0, 1),
                      ASCENDC_TPL_UINT_DECL(typeTilingKey, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 2),
                      ASCENDC_TPL_BOOL_DECL(deterministic, 0, 1),
                      ASCENDC_TPL_UINT_DECL(blockM, 9, ASCENDC_TPL_UI_LIST, 256),
                      ASCENDC_TPL_UINT_DECL(blockN, 9, ASCENDC_TPL_UI_LIST, 256),
                      ASCENDC_TPL_UINT_DECL(blockK, 10, ASCENDC_TPL_UI_LIST, 512), );

// jagged格式：固定 256×256×512，仅 typeTilingKey=1
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
                                     ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_RANGE, 1, 1, 1),
                                     ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 256),
                                     ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 256),
                                     ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 512), ), );

#endif
