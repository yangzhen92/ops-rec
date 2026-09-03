#ifndef HSTU_KERNEL_TILING_KEY_JAGGED_H
#define HSTU_KERNEL_TILING_KEY_JAGGED_H

#include "ascendc/host_api/tiling/template_argument.h"

// 本文件与 hstu_kernel_tiling_key_dense.h / hstu_kernel_tiling_key_paged.h 由
// 原 hstu_kernel_tiling_key.h 按 typeTilingKey 拆分而来：每个 kernel 只编译自己
// 实际派发的组合，避免 .o 携带其它 op 的实例化（死代码）。
// DECL 必须与另两个文件逐字一致 —— tiling key 的位宽布局由 DECL 决定，
// 任何差异都会导致 key 编码错位；host 侧 op_host/{soc}/hstu_kernel_tiling_key.h
// 保留全表，三处修改需同步。

// ——— dtype 窄化 ———
// DTYPE_Q 由构建框架按 dtype 逐编译单元注入（类型 token）：fp8 单元展开为
// fp8_e4m3fn_t、fp32 单元为 float（与 kernel 内 std::is_same 判断一致）。
// host 侧 TilingKeySetImpl（op_host/atlas950/tiling_policy_arch.cpp）对 fp8/fp32
// 固定只派发 (256,256,512)，因此这两个 dtype 的编译单元只需实例化该单 tile
// （mask×bias×det = 3×2×2 = 12 组合），其余 27 tile × 12 是永不派发的死实例；
// fp16/bf16 走 FindMatchShape 按 seq 长度/head_dim 动态选 tile，保留全部 28 组。
// 若框架注入的 token 拼写与本表不符，判别宏展开为未定义 → #if 视为 0 →
// 该 dtype 回退全量 SEL（功能不受影响，仅失去窄化），不会编译失败。
#define HSTU_TPL_CAT_I(a, b) a##b
#define HSTU_TPL_CAT(a, b) HSTU_TPL_CAT_I(a, b)
#define HSTU_TPL_TOKEN_IS_IMPL_float 1
#define HSTU_TPL_TOKEN_IS_IMPL_fp8_e4m3fn_t 1
#define HSTU_TPL_IS_NARROW_DTYPE HSTU_TPL_CAT(HSTU_TPL_TOKEN_IS_IMPL_, DTYPE_Q)

// maskedType: { 0:mask_tril, 2:mask_none, 3:mask_custom }
// enableBias: { 0:no_bias, 1:bias }
// typeTilingKey: { 0:normal, 1:jagged, 2:paged }
// deterministic: {0:False, 1:true}
ASCENDC_TPL_ARGS_DECL(HstuDenseForward,  // 算子OpType
                      ASCENDC_TPL_UINT_DECL(maskedType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 2, 3),
                      ASCENDC_TPL_BOOL_DECL(enableBias, 0, 1),
                      ASCENDC_TPL_UINT_DECL(typeTilingKey, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, 0, 1, 2),
                      ASCENDC_TPL_BOOL_DECL(deterministic, 0, 1),
                      ASCENDC_TPL_UINT_DECL(blockM, 9, ASCENDC_TPL_UI_LIST, 32, 64, 128, 256),
                      ASCENDC_TPL_UINT_DECL(blockN, 13, ASCENDC_TPL_UI_LIST, 32, 64, 128, 256, 512, 1024),
                      ASCENDC_TPL_UINT_DECL(blockK, 10, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512), );

#if HSTU_TPL_IS_NARROW_DTYPE
// 窄化分支（fp8_e4m3fn_t / float 编译单元）：host 固定只派发 (256,256,512)，
// 只实例化该单 tile —— maskedType{0,2,3} × enableBias{0,1} × deterministic{0,1} = 12 组合。
// 其余 27 tile × 12 组合为永不派发的死实例，不再编译。
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
                                     ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 256),
                                     ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 256),
                                     ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 512), ), );
#else
// 全量分支（fp16/bf16 及其它 dtype）：FindMatchShape 按 seq 长度/head_dim 动态选 tile，
// 保留全部 28 组。
// jagged格式：FindMatchShape 按 seq 长度选 tile shape（仅 typeTilingKey=1）
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
                                     ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 256),
                                     ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 256),
                                     ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
                                     ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 32),
                                     ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 32),
                                     ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
                                     ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 64),
                                     ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 64, 1024),
                                     ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512), ),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(maskedType, ASCENDC_TPL_UI_LIST, 0, 2, 3),
                                     ASCENDC_TPL_BOOL_SEL(enableBias, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(typeTilingKey, ASCENDC_TPL_UI_LIST, 1),
                                     ASCENDC_TPL_BOOL_SEL(deterministic, 0, 1),
                                     ASCENDC_TPL_UINT_SEL(blockM, ASCENDC_TPL_UI_LIST, 128),
                                     ASCENDC_TPL_UINT_SEL(blockN, ASCENDC_TPL_UI_LIST, 128, 256, 512),
                                     ASCENDC_TPL_UINT_SEL(blockK, ASCENDC_TPL_UI_LIST, 64, 128, 256, 512), ), );
#endif

#undef HSTU_TPL_CAT_I
#undef HSTU_TPL_CAT
#undef HSTU_TPL_TOKEN_IS_IMPL_float
#undef HSTU_TPL_TOKEN_IS_IMPL_fp8_e4m3fn_t
#undef HSTU_TPL_IS_NARROW_DTYPE

#endif
