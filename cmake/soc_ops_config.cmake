# ===========================================================================
# soc_ops_config.cmake — ops-rec AscendC 算子列表（按 SOC 芯片管理）
# 控制每个 SOC variant 编译哪些算子，同时用于生成 config.ini 的 load_priority。
# 新增算子时在对应芯片列表追加算子名，无需改动 CMakeLists.txt。
# ================================ A5 ops ================================
set(OPS_REC_OPS_ATLAS_950
    ln_mul
    disentangle_attention
)

# ================================ A3 ops ================================
set(OPS_REC_OPS_ATLAS_A3
    ln_mul
    disentangle_attention
)

# ================================ A2 ops ================================
set(OPS_REC_OPS_ATLAS_A2
    ln_mul
    disentangle_attention
)
