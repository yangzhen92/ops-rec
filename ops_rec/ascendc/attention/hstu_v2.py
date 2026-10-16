"""High-level HSTU V2 entry point using the unified attention plan."""

# pylint: disable=no-name-in-module
# 各算子的 autograd 源码位于 ascendc/**/torch_extension/*_autograd.py，由 build.sh 的
# sync_autograd() 在打包阶段同步进 ops_rec/autograd/。因此该模块在源码树中不存在，
# pylint 无法静态解析，但打包后运行时可用。此处为有意为之，勿改为静态可见的导入路径。
from ops_rec.autograd.attn_plan_autograd import (
    BwdAttnPlan,
    FwdAttnPlan,
    get_bwd_attn_plan,
    get_fwd_attn_plan,
    hstu_v2,
)

__all__ = ["BwdAttnPlan", "FwdAttnPlan", "get_bwd_attn_plan", "get_fwd_attn_plan", "hstu_v2"]
