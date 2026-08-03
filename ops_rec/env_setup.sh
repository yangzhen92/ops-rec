#!/bin/bash
# ============================================================================
# ops_rec 环境设置脚本
# ----------------------------------------------------------------------------
# 使用方式：
#   source env_setup.sh
#
# import ops_rec 时会通过 __init__.py 在当前进程内自动刷新 OPP 路径，
# 本脚本仅在需要 shell 级别预先配置环境时使用。
# ============================================================================

# 获取脚本所在目录（ops_rec 包目录）
_ops_rec_pkg_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_opp_dir="${_ops_rec_pkg_dir}/opp"

# 遍历 opp/ 下所有 variant 目录（如 atlas_950, atlas_a2, atlas_a3），
# 将每个 variant 的 vendors/ 和 variant 根目录加入 ASCEND_CUSTOM_OPP_PATH。
# 这样不依赖 SOC 检测，适合多芯片环境下的 shell 级预配置。
for _variant_dir in "${_opp_dir}"/*/; do
    if [ -d "${_variant_dir}vendors" ]; then
        # 添加 vendors 下每个 vendor 目录
        for _vendor_dir in "${_variant_dir}vendors/"*/; do
            if [ -d "$_vendor_dir" ]; then
                if [ -z "${ASCEND_CUSTOM_OPP_PATH}" ]; then
                    export ASCEND_CUSTOM_OPP_PATH="${_vendor_dir%/}"
                else
                    case ":${ASCEND_CUSTOM_OPP_PATH}:" in
                        *":${_vendor_dir%/}:"*) ;;
                        *) ASCEND_CUSTOM_OPP_PATH="${_vendor_dir%/}:${ASCEND_CUSTOM_OPP_PATH}" ;;
                    esac
                    export ASCEND_CUSTOM_OPP_PATH
                fi
            fi
        done
        # 添加 variant 根目录
        _variant_root="${_variant_dir%/}"
        case ":${ASCEND_CUSTOM_OPP_PATH}:" in
            *":${_variant_root}:"*) ;;
            *) ASCEND_CUSTOM_OPP_PATH="${_variant_root}:${ASCEND_CUSTOM_OPP_PATH}" ;;
        esac
        export ASCEND_CUSTOM_OPP_PATH
    fi
done

unset _ops_rec_pkg_dir _opp_dir _variant_dir _vendor_dir _variant_root
