"""Official user_case benchmark matrix for the current planned HSTU V2 path.

The profiler driver runs this matrix once for each implementation mode.  Each
case issues exactly ten warm-up and ten measured plan-plus-kernel launches.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sysconfig

import pytest
import torch
import torch_npu  # noqa: F401


WARMUP = 10
REPEATS = 10
SEED = 123
CURRENT_SO = f"{sysconfig.get_path('purelib')}/ops_rec/ops_rec_atlas_950.so"
BETA_SO = os.getenv("HSTU_BETA_SO", "/tmp/agent_task1_env/beta_legacy/libhstu_v2.so")
BETA_METADATA_SO = os.getenv("HSTU_BETA_METADATA_SO", "/tmp/agent_task1_env/beta_official/libhstu_attn_metadata.so")


@dataclass(frozen=True)
class UserCase:
    case_id: int
    family: str
    batch: int
    heads: int
    dim_qk: int
    dim_v: int
    max_q: int
    max_k: int
    has_rab: bool
    dtype: torch.dtype
    has_af: bool = False

    @property
    def dtype_name(self):
        return "float16" if self.dtype == torch.float16 else "bfloat16"

    @property
    def pytest_id(self):
        return (
            f"uc{self.case_id:03d}-{self.family}-b{self.batch}-h{self.heads}-"
            f"q{self.max_q}-k{self.max_k}-dq{self.dim_qk}-dv{self.dim_v}-"
            f"{self.dtype_name}-rab{int(self.has_rab)}"
        )

    def as_dict(self):
        return {
            "case_id": self.case_id,
            "family": self.family,
            "batch": self.batch,
            "heads": self.heads,
            "dim_qk": self.dim_qk,
            "dim_v": self.dim_v,
            "max_q": self.max_q,
            "max_k": self.max_k,
            "has_rab": self.has_rab,
            "rab_heads": 1 if self.has_rab else 0,
            "has_af": self.has_af,
            "dtype": self.dtype_name,
            "seed": SEED,
        }


def user_cases():
    """Return the official user_case_1 + user_case_2 configurations."""
    specs = [
        ("user_case_1", 2048, 4, 64, 64, 52, 1000, False, torch.float16),
        ("user_case_1", 16, 4, 64, 64, 501, 1000, False, torch.float16),
        ("user_case_1", 8, 8, 128, 128, 8000, 8000, False, torch.float16),
        ("user_case_1", 8, 8, 256, 256, 8000, 8000, False, torch.float16),
        ("user_case_1", 2048, 2, 256, 256, 32, 32, False, torch.float16),
        ("user_case_1", 96, 2, 256, 256, 512, 3072, False, torch.float16),
        ("user_case_1", 16, 4, 64, 32, 32, 499, False, torch.float16),
        ("user_case_1", 4, 4, 128, 48, 1001, 901, False, torch.float16),
        ("user_case_1", 128, 4, 128, 128, 7621, 7621, False, torch.bfloat16),
        ("user_case_1", 96, 3, 256, 256, 674, 674, False, torch.bfloat16),
    ]
    for dtype in (torch.float16, torch.bfloat16):
        for has_rab in (True, False):
            for dim in (32, 64, 128, 256):
                for seq in (512, 1024, 2048, 4096):
                    specs.append(("user_case_2", 32, 8, dim, dim, seq, seq, has_rab, dtype))
    return [UserCase(index, *spec) for index, spec in enumerate(specs, 1)]


def pytest_params():
    return [pytest.param(case, id=case.pytest_id) for case in user_cases()]


class UserCaseRunner:
    def __init__(self, direction):
        self.direction = direction
        self.mode = os.getenv("HSTU_BENCHMARK_MODE", "current_deterministic")
        if self.mode not in ("beta", "current_deterministic", "current_nondeterministic"):
            raise ValueError(f"unsupported HSTU_BENCHMARK_MODE={self.mode}")
        torch.ops.load_library(BETA_SO if self.mode == "beta" else CURRENT_SO)
        if self.mode == "beta":
            torch.ops.load_library(BETA_METADATA_SO)
        torch.npu.set_device(int(os.getenv("ASCEND_DEVICE_ID", "0")))
        self.record_path = Path(os.getenv("HSTU_BENCHMARK_RECORD", f"/tmp/hstu_{direction}_{self.mode}.jsonl"))
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_path.write_text("", encoding="utf-8")
        self.unsupported_path = self.record_path.with_suffix(".unsupported.jsonl")
        self.unsupported_path.write_text("", encoding="utf-8")

    @property
    def deterministic(self):
        return self.mode != "current_nondeterministic"

    def _beta_metadata(self, case, data):
        _, q, _, v, _, offsets_q, offsets_k = data
        return torch.ops.mxrec.hstu_attn_metadata(
            offsets_q,
            offsets_k,
            None,
            None,
            case.batch,
            case.max_q,
            case.max_k,
            case.heads,
            case.heads,
            max(case.dim_qk, case.dim_v),
            0,
            -1,
            -1,
            "TND",
            "TND",
            "TND",
        )

    def _invoke_beta(self, case, data, metadata):
        grad, q, k, v, rab, offsets_q, offsets_k = data
        if self.direction == "forward":
            return torch.ops.mxrec.hstu_forward_v2(
                q,
                k,
                v,
                None,
                rab,
                0,
                case.max_q,
                case.max_k,
                1 / 1024,
                offsets_q,
                offsets_k,
                None,
                None,
                None,
                0.5,
                metadata,
            )
        return torch.ops.mxrec.hstu_backward_v2(
            grad,
            q,
            k,
            v,
            case.max_q,
            case.max_k,
            offsets_q,
            offsets_k,
            rab,
            None,
            None,
            1 / 1024,
            None,
            0.5,
            -1,
            -1,
            metadata,
            None,
            None,
        )

    @staticmethod
    def _allocate(case):
        torch.manual_seed(SEED + case.case_id)
        q_len = max(1, int(case.max_q * 0.9))
        k_len = max(1, int(case.max_k * 0.9))
        total_q = q_len * case.batch
        total_k = k_len * case.batch
        q = torch.empty((total_q, case.heads, case.dim_qk), dtype=case.dtype, device="npu").uniform_(-0.2, 0.2)
        k = torch.empty((total_k, case.heads, case.dim_qk), dtype=case.dtype, device="npu").uniform_(-0.2, 0.2)
        v = torch.empty((total_k, case.heads, case.dim_v), dtype=case.dtype, device="npu").uniform_(-0.2, 0.2)
        grad = torch.empty((total_q, case.heads, case.dim_v), dtype=case.dtype, device="npu").uniform_(-0.2, 0.2)
        rab = (
            torch.empty((case.batch, 1, case.max_q, case.max_k), dtype=case.dtype, device="npu").uniform_(-0.1, 0.1)
            if case.has_rab
            else None
        )
        offsets_q = torch.arange(case.batch + 1, dtype=torch.int32, device="npu") * q_len
        offsets_k = torch.arange(case.batch + 1, dtype=torch.int32, device="npu") * k_len
        return grad, q, k, v, rab, offsets_q, offsets_k

    def _invoke(self, case, data):
        grad, q, k, v, rab, offsets_q, offsets_k = data
        context = target = af = None
        if case.has_af:
            context = torch.zeros(case.batch, dtype=torch.int32, device="npu")
            target = torch.zeros(case.batch, dtype=torch.int32, device="npu")
            af = torch.ops.ops_rec.build_hstu_mask_arbitrary_func(offsets_k, context, target, 1, k.shape[0])
        plan_op = (
            torch.ops.ops_rec.get_fwd_attn_plan if self.direction == "forward" else torch.ops.ops_rec.get_bwd_attn_plan
        )
        tile, sparse, metadata, use_aicpu = plan_op(
            q,
            k,
            v,
            af,
            offsets_q,
            offsets_k,
            case.max_q,
            case.max_k,
            context,
            target,
            1,
            -1,
            0,
            0,
            self.deterministic,
        )
        if self.direction == "forward":
            output = torch.ops.ops_rec.hstu_forward_v2(
                q,
                k,
                v,
                None,
                rab,
                0,
                case.max_q,
                case.max_k,
                1 / 1024,
                offsets_q,
                offsets_k,
                context,
                target,
                1,
                0.5,
                metadata,
                sparse or None,
                -1,
                0,
                af,
                self.deterministic,
            )
        else:
            output = torch.ops.ops_rec.hstu_backward_v2(
                grad,
                q,
                k,
                v,
                case.max_q,
                case.max_k,
                offsets_q,
                offsets_k,
                rab,
                context,
                target,
                1 / 1024,
                1,
                0.5,
                -1,
                0,
                metadata,
                sparse or None,
                self.deterministic,
            )
        return output, use_aicpu, list(tile)

    def run(self, case):
        if self.mode == "beta" and case.has_rab and case.heads != 1:
            with self.unsupported_path.open("a", encoding="utf-8") as stream:
                for pipeline in ("no_metadata", "metadata"):
                    stream.write(
                        json.dumps(
                            {
                                **case.as_dict(),
                                "direction": self.direction,
                                "mode": self.mode,
                                "pipeline": pipeline,
                                "supported": False,
                                "reason": "v26.1.0-beta.0 requires rab.size(1) == heads",
                            }
                        )
                        + "\n"
                    )
            return
        data = self._allocate(case)
        if self.mode == "beta":
            for pipeline in ("no_metadata", "metadata"):
                metadata = self._beta_metadata(case, data) if pipeline == "metadata" else None
                torch.npu.synchronize()
                for _ in range(WARMUP):
                    self._invoke_beta(case, data, metadata)
                torch.npu.synchronize()
                for _ in range(REPEATS):
                    self._invoke_beta(case, data, metadata)
                torch.npu.synchronize()
                with self.record_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {
                                **case.as_dict(),
                                "direction": self.direction,
                                "mode": self.mode,
                                "pipeline": pipeline,
                                "warmup": WARMUP,
                                "repeats": REPEATS,
                            }
                        )
                        + "\n"
                    )
            del data
            torch.npu.empty_cache()
            return
        torch.npu.synchronize()
        use_aicpu = False
        tile = None
        for _ in range(WARMUP):
            _, use_aicpu, tile = self._invoke(case, data)
        torch.npu.synchronize()
        for _ in range(REPEATS):
            _, use_aicpu, tile = self._invoke(case, data)
        torch.npu.synchronize()
        with self.record_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        **case.as_dict(),
                        "direction": self.direction,
                        "mode": self.mode,
                        "pipeline": "plan",
                        "use_aicpu": bool(use_aicpu),
                        "block_m": tile[0],
                        "block_n": tile[1],
                        "block_k": tile[2],
                        "warmup": WARMUP,
                        "repeats": REPEATS,
                    }
                )
                + "\n"
            )
        del data
        torch.npu.empty_cache()
