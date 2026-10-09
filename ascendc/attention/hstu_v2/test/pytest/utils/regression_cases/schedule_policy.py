import pathlib
import subprocess
import tempfile


REPO_ROOT = pathlib.Path(__file__).resolve().parents[7]
HEADER_DIR = REPO_ROOT / "ascendc/attention/hstu_v2/common"


def test_backend_neutral_schedule_policy():
    source = r'''
#include <cassert>
#include <cstdint>
#include <vector>
#include "hstu_schedule_policy.h"

using hstu::schedule::BoundaryPolicy;
using hstu::schedule::CoreRange;
using hstu::schedule::UniformCoreRange;
using hstu::schedule::WeightedCoreRangeBuilder;
using hstu::schedule::ShouldUseAicpuSchedule;
using hstu::schedule::L2BudgetBytes;

static CoreRange Weighted(const std::vector<uint64_t>& weights, uint32_t core,
                          uint32_t cores, BoundaryPolicy policy)
{
    uint64_t total = 0;
    for (auto weight : weights) total += weight;
    WeightedCoreRangeBuilder builder(weights.size(), total, core, cores, policy);
    for (auto weight : weights) builder.Push(weight);
    return builder.Finish();
}

static void AssertCover(const std::vector<uint64_t>& weights, uint32_t cores,
                        BoundaryPolicy policy)
{
    uint32_t cursor = 0;
    for (uint32_t core = 0; core < cores; ++core) {
        const auto range = Weighted(weights, core, cores, policy);
        assert(range.beginBlock == cursor);
        assert(range.endBlock >= range.beginBlock);
        cursor = range.endBlock;
    }
    assert(cursor == weights.size());
}

int main()
{
    assert(!ShouldUseAicpuSchedule(80, 100));
    assert(ShouldUseAicpuSchedule(81, 100));
    assert(!ShouldUseAicpuSchedule(50, 100, 1, 2));
    assert(ShouldUseAicpuSchedule(51, 100, 1, 2));
    assert(L2BudgetBytes(100) == 80);
    assert(L2BudgetBytes(101) == 80);
    // FastSplitCore-compatible quotient/remainder boundaries.
    const CoreRange u0 = UniformCoreRange(10, 0, 4);
    const CoreRange u1 = UniformCoreRange(10, 1, 4);
    const CoreRange u2 = UniformCoreRange(10, 2, 4);
    const CoreRange u3 = UniformCoreRange(10, 3, 4);
    assert(u0.beginBlock == 0 && u0.endBlock == 3);
    assert(u1.beginBlock == 3 && u1.endBlock == 6);
    assert(u2.beginBlock == 6 && u2.endBlock == 8);
    assert(u3.beginBlock == 8 && u3.endBlock == 10);
    const CoreRange empty = UniformCoreRange(2, 3, 4);
    assert(empty.beginBlock == 2 && empty.endBlock == 2);
    // guide_new1 short-sequence fast path partitions batch*heads without
    // scanning sequence offsets.  Every logical (batch, head) block is
    // covered exactly once, including the uneven quotient/remainder tail.
    uint32_t shortCursor = 0;
    const uint32_t shortBlocks = 2048U * 2U;
    for (uint32_t core = 0; core < 72U; ++core) {
        const CoreRange range = UniformCoreRange(shortBlocks, core, 72U);
        assert(range.beginBlock == shortCursor);
        shortCursor = range.endBlock;
    }
    assert(shortCursor == shortBlocks);

    // A heavy row demonstrates the two explicitly supported boundary modes.
    const std::vector<uint64_t> heavy{1, 1, 10, 1, 1};
    const CoreRange post0 = Weighted(heavy, 0, 2, BoundaryPolicy::REMAINING_POST_ROW);
    const CoreRange near0 = Weighted(heavy, 0, 2, BoundaryPolicy::NEAREST);
    assert(post0.beginBlock == 0 && post0.endBlock == 3);
    assert(near0.beginBlock == 0 && near0.endBlock == 2);

    // Zero sparse workload retains geometric ownership for device zero-fill.
    const std::vector<uint64_t> zero(5, 0);
    for (uint32_t core = 0; core < 8; ++core) {
        const auto got = Weighted(zero, core, 8, BoundaryPolicy::NEAREST);
        const auto want = UniformCoreRange(5, core, 8);
        assert(got.beginBlock == want.beginBlock && got.endBlock == want.endBlock);
    }

    AssertCover({4, 4, 2, 9, 1, 3, 8}, 4, BoundaryPolicy::FIXED_POST_ROW);
    AssertCover({4, 4, 2, 9, 1, 3, 8}, 4, BoundaryPolicy::REMAINING_POST_ROW);
    AssertCover({4, 4, 2, 9, 1, 3, 8}, 4, BoundaryPolicy::NEAREST);
    AssertCover({1, 1}, 8, BoundaryPolicy::FIXED_POST_ROW);
    AssertCover({1, 1}, 8, BoundaryPolicy::REMAINING_POST_ROW);
    AssertCover({1, 1}, 8, BoundaryPolicy::NEAREST);
    AssertCover({UINT64_C(1) << 40, UINT64_C(1) << 40, 1}, 3,
                BoundaryPolicy::FIXED_POST_ROW);
    return 0;
}
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        src = pathlib.Path(tmpdir) / "schedule_policy_test.cpp"
        exe = pathlib.Path(tmpdir) / "schedule_policy_test"
        src.write_text(source)
        subprocess.run(
            ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{HEADER_DIR}", str(src), "-o", str(exe)],
            check=True,
        )
        subprocess.run([str(exe)], check=True)
