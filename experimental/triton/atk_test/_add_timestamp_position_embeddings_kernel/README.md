# `_add_timestamp_position_embeddings_kernel` 绠楀瓙鏂囨。

## 鏉ユ簮

`_add_timestamp_position_embeddings_kernel` 绠楀瓙婧愯嚜 NVIDIA recsys-examples 浠ｇ爜浠擄細

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_position.py>

## 绠楀瓙鍔熻兘

`_add_timestamp_position_embeddings_kernel` 鐢ㄤ簬鎶?timestamp/position embeddings 鍙犲姞鍒板簭鍒?embedding 涓婏紙HSTU 鐨勬椂闂?浣嶇疆缂栫爜铻嶅悎锛夛細

- `Out[n] = SeqEmb[n] + PosEmb[pos_inds[n]] + TsEmb[ts_inds[n]]`
- 浣嶇疆绱㈠紩 `pos_inds` 鎸夆€滃€掓暟浣嶇疆鈥濊绠楋紙绂?query 瓒婅繎鐨勮绱㈠紩瓒婂皬锛夛紝骞跺彈 `max_contextual_seq_len` 涓?`max_pos_ind` 閽充綅銆?- 鏃堕棿绱㈠紩 `ts_inds` 鐢?`query_time - ts + time_delta` 缁?`sqrt`锛堟垨 `log`锛夊垎妗跺緱鍒帮紝妗舵暟涓婇檺涓?`num_time_buckets`銆?- `TRAINING=True` 鏃舵妸璁＄畻鍑虹殑 `pos_inds` / `ts_inds` 鍐欏洖 `PosInds` / `TsInds`锛屼緵鍙嶅悜澶嶇敤銆?
## 杈撳叆

| 鍙傛暟 | 璇存槑 |
|------|------|
| `SeqEmb` | 灞曞钩 token 鐨勫簭鍒?embedding锛宻hape `(jagged_size, D)`锛宒type 鏀寔 fp32/fp16/bf16 |
| `Offsets` | 姣忎釜 batch 鐨勮捣濮嬭鍋忕Щ锛宻hape `(B+1,)` |
| `Lengths` | 姣忎釜 batch 鐨勫簭鍒楅暱搴︼紝shape `(B,)` |
| `PosEmb` | 浣嶇疆缂栫爜鏌ユ壘琛紝shape `(N_p, D)`锛宍N_p = max_contextual_seq_len + max_seq_len` |
| `TsEmb` | 鏃堕棿缂栫爜鏌ユ壘琛紝shape `(N_t, D)`锛宍N_t = num_time_buckets + 1` |
| `TS` | 姣忎釜 token 鐨勬椂闂存埑锛宻hape `(jagged_size,)` |
| `PosInds` / `TsInds` | 浣嶇疆/鏃堕棿妗剁储寮曡緭鍑猴紙TRAINING=True 鏃跺啓鍥烇級锛宻hape `(jagged_size,)` |
| `NumTargets` | 姣忎釜 batch 鐨?target 鏁帮紙HAS_MULTIPLE_TARGETS=True 鏃朵娇鐢級 |
| `num_time_buckets` | 鏃堕棿妗舵暟閲忥紙鍥哄畾 2048锛?|
| `max_contextual_seq_len` | 浣嶇疆妗跺熀鍑嗛暱搴?|
| `BLOCK_D` / `BLOCK_N` | 鍒?琛屽潡澶у皬锛坈onstexpr锛?|

## 杈撳嚭

`Out`锛氬舰鐘?`(jagged_size, D)`锛宒type 涓?`SeqEmb` 鐩稿悓銆俴ernel 閫?token 鍏ㄩ噺瑕嗙洊鍐欙紝鏃犻渶褰掗浂銆?
## 浣跨敤鏍蜂緥

### 鐩存帴璋冪敤 kernel

```python
import torch
import triton
from _add_timestamp_position_embeddings_kernel import _add_timestamp_position_embeddings_kernel

seq_emb = torch.randn(200, 16, dtype=torch.float32, device="npu")
offsets = torch.tensor([0, 100, 200], dtype=torch.int32, device="npu")
lengths = torch.tensor([100, 100], dtype=torch.int32, device="npu")
pos_emb = torch.randn(164, 16, dtype=torch.float32, device="npu")
ts_emb = torch.randn(2049, 16, dtype=torch.float32, device="npu")
ts = torch.rand(200, dtype=torch.float32, device="npu")
pos_inds = torch.empty(200, dtype=torch.int32, device="npu")
ts_inds = torch.empty(200, dtype=torch.int32, device="npu")
num_targets = torch.zeros(2, dtype=torch.int32, device="npu")
out = torch.empty(200, 16, dtype=torch.float32, device="npu")

grid = (2, triton.cdiv(100, 64))
_add_timestamp_position_embeddings_kernel[grid](
    SeqEmb=seq_emb,
    Offsets=offsets,
    Lengths=lengths,
    PosEmb=pos_emb,
    TsEmb=ts_emb,
    Out=out,
    TS=ts,
    PosInds=pos_inds,
    TsInds=ts_inds,
    NumTargets=num_targets,
    AUTOTUNE_MAX_SEQ_LEN=64,
    D=16,
    num_time_buckets=2048,
    time_bucket_increments=1.0,
    time_bucket_scale=1.0,
    time_delta=0.0,
    max_contextual_seq_len=64,
    max_pos_ind=164,
    stride_sn=seq_emb.stride(0),
    stride_pn=pos_emb.stride(0),
    stride_tn=ts_emb.stride(0),
    stride_on=out.stride(0),
    TRAINING=True,
    HAS_MULTIPLE_TARGETS=False,
    INTERLEAVE_TARGETS=False,
    TIME_BUCKET_FN="sqrt",
    BLOCK_D=16,
    BLOCK_N=64,
)
```

## 鍏抽敭绾︽潫

- `SeqEmb.ndim == 2` 涓?`jagged_size >= num_batches`锛堟瘡涓?batch 鑷冲皯涓€涓?token锛夈€?- 姣忓簭鍒楅暱搴?`<= 4096`锛氫繚璇?`(seq_len-1-n)^2` 鍦?fp32 涓簿纭〃绀猴紝`sqrt` 妗剁储寮曠‘瀹氭€ф垚绔嬨€?- `TIME_BUCKET_FN` 鍥哄畾 `"sqrt"`锛堢‘瀹氭€ц姹傦紝API 榛樿鍊硷級銆?- `TRAINING=True` 鏃?`PosInds` / `TsInds` 浼氳瑕嗙洊鍐欍€?
## 鐩綍鏂囦欢

```text
_add_timestamp_position_embeddings_kernel/
鈹溾攢鈹€ README.md
鈹溾攢鈹€ gpu_atk_test/
鈹?  鈹溾攢鈹€ _add_timestamp_position_embeddings_kernel.py       # 鍘熺増绠楀瓙 kernel 瀹炵幇
鈹?  鈹斺攢鈹€ _add_timestamp_position_embeddings_kernel_api.py   # GPU 绔?ATK 鑷畾涔?API 灏佽
鈹斺攢鈹€ npu_atk_test/
    鈹溾攢鈹€ _add_timestamp_position_embeddings_kernel.py       # NPU 閫傞厤浼樺寲鍚庣畻瀛?kernel 瀹炵幇
    鈹溾攢鈹€ _add_timestamp_position_embeddings_kernel_api.py   # NPU 绔?ATK 鑷畾涔?API 灏佽
    鈹溾攢鈹€ _add_timestamp_position_embeddings_kernel.yaml     # ATK 绠楀瓙鐢ㄤ緥鍙傛暟
    鈹溾攢鈹€ generate_add_timestamp_position_embeddings_kernel.py  # 闄愬埗鍙傛暟鐢熸垚鑴氭湰
    鈹斺攢鈹€ all__add_timestamp_position_embeddings_kernel.json    # 鎬ц兘娴嬭瘯鏍蜂緥鏁版嵁
```

## ATK 娴嬭瘯

娴嬭瘯鍦?NPU / GPU 涓ょ鍗忓悓杩涜锛?
- `npu_atk_test/` 閮ㄧ讲鍒?NPU 绔紱`gpu_atk_test/` 閮ㄧ讲鍒?GPU 绔€?- 绮惧害娴嬭瘯鐢ㄤ緥鍦?NPU 绔敓鎴愩€?- 鎬ц兘娴嬭瘯鐢ㄤ緥澶嶇敤 `npu_atk_test/all__add_timestamp_position_embeddings_kernel.json`銆?- GPU 绔惎鍔?ATK server锛孨PU 绔綔涓?ATK 瀹㈡埛绔媺璧风簿搴?鎬ц兘娴嬭瘯銆?
### 鐢熸垚绮惧害娴嬭瘯鏁版嵁

```bash
atk case -f _add_timestamp_position_embeddings_kernel.yaml -p generate_add_timestamp_position_embeddings_kernel.py
```

鐢熸垚鐨勭簿搴︽祴璇曠敤渚嬩綅浜?`result/_add_timestamp_position_embeddings_kernel/json/` 涓嬶紝鏂囦欢鍚嶄负 `all__add_timestamp_position_embeddings_kernel.json`銆?
### 杩滅 GPU 鏈嶅姟鍣ㄨ捣鐩戝惉鍛戒护

```bash
atk server --devices 0 --plugin_path /path/to/api鏂囦欢鎵€鍦ㄦ枃浠跺す
```

### 绮惧害娴嬭瘯

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all__add_timestamp_position_embeddings_kernel.json --task accuracy \
    -p _add_timestamp_position_embeddings_kernel_api.py
```

### 鎬ц兘娴嬭瘯

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all_add_timestamp_position_embeddings_kernel.json --task performance_device \
    -p _add_timestamp_position_embeddings_kernel_api.py
```
