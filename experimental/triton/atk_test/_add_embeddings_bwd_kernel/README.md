# `_add_embeddings_bwd_kernel` 绠楀瓙鏂囨。

## 鏉ユ簮

`_add_embeddings_bwd_kernel` 绠楀瓙婧愯嚜 NVIDIA recsys-examples 浠ｇ爜浠擄細

<https://github.com/NVIDIA/recsys-examples/blob/main/examples/hstu/ops/triton_ops/triton_position.py>

## 绠楀瓙鍔熻兘

`_add_embeddings_bwd_kernel` 鐢ㄤ簬灏?position/timestamp embeddings 鐨?*鍙嶅悜姊害**鎸夊綊骞?key 鍋?scatter-add锛堝師瀛愮疮鍔狅級锛屾瘡缁勫唴鍚屼竴 key 鐨?token 姊害姹傚拰銆?

杈撳叆涓哄睍骞?token 鐨勪笂娓告搴?`In` 涓庢帓搴忓垎娈电殑 `KeyInds`/`ValueInds` 绱㈠紩瀵广€傚鐩稿悓 key 鐨勫垎缁勶紝缁勫唴 `In[ValueInds[i]]` 鍚勮鍏堟湰鍦扮疮鍔狅紝鍐?`atomic_add`锛坮elaxed锛夋眹鍏?`Out[KeyInds[i]]`銆?

`KeyInds`/`ValueInds` 鐢?`num_keys` 涓?`jagged_size` 纭畾鎬ф帹瀵硷紙`ValueInds[i] = i`锛宍KeyInds[i] = i * num_keys // jagged_size`锛夛紝淇濊瘉 GPU/NPU 涓ょ杈撳叆涓€鑷淬€?

`Out` 浠ラ浂鍒濆鍖栵紝`atomic_add` 绱姞瑕佹眰姣忔璋冪敤浠庨浂寮€濮嬨€?

## 杈撳叆

| 鍙傛暟 | 璇存槑 |
|------|------|
| `In` | 灞曞钩 token 鐨勪笂娓告搴︼紝shape `(jagged_size, D)`锛宒type 鏀寔 fp32/fp16/bf16 |
| `KeyInds` | 姣忎釜 token 褰掑苟鍒扮殑 key锛堝凡鎺掑簭鍒嗘锛屽€煎煙 `[0, num_keys)`锛?|
| `ValueInds` | 琚疮鍔犵殑琛岀储寮曪紝`ValueInds[i] = i` |
| `num_keys` | 杈撳嚭 key 鏁帮紙`Out` 琛屾暟锛夛紝瑕佹眰 `jagged_size >= num_keys` |
| `Out` | 杈撳嚭姊害锛坋mbedding 琛ㄥ悇琛屾搴︼級锛宻hape `(num_keys, D)`锛屼互闆跺垵濮嬪寲 |
| `BLOCK_D` | 鍒楀潡澶у皬锛坈onstexpr锛夛紝鍙?`next_power_of_2(D)` |
| `BLOCK` | 琛屽潡澶у皬锛坈onstexpr锛?|

## 杈撳嚭

`Out`锛氬舰鐘?`(num_keys, D)`锛屽叾涓?`num_keys` 鐢辫緭鍏ユ寚瀹氾紝dtype 涓?`In` 鐩稿悓銆俙atomic_add` 绱姞瑕佹眰姣忔璋冪敤浠庨浂寮€濮嬶紙API 鍦?`__call__` 鍓?`zero_()`锛夈€?

## 浣跨敤鏍蜂緥

### 鐩存帴璋冪敤 kernel

```python
import torch
import triton
from _add_embeddings_bwd_kernel import _add_embeddings_bwd_kernel

in_data = torch.randn(200, 16, dtype=torch.float32, device="npu")
num_keys = 4
key_inds = ((torch.arange(200) * num_keys) // 200).to(torch.int32).to("npu")
value_inds = torch.arange(200, dtype=torch.int32, device="npu")
out = torch.zeros(4, 16, dtype=torch.float32, device="npu")

grid = (triton.cdiv(200, 128),)
_add_embeddings_bwd_kernel[grid](
    In=in_data,
    KeyInds=key_inds,
    ValueInds=value_inds,
    Out=out,
    AUTOTUNE_MAX_SEQ_LEN=1024,
    B=1,
    AUTOTUNE_B=1,
    D=16,
    jagged_size=200,
    stride_in=in_data.stride(0),
    stride_on=out.stride(0),
    BLOCK_D=16,
    BLOCK=128,
)
```

## 鍏抽敭绾︽潫

- In.ndim == 2 涓?jagged_size >= num_keys锛堟瘡涓?key 鑷冲皯琚竴涓?token 瀵诲潃锛夈€?
- Out 浠?torch.zeros 鍒濆鍖栵紝atomic_add 闇€瑕侀浂璧风偣锛涘娆¤皟鐢ㄥ墠闇€褰掗浂銆?
- KeyInds 宸叉帓搴忥紙API 鎸?i * num_keys // jagged_size 纭畾鎬х敓鎴愶級锛岀粍鍐呯疮鍔犳墠鎴愮珛銆?

## 鐩綍鏂囦欢

```text
_add_embeddings_bwd_kernel/
鈹溾攢鈹€ README.md
鈹溾攢鈹€ gpu_atk_test/
鈹?  鈹溾攢鈹€ _add_embeddings_bwd_kernel.py              # 鍘熺増绠楀瓙 kernel 瀹炵幇
鈹?  鈹斺攢鈹€ _add_embeddings_bwd_kernel_api.py          # GPU 绔?ATK 鑷畾涔?API 灏佽
鈹斺攢鈹€ npu_atk_test/
    鈹溾攢鈹€ _add_embeddings_bwd_kernel.py              # NPU 閫傞厤浼樺寲鍚庣畻瀛?kernel 瀹炵幇
    鈹溾攢鈹€ _add_embeddings_bwd_kernel_api.py          # NPU 绔?ATK 鑷畾涔?API 灏佽
    鈹溾攢鈹€ _add_embeddings_bwd_kernel.yaml            # ATK 绠楀瓙鐢ㄤ緥鍙傛暟
    鈹溾攢鈹€ generate_add_embeddings_bwd_kernel.py      # 闄愬埗鍙傛暟鐢熸垚鑴氭湰
    鈹斺攢鈹€ all__add_embeddings_bwd_kernel.json        # 鎬ц兘娴嬭瘯鏍蜂緥鏁版嵁
 ```

## ATK 娴嬭瘯

娴嬭瘯鍦?NPU / GPU 涓ょ鍗忓悓杩涜锛?

- `npu_atk_test/` 閮ㄧ讲鍒?NPU 绔紱`gpu_atk_test/` 閮ㄧ讲鍒?GPU 绔€?
- 绮惧害娴嬭瘯鐢ㄤ緥鍦?NPU 绔敓鎴愩€?
- 鎬ц兘娴嬭瘯鐢ㄤ緥澶嶇敤 `npu_atk_test/all__add_embeddings_bwd_kernel.json`銆?
- GPU 绔惎鍔?ATK server锛孨PU 绔綔涓?ATK 瀹㈡埛绔媺璧风簿搴?鎬ц兘娴嬭瘯銆?

### 鐢熸垚绮惧害娴嬭瘯鏁版嵁

```bash
atk case -f _add_embeddings_bwd_kernel.yaml -p generate_add_embeddings_bwd_kernel.py
```

鐢熸垚鐨勭簿搴︽祴璇曠敤渚嬩綅浜?`result/_add_embeddings_bwd_kernel/json/` 涓嬶紝鏂囦欢鍚嶄负 `all__add_embeddings_bwd_kernel.json`銆?

### 杩滅 GPU 鏈嶅姟鍣ㄨ捣鐩戝惉鍛戒护

```bash
atk server --devices 0 --plugin_path /path/to/api鏂囦欢鎵€鍦ㄦ枃浠跺す
```

### 绮惧害娴嬭瘯

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all__add_embeddings_bwd_kernel.json --task accuracy \
    -p _add_embeddings_bwd_kernel_api.py
```

### 鎬ц兘娴嬭瘯

```bash
atk node --backend triton --devices 0 node --backend npu --devices 0 \
    task -c all_add_embeddings_bwd_kernel.json --task performance_device \
    -p _add_embeddings_bwd_kernel_api.py
```
