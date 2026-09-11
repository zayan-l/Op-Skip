<h1 align="center">
  <a href="opskip.pdf"><img src="assets/opskip-logo.png" alt="Op-Skip logo" width="48" align="absmiddle"></a>
  Op-Skip
</h1>

Code for **Attend, Transform, or Silence: Operator-Level Visual Skipping for
Efficient Multimodal LLM Inference** ([paper](https://arxiv.org/abs/2606.31903)).

Op-Skip accelerates multimodal prefill by selectively skipping visual Attention
updates, FFN updates, or both. It retains the visual sequence and full per-layer
KV cache, without training or changing model weights.

## 🔥 News

- `2026.09.11` 📦 Added Op-Skip implementations, preset policies, and evaluation scripts for Qwen2.5-VL, Qwen3-VL, LLaVA-1.5, and LLaVA-NeXT.
- `2026.06.30` 📄 Our paper [Attend, Transform, or Silence](https://arxiv.org/abs/2606.31903) is available on arXiv!

## 🛠️ Installation

Use a separate environment for each model family. From the repository root:

```bash
# Choose: qwen3_vl, qwen2_5_vl, llava, llava_next
FAMILY=qwen3_vl
python -m pip install -r requirements/${FAMILY}.txt
python -m pip install --no-deps -e .
```

| Family | Checkpoint | Python | Transformers |
|---|---|---|---|
| `qwen3_vl` | `Qwen/Qwen3-VL-8B-Instruct` | 3.12 | 5.5.4 |
| `qwen2_5_vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | 3.12 | 4.51.3 |
| `llava` | `liuhaotian/llava-v1.5-7b` | 3.12 | 4.37.2 |
| `llava_next` | `liuhaotian/llava-v1.6-vicuna-7b` | 3.10 | Commit pinned in requirements |

For LLaVA, also install the matching model source in its own environment:

```bash
# LLaVA-1.5:
python -m pip install --no-deps 'git+https://github.com/haotian-liu/LLaVA.git'
# LLaVA-NeXT:
python -m pip install --no-deps 'git+https://github.com/LLaVA-VL/LLaVA-NeXT.git'
```

These projects share the `llava` namespace; do not install both in one environment.
Keep the pinned dependencies and record the model-source commit used for reproduction.

## 🚀 Quick start

```python
import torch
from transformers import Qwen3VLForConditionalGeneration
from opskip import apply_opskip

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="sdpa",
).eval()

apply_opskip(model, "configs/qwen3_vl/preset20.json")
# Prepare inputs with the model processor, then call model.generate as usual.
```

For a complete image example:

```bash
python examples/infer.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --policy configs/qwen3_vl/preset20.json \
  --image /path/to/image.jpg \
  --prompt 'What is shown in this image?'
```

Supported setup: the dense checkpoints above, batch size 1, greedy generation,
and SDPA. `remove_opskip(model)` restores the patched instance.

## 🎛️ Policies

Select a JSON file from `configs/<family>/`. The preset number counts selected
layers, not skipped operators or a FLOPs percentage. Layer indices are zero-based.

| Family | Presets |
|---|---|
| Qwen3-VL | 12, 16, 20, 24, 28, 32, 36 |
| Qwen2.5-VL | 8, 12, 16, 20, 24, 28 |
| LLaVA-1.5 / LLaVA-NeXT | 8, 12, 16, 20, 24, 28, 32 |

`attention_only` keeps visual Attention updates; `ffn_only` keeps visual FFN
updates; `freeze` skips both. Unlisted layers run normally. In the released
policies, `freeze` also freezes the text prefix before the visual span, updating
only the text suffix while retaining K/V for every token.

## 📊 Evaluation

Install a clean upstream **lmms-eval 0.6.1** in the chosen model environment:

```bash
python -m pip install -r requirements/${FAMILY}.txt -r requirements/eval.txt
python -m pip install --no-deps 'lmms-eval==0.6.1'
```

The separate runtime requirements and `--no-deps` preserve model-specific versions.
Prepare the datasets following [lmms-eval's task instructions](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/v0.6.1/lmms_eval/tasks);
use `HF_HOME` / `HF_DATASETS_CACHE` to reuse existing caches.

```bash
bash scripts/eval_qwen3_vl.sh \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --policy configs/qwen3_vl/preset20.json \
  --output outputs/qwen3_preset20
```

For other models, use `eval_qwen2_5_vl.sh`, `eval_llava.sh`, or
`eval_llava_next.sh` with the corresponding checkpoint and policy.

- `--policy off`: Vanilla baseline.
- `--tasks textvqa_val,pope`: select tasks.
- `--dry-run`: inspect resolved settings without loading a model.

Use the same checkpoint, data, prompts, and image settings for comparisons.
Each run saves its policy and evaluator arguments in `opskip_config.json`.

## ⚡ Prefill speedup

Optimized Op-Skip versus Vanilla on A800-SXM4-80GB, batch size 1.
Each cell is the **median of ten task-level speedup ratios at that preset**.

| Preset | Qwen2.5-VL-7B | Qwen3-VL-8B | LLaVA-1.5-7B | LLaVA-NeXT-7B |
|---:|---:|---:|---:|---:|
| 8 | 1.278× | — | 1.182× | 1.252× |
| 12 | 1.296× | 1.158× | 1.219× | 1.343× |
| 16 | 1.358× | 1.187× | 1.264× | 1.447× |
| 20 | 1.403× | 1.205× | 1.301× | 1.566× |
| 24 | 1.428× | 1.216× | 1.361× | 1.708× |
| 28 | 1.458× | 1.228× | 1.418× | 1.877× |
| 32 | — | 1.247× | 1.488× | 2.084× |
| 36 | — | 1.269× | — | — |

## ⏱️ Latency benchmark

To measure the current code on your hardware:

```bash
python scripts/benchmark_latency.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --policy configs/qwen3_vl/preset20.json \
  --image /path/to/image.jpg \
  --warmup 3 --repeats 10 \
  --output outputs/latency.json
```

## 📄 Citation

```bibtex
@article{luo2026attend,
  title={Attend, Transform, or Silence: Operator-Level Visual Skipping for Efficient Multimodal LLM Inference},
  author={Luo, Zhaoyang and Dong, Runmin and Yang, Miao and Wei, Fan and Lai, Yushan and Luo, Bin and Fu, Haohuan},
  journal={arXiv preprint arXiv:2606.31903},
  year={2026}
}
```

## 🙏 Acknowledgements

Our code builds on the excellent open-source contributions of Transformers, Qwen, LLaVA, LLaVA-NeXT, and lmms-eval. We express our sincere gratitude to their authors and contributors for making this work possible.
