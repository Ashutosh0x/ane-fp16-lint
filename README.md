# ane-fp16-lint

**Detect silent fp16 overflow/underflow in Core ML models before they corrupt inference on Apple Neural Engine.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## The Problem

Apple's Neural Engine (ANE) executes neural network inference exclusively in **IEEE 754 half-precision (fp16)**, with a maximum representable value of **65,504**. Many common operations silently overflow or underflow at this precision, producing corrupted outputs with no error message.

With Apple's AFM 3 Core Advanced deploying a **20-billion-parameter sparse model** on ANE via fp16, numerical safety is no longer an academic concern — it is a production requirement affecting over one billion devices.

## What This Tool Does

```bash
$ ane-fp16-lint model.mlpackage

  [SAFE]     conv2d              — Linear, bounded by weights
  [SAFE]     relu                — Monotone, non-amplifying
  [CRITICAL] softplus            — OVERFLOW RISK: exp(x) overflows at x > 11.09
  [CRITICAL] reduce_log_sum_exp  — OVERFLOW RISK: sum(exp(x)) overflows at x > 7.63 (C=32)
  [WARNING]  softmax             — UNDERFLOW RISK: small probabilities flush to 0
  [CRITICAL] log                 — DIVERGENCE RISK: log(0) = -inf after softmax underflow

  3 CRITICAL | 1 WARNING | 2 SAFE
```

## Installation

```bash
pip install ane-fp16-lint
```

## Quick Start

```python
from ane_fp16_lint import lint_model

report = lint_model("path/to/model.mlpackage")
report.print()

# Auto-fix mode (applies stable decompositions)
fixed_model = lint_model("model.mlpackage", fix=True)
fixed_model.save("model_fixed.mlpackage")
```

## Detected Patterns

| Operation | Risk | Threshold | Stable Alternative |
|-----------|------|-----------|-------------------|
| `softplus` | CRITICAL | x > 11.09 | max(x,0) + log(1+exp(-\|x\|)) |
| `reduce_log_sum_exp` | CRITICAL | x > ln(65504/C) | max(x) + log(sum(exp(x-max(x)))) |
| `log_softmax` | CRITICAL | softmax underflow | x - max(x) - log(sum(exp(x-max(x)))) |
| `exp` | HIGH | x > 11.09 | Check if bounded |
| `log` | HIGH | x <= 0 | Check preceding op |
| `softmax` | MEDIUM | large logit spread | Use fused log_softmax |
| `reduce_l2_norm` | MEDIUM | \|x\| > 256 | Scale by max(\|x\|) |
| `pow` | MEDIUM | base^exp > 65504 | Check exponent value |

## How It Works

1. **Load**: Parse `.mlpackage` and extract the MIL (Model Intermediate Language) graph
2. **Walk**: Traverse operations in topological order
3. **Match**: Check each operation against the rule database of fp16-unsafe patterns
4. **Report**: Classify as SAFE / WARNING / CRITICAL with exact thresholds
5. **Fix** (optional): Apply algebraically equivalent stable decompositions

## Background

This tool is based on research identifying [5 silent fp16 overflow vulnerabilities](https://github.com/apple/coremltools/pull/2725) in Apple's `coremltools` converter, with production fixes currently under review by Apple's Core ML team:

- [PR #2725](https://github.com/apple/coremltools/pull/2725) — Softplus/Mish stable decomposition
- [PR #2726](https://github.com/apple/coremltools/pull/2726) — LogSumExp max-shift decomposition
- [PR #2727](https://github.com/apple/coremltools/pull/2727) — LogSoftmax/LogCumSumExp decomposition

PyTorch's AMP (Automatic Mixed Precision) classifies `exp`, `log`, `softmax`, and `log_softmax` as **FP32-only operations**, confirming they are fundamentally unsafe in half-precision.

## Citation

```bibtex
@software{ane_fp16_lint,
  author = {Singh, Ashutosh Kumar},
  title = {ane-fp16-lint: Static Analysis for FP16 Safety in Core ML Models},
  year = {2026},
  url = {https://github.com/Ashutosh0x/ane-fp16-lint}
}
```

## License

MIT
