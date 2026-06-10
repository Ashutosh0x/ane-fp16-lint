"""Rule database for fp16-unsafe operation patterns.

Each rule defines an operation type, its risk level, the exact fp16
overflow/underflow threshold, and the stable alternative formula.

These thresholds are derived from IEEE 754 half-precision specifications:
  - Max finite value: 65,504
  - exp() overflow: x > ln(65504) = 11.0903
  - Min positive normal: 6.10e-5
  - Min positive subnormal: 5.96e-8
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# IEEE 754 half-precision constants
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 6.103515625e-5  # 2^-14
FP16_MIN_SUBNORMAL = 5.960464477539063e-8  # 2^-24
FP16_EXP_OVERFLOW = math.log(FP16_MAX)  # 11.0903...
FP16_SQRT_MAX = math.sqrt(FP16_MAX)  # 255.93...


class RiskLevel(Enum):
    """Risk classification for fp16 safety."""
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class FP16Rule:
    """A rule describing an fp16-unsafe operation pattern."""
    op_type: str
    risk: RiskLevel
    threshold: str
    description: str
    stable_alternative: str
    pytorch_amp_class: str = "UNKNOWN"
    category: str = "uncategorized"
    overflow_value: Optional[float] = None

    def check(self, op) -> bool:
        """Check if this rule applies to the given MIL operation."""
        return op.op_type == self.op_type


# ============================================================
# Category A: Exponential Overflow (exp(x) > 65504)
# ============================================================

SOFTPLUS_RULE = FP16Rule(
    op_type="softplus",
    risk=RiskLevel.CRITICAL,
    threshold=f"x > {FP16_EXP_OVERFLOW:.2f}",
    description="log(1+exp(x)) overflows when exp(x) > 65504. "
                "ANE exhibits a cliff at x ≈ 10.4 due to internal rounding.",
    stable_alternative="max(x, 0) + log(1 + exp(-|x|))",
    pytorch_amp_class="FP32_ONLY",
    category="exponential_overflow",
    overflow_value=FP16_EXP_OVERFLOW,
)

REDUCE_LOG_SUM_EXP_RULE = FP16Rule(
    op_type="reduce_log_sum_exp",
    risk=RiskLevel.CRITICAL,
    threshold="x > ln(65504/C) where C = channel count",
    description="log(Σ exp(x_i)) overflows when Σ exp(x_i) > 65504. "
                "For C=32 channels at equal value v: overflow at v > 7.63.",
    stable_alternative="max(x) + log(Σ exp(x_i - max(x)))",
    pytorch_amp_class="FP32_ONLY",
    category="exponential_overflow",
    overflow_value=7.63,
)

EXP_RULE = FP16Rule(
    op_type="exp",
    risk=RiskLevel.HIGH,
    threshold=f"x > {FP16_EXP_OVERFLOW:.2f}",
    description="exp(x) overflows to inf in fp16.",
    stable_alternative="Check if input is bounded by a stable decomposition.",
    pytorch_amp_class="FP32_ONLY",
    category="exponential_overflow",
    overflow_value=FP16_EXP_OVERFLOW,
)

# ============================================================
# Category B: Probability Underflow (softmax → 0 → log(-inf))
# ============================================================

SOFTMAX_RULE = FP16Rule(
    op_type="softmax",
    risk=RiskLevel.MEDIUM,
    threshold="max logit spread > 16",
    description="Small softmax probabilities underflow to 0 in fp16. "
                "Dangerous when followed by log() (produces -inf).",
    stable_alternative="Use fused log_softmax: x - max(x) - log(Σ exp(x - max(x)))",
    pytorch_amp_class="FP32_ONLY",
    category="probability_underflow",
)

LOG_RULE = FP16Rule(
    op_type="log",
    risk=RiskLevel.HIGH,
    threshold=f"x < {FP16_MIN_NORMAL:.2e} (below min normal)",
    description="log(x) diverges to -inf for x approaching 0. "
                "Critical when preceded by softmax (underflowed probabilities).",
    stable_alternative="Fuse with preceding softmax into log_softmax decomposition.",
    pytorch_amp_class="FP32_ONLY",
    category="probability_underflow",
    overflow_value=FP16_MIN_NORMAL,
)

# ============================================================
# Category C: Squared-Value Overflow (x² > 65504)
# ============================================================

REDUCE_L2_NORM_RULE = FP16Rule(
    op_type="reduce_l2_norm",
    risk=RiskLevel.MEDIUM,
    threshold=f"|x| > {FP16_SQRT_MAX:.1f}",
    description="L2 norm squares inputs: x² overflows at |x| > 255.9 in fp16.",
    stable_alternative="Scale by max(|x|) before squaring, then rescale.",
    pytorch_amp_class="FP32_ONLY",
    category="squared_overflow",
    overflow_value=FP16_SQRT_MAX,
)

# ============================================================
# Category D: Reciprocal / Division
# ============================================================

RECIPROCAL_RULE = FP16Rule(
    op_type="reciprocal",
    risk=RiskLevel.MEDIUM,
    threshold=f"|x| > {FP16_MAX}",
    description="1/x underflows to 0 for large x in fp16.",
    stable_alternative="Check if x is bounded.",
    pytorch_amp_class="FP32_ONLY",
    category="reciprocal_underflow",
)

RSQRT_RULE = FP16Rule(
    op_type="rsqrt",
    risk=RiskLevel.MEDIUM,
    threshold=f"x > {FP16_MAX}",
    description="1/sqrt(x) underflows for large x in fp16.",
    stable_alternative="Scale input before computing rsqrt.",
    pytorch_amp_class="FP32_ONLY",
    category="reciprocal_underflow",
)

POW_RULE = FP16Rule(
    op_type="pow",
    risk=RiskLevel.MEDIUM,
    threshold="base^exp > 65504",
    description="Power operation overflows quickly in fp16. "
                "x³ overflows at |x| > 40.3, x⁴ at |x| > 16.",
    stable_alternative="Check exponent value; clamp if needed.",
    pytorch_amp_class="FP32_ONLY",
    category="polynomial_overflow",
)

# ============================================================
# Safe Operations (for completeness)
# ============================================================

SAFE_OPS = {
    "relu", "leaky_relu", "sigmoid", "tanh", "conv", "linear",
    "matmul", "add", "sub", "mul", "concat", "reshape", "transpose",
    "flatten", "slice_by_index", "pad", "pool", "batch_norm",
    "instance_norm", "layer_norm", "group_norm", "dropout",
    "split", "stack", "gather", "scatter",
}


# ============================================================
# Master Rule Database
# ============================================================

UNSAFE_PATTERNS: dict[str, FP16Rule] = {
    rule.op_type: rule
    for rule in [
        SOFTPLUS_RULE,
        REDUCE_LOG_SUM_EXP_RULE,
        EXP_RULE,
        SOFTMAX_RULE,
        LOG_RULE,
        REDUCE_L2_NORM_RULE,
        RECIPROCAL_RULE,
        RSQRT_RULE,
        POW_RULE,
    ]
}
