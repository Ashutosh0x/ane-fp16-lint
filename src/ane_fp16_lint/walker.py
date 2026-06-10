"""Walk MIL computation graphs and match against fp16-unsafe patterns."""

from dataclasses import dataclass, field
from typing import Any

from ane_fp16_lint.rules import UNSAFE_PATTERNS, SAFE_OPS, FP16Rule, RiskLevel


@dataclass
class Finding:
    """A single finding from the lint walk."""
    op_name: str
    op_type: str
    rule: FP16Rule
    function_name: str = "main"
    block_depth: int = 0

    @property
    def risk(self) -> RiskLevel:
        return self.rule.risk

    def __str__(self) -> str:
        icon = {
            RiskLevel.CRITICAL: "❌",
            RiskLevel.HIGH: "🔶",
            RiskLevel.MEDIUM: "⚠️",
            RiskLevel.LOW: "ℹ️",
            RiskLevel.SAFE: "✅",
        }[self.risk]
        return (
            f"{icon} {self.op_type:<25s} — {self.rule.risk.value}: "
            f"{self.rule.description[:80]}"
        )


def walk_mil_graph(program: Any) -> list[Finding]:
    """Walk the MIL program graph and flag fp16-unsafe operations.

    Args:
        program: A coremltools MIL Program object.

    Returns:
        A list of Finding objects for each unsafe operation found.
    """
    findings: list[Finding] = []

    for func_name, func in program.functions.items():
        _walk_block(func, func_name, findings, depth=0)

    return findings


def _walk_block(block: Any, func_name: str, findings: list[Finding], depth: int):
    """Recursively walk a MIL block and its nested blocks."""
    for op in block.operations:
        op_type = op.op_type

        # Check against unsafe patterns
        if op_type in UNSAFE_PATTERNS:
            rule = UNSAFE_PATTERNS[op_type]
            findings.append(Finding(
                op_name=getattr(op, "name", op_type),
                op_type=op_type,
                rule=rule,
                function_name=func_name,
                block_depth=depth,
            ))

        # Check for softmax → log pattern (fuse opportunity)
        if op_type == "softmax":
            _check_softmax_log_pattern(op, func_name, findings, depth)

        # Recurse into nested blocks (e.g., cond, while_loop)
        for block_attr in ("body", "cond", "blocks"):
            nested = getattr(op, block_attr, None)
            if nested is not None:
                if isinstance(nested, (list, tuple)):
                    for sub_block in nested:
                        _walk_block(sub_block, func_name, findings, depth + 1)
                else:
                    _walk_block(nested, func_name, findings, depth + 1)


def _check_softmax_log_pattern(softmax_op, func_name, findings, depth):
    """Detect softmax followed by log (should be fused log_softmax)."""
    for output in softmax_op.outputs:
        for child_op in getattr(output, "child_ops", []):
            if child_op.op_type == "log":
                findings.append(Finding(
                    op_name=f"{softmax_op.name}→{child_op.name}",
                    op_type="softmax→log",
                    rule=FP16Rule(
                        op_type="softmax→log",
                        risk=RiskLevel.CRITICAL,
                        threshold="softmax outputs < 6.1e-5 underflow to 0",
                        description="softmax followed by log should be fused into "
                                    "log_softmax decomposition to prevent log(0)=-inf.",
                        stable_alternative="x - max(x) - log(Σ exp(x - max(x)))",
                        pytorch_amp_class="FP32_ONLY",
                        category="fuse_opportunity",
                    ),
                    function_name=func_name,
                    block_depth=depth,
                ))


def get_summary(findings: list[Finding]) -> dict[str, int]:
    """Summarize findings by risk level."""
    summary = {level.value: 0 for level in RiskLevel}
    for f in findings:
        summary[f.risk.value] += 1
    return summary
