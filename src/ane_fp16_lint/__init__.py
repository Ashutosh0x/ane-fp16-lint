"""ane-fp16-lint: Static analysis for fp16 safety in Core ML models."""

__version__ = "0.1.0"

from ane_fp16_lint.rules import UNSAFE_PATTERNS
from ane_fp16_lint.walker import walk_mil_graph
from ane_fp16_lint.reporter import LintReport

__all__ = ["lint_model", "UNSAFE_PATTERNS", "walk_mil_graph", "LintReport"]


def lint_model(model_path: str, *, fix: bool = False) -> "LintReport":
    """Lint a Core ML model for fp16-unsafe operations.

    Args:
        model_path: Path to a .mlpackage file.
        fix: If True, apply stable decompositions to unsafe ops.

    Returns:
        A LintReport with findings.
    """
    from ane_fp16_lint.loader import load_mil_program
    from ane_fp16_lint.walker import walk_mil_graph

    program = load_mil_program(model_path)
    findings = walk_mil_graph(program)

    if fix:
        from ane_fp16_lint.fixer import apply_fixes
        program = apply_fixes(program, findings)

    return LintReport(findings=findings, program=program, model_path=model_path)
