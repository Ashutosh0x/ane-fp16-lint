"""Rich terminal reporter for lint findings."""

from dataclasses import dataclass, field
from typing import Any

from ane_fp16_lint.walker import Finding, get_summary
from ane_fp16_lint.rules import RiskLevel


@dataclass
class LintReport:
    """Complete lint report for a Core ML model."""
    findings: list[Finding]
    program: Any = None
    model_path: str = ""

    @property
    def is_safe(self) -> bool:
        return all(f.risk == RiskLevel.SAFE for f in self.findings)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.risk == RiskLevel.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.risk == RiskLevel.HIGH)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.risk == RiskLevel.MEDIUM)

    def print(self, verbose: bool = False):
        """Print the lint report to the terminal."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            self._print_rich(verbose)
        except ImportError:
            self._print_plain(verbose)

    def _print_rich(self, verbose: bool = False):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # Header
        console.print()
        console.print(
            Panel(
                f"[bold]ane-fp16-lint[/bold] — {self.model_path}",
                subtitle=f"{len(self.findings)} findings",
            )
        )

        if not self.findings:
            console.print("  [green]✅ No fp16-unsafe operations detected.[/green]")
            console.print()
            return

        # Findings table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Risk", width=10)
        table.add_column("Operation", width=25)
        table.add_column("Threshold", width=25)
        table.add_column("Description", width=50)

        risk_style = {
            RiskLevel.CRITICAL: "[bold red]❌ CRITICAL[/]",
            RiskLevel.HIGH: "[yellow]🔶 HIGH[/]",
            RiskLevel.MEDIUM: "[cyan]⚠️  MEDIUM[/]",
            RiskLevel.LOW: "[dim]ℹ️  LOW[/]",
            RiskLevel.SAFE: "[green]✅ SAFE[/]",
        }

        for f in sorted(self.findings, key=lambda x: list(RiskLevel).index(x.risk)):
            table.add_row(
                risk_style[f.risk],
                f"[bold]{f.op_type}[/]",
                f.rule.threshold[:25],
                f.rule.description[:50],
            )
            if verbose:
                table.add_row(
                    "", "", "Fix →",
                    f"[green]{f.rule.stable_alternative}[/]",
                )

        console.print(table)

        # Summary
        console.print()
        parts = []
        if self.critical_count:
            parts.append(f"[bold red]{self.critical_count} CRITICAL[/]")
        if self.high_count:
            parts.append(f"[yellow]{self.high_count} HIGH[/]")
        if self.warning_count:
            parts.append(f"[cyan]{self.warning_count} WARNING[/]")
        console.print("  " + " | ".join(parts))
        console.print()

    def _print_plain(self, verbose: bool = False):
        """Fallback plain-text output if rich is not installed."""
        print(f"\nane-fp16-lint — {self.model_path}")
        print(f"{'=' * 60}")

        if not self.findings:
            print("  ✅ No fp16-unsafe operations detected.\n")
            return

        for f in self.findings:
            print(f"  {f}")
            if verbose:
                print(f"    Fix → {f.rule.stable_alternative}")

        print(f"\n  {self.critical_count} CRITICAL | "
              f"{self.high_count} HIGH | {self.warning_count} WARNING\n")

    def to_dict(self) -> dict:
        """Serialize report to a dictionary (for JSON output)."""
        return {
            "model_path": self.model_path,
            "total_findings": len(self.findings),
            "critical": self.critical_count,
            "high": self.high_count,
            "warnings": self.warning_count,
            "findings": [
                {
                    "op_name": f.op_name,
                    "op_type": f.op_type,
                    "risk": f.risk.value,
                    "threshold": f.rule.threshold,
                    "description": f.rule.description,
                    "stable_alternative": f.rule.stable_alternative,
                    "pytorch_amp_class": f.rule.pytorch_amp_class,
                }
                for f in self.findings
            ],
        }
