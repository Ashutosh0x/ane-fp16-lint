"""Load a Core ML .mlpackage and extract the MIL program."""

from pathlib import Path
from typing import Any


def load_mil_program(model_path: str) -> Any:
    """Load a .mlpackage file and return its MIL program.

    Args:
        model_path: Path to a .mlpackage file or a ct.models.MLModel.

    Returns:
        The MIL Program object from coremltools.

    Raises:
        FileNotFoundError: If the model path doesn't exist.
        ValueError: If no MIL program can be extracted.
    """
    import coremltools as ct

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not str(path).endswith(".mlpackage"):
        raise ValueError(
            f"Expected .mlpackage file, got: {path.suffix}. "
            f"Only ML Program format is supported."
        )

    model = ct.models.MLModel(str(path))

    # Try to access the MIL program
    mil_program = getattr(model, "_mil_program", None)
    if mil_program is None:
        raise ValueError(
            f"Could not extract MIL program from {model_path}. "
            f"The model may need to be re-converted with coremltools >= 7.0."
        )

    return mil_program


def load_from_conversion(model, example_input, **convert_kwargs) -> Any:
    """Convert a PyTorch model and return the MIL program for linting.

    This is useful for linting models before saving them.

    Args:
        model: A PyTorch model (traced or scripted).
        example_input: Example input tensor.
        **convert_kwargs: Additional arguments for ct.convert().

    Returns:
        The MIL Program object.
    """
    import coremltools as ct
    import torch

    if isinstance(model, torch.nn.Module):
        traced = torch.jit.trace(model.eval(), example_input)
    else:
        traced = model

    defaults = {
        "convert_to": "mlprogram",
        "compute_precision": ct.precision.FLOAT16,
    }
    defaults.update(convert_kwargs)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=example_input.shape)],
        **defaults,
    )

    mil_program = getattr(mlmodel, "_mil_program", None)
    if mil_program is None:
        raise ValueError("Could not extract MIL program after conversion.")

    return mil_program
