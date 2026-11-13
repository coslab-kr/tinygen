"""Run native tools in isolated working directories without modifying package data."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from ._kernels import collect_kernels
from ._optimization import process_one_tinygen_dir

_PACKAGE = Path(__file__).resolve().parent
_REPO = _PACKAGE.parents[1]


def _tool(name: str, explicit: str | os.PathLike | None = None) -> Path:
    override = explicit or os.environ.get(
        "TINYGEN_TINYC" if name == "tinyc" else "TINYGEN_MLIR_TRANSLATE"
    )
    candidates = [override] if override else [
        _PACKAGE / "bin" / name,
        _REPO / "build" / "compiler" / name,
        name,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = shutil.which(str(candidate))
        if resolved:
            return Path(resolved).resolve()
    raise FileNotFoundError(
        f"{name} executable not found. Install a compiler-enabled TinyGen wheel, "
        f"add {name} to PATH, or pass its executable path explicitly."
    )


def _assets() -> tuple[Path, Path]:
    installed = _PACKAGE / "data"
    if (installed / "kernels").is_dir():
        return installed / "kernels", installed / "CMSIS-NN"
    return _REPO / "kernels", _REPO / "third_party" / "CMSIS-NN"


def _input(model_path: str | os.PathLike) -> Path:
    path = Path(model_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return path


def _lower(model: Path, work: Path, tinyc: Path) -> tuple[Path, Path]:
    subprocess.run([str(tinyc), str(model)], cwd=work, check=True)
    emitc = work / "model_output.mlir"
    ops = work / "model_ops_list.txt"
    for path in (emitc, ops):
        if not path.is_file():
            raise RuntimeError(f"tinyc did not produce {path.name}")
    return emitc, ops


def lower_model(
    model_path: str | os.PathLike,
    output_dir: str | os.PathLike = "gen",
    *,
    tinyc: str | os.PathLike | None = None,
) -> Path:
    """Invoke tinyc and save EmitC MLIR and the operator list. Return the MLIR path."""
    model = _input(model_path)
    compiler = _tool("tinyc", tinyc)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tinygen-") as tmp:
        emitc, ops = _lower(model, Path(tmp), compiler)
        destination = output / f"{model.stem}_emitc.mlir"
        shutil.copy2(emitc, destination)
        shutil.copy2(ops, output / "model_ops_list.txt")
    return destination


def compile_model(
    model_path: str | os.PathLike,
    output_dir: str | os.PathLike = "gen",
    *,
    tinyc: str | os.PathLike | None = None,
    mlir_translate: str | os.PathLike | None = None,
    optimize: bool = True,
) -> Path:
    """Generate C++ and standalone kernel sources; return <output>/<model>_tinygen.

    ``tinyc`` and ``mlir_translate`` may override the bundled executable paths.
    Native subprocess failures propagate as subprocess.CalledProcessError.
    """
    model = _input(model_path)
    compiler = _tool("tinyc", tinyc)
    translator = _tool("mlir-translate", mlir_translate)
    kernels, cmsis = _assets()
    output = Path(output_dir).resolve()
    destination = output / f"{model.stem}_tinygen"
    # Refuse stale outputs and preserve earlier results on tool failure.
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tinygen-", dir=output) as tmp:
        work = Path(tmp)
        emitc, ops = _lower(model, work, compiler)
        stage = work / "result"
        stage.mkdir()
        cpp = stage / f"{model.stem}.cpp"
        with cpp.open("w", encoding="utf-8") as stream:
            subprocess.run(
                [str(translator), "--mlir-to-cpp", str(emitc)],
                cwd=work, stdout=stream, check=True,
            )
        shutil.copy2(emitc, stage / f"{model.stem}_emitc.mlir")
        shutil.copy2(ops, stage / "model_ops_list.txt")
        collect_kernels(ops, stage, kernels, cmsis)
        if optimize:
            process_one_tinygen_dir(stage)
        stage.rename(destination)
    return destination
