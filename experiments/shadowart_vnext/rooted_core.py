"""Root-connected unions of width-cubed blocks, with exact bottleneck gradients.

This is a soft connected-thickness proxy, not a printability guarantee. The root
is fixed in the valid block-anchor grid before training. Widest paths and pooling
are piecewise differentiable; ties are nonsmooth. Backward selects one active
path's bottleneck and does not use a straight-through estimator. Second-order
autograd is intentionally unsupported.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
from numbers import Integral
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy import ndimage
import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F


@dataclass(frozen=True)
class Backend:
    """Reusable loaded native helper and identity of the captured build inputs."""

    build_dir: Path
    source_path: Path
    source_sha256: str
    library_path: Path
    library_sha256: str
    compile_command: tuple[str, ...]
    library: ctypes.CDLL


def build_backend(build_dir, *, source_path=None) -> Backend:
    """Compile in a new directory, preserving the exact native source compiled.

    An existing directory is always an error, including one from a failed build.
    Compiler and load errors propagate. This performs no install or network IO.
    """
    build_dir = Path(build_dir).resolve()
    source = Path(source_path) if source_path is not None else Path(__file__).with_name(
        "rooted_core_kernel.cpp")
    source = source.resolve(strict=True)
    build_dir.mkdir(parents=True, exist_ok=False)
    captured = build_dir / "rooted_core_kernel.cpp"
    shutil.copyfile(source, captured)
    library_path = build_dir / "rooted_core_kernel.so"
    command = ("/usr/bin/c++", "-std=c++17", "-O3", "-shared", "-fPIC",
               str(captured), "-o", str(library_path))
    subprocess.run(command, check=True, capture_output=True, text=True)
    library = ctypes.CDLL(str(library_path))
    for suffix, ctype in (("f32", ctypes.c_float), ("f64", ctypes.c_double)):
        function = getattr(library, "rooted_reconstruct_" + suffix)
        function.argtypes = (ctypes.POINTER(ctype), ctypes.c_int64, ctypes.c_int64,
                             ctypes.POINTER(ctype), ctypes.POINTER(ctypes.c_int64))
        function.restype = ctypes.c_int
    return Backend(build_dir, captured, hashlib.sha256(captured.read_bytes()).hexdigest(),
                   library_path, hashlib.sha256(library_path.read_bytes()).hexdigest(),
                   command, library)


def _validate_probability(probability):
    if not isinstance(probability, torch.Tensor):
        raise TypeError("probability must be a torch tensor")
    if probability.device.type != "cpu":
        raise ValueError("probability must be on CPU")
    if probability.layout != torch.strided:
        raise ValueError("probability must be a dense strided tensor")
    if probability.dtype not in (torch.float32, torch.float64):
        raise TypeError("probability must have float32 or float64 dtype")
    if probability.ndim != 3 or probability.shape[0] == 0 or len(set(probability.shape)) != 1:
        raise ValueError("probability must be a nonempty XYZ cube")
    if not torch.isfinite(probability).all().item():
        raise ValueError("probability must be finite")
    if probability.min().item() < 0 or probability.max().item() > 1:
        raise ValueError("probability must lie in [0, 1]")


def _validate_width(width, n):
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, Integral):
        raise TypeError("width must be an integer")
    if width < 1 or width > n:
        raise ValueError("width must lie in [1, N]")
    return int(width)


def _validate_root(root, n):
    try:
        root = tuple(root)
    except TypeError as exc:
        raise TypeError("root must contain exactly three integer indices") from exc
    if len(root) != 3:
        raise ValueError("root must contain exactly three integer indices")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in root):
        raise TypeError("root must contain integer indices")
    if any(v < 0 or v >= n for v in root):
        raise ValueError("root is outside the valid anchor grid")
    return tuple(int(v) for v in root)


class _RootedReconstruction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, capacity, root, backend):
        contiguous = capacity.detach().contiguous()
        output = torch.empty_like(contiguous)
        bottleneck = torch.empty(contiguous.shape, dtype=torch.int64, device="cpu")
        ctype = ctypes.c_float if capacity.dtype == torch.float32 else ctypes.c_double
        function = getattr(backend.library, "rooted_reconstruct_" + (
            "f32" if capacity.dtype == torch.float32 else "f64"))
        n = capacity.shape[0]
        flat_root = (root[0] * n + root[1]) * n + root[2]
        status = function(ctypes.cast(contiguous.data_ptr(), ctypes.POINTER(ctype)), n,
                          flat_root, ctypes.cast(output.data_ptr(), ctypes.POINTER(ctype)),
                          ctypes.cast(bottleneck.data_ptr(), ctypes.POINTER(ctypes.c_int64)))
        if status:
            reasons = {1: "invalid native arguments", 2: "native allocation failed",
                       3: "unexpected native exception"}
            raise RuntimeError(f"rooted reconstruction failed ({status}): "
                               f"{reasons.get(status, 'unknown native status')}")
        ctx.save_for_backward(bottleneck)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, output_gradient):
        (bottleneck,) = ctx.saved_tensors
        gradient = torch.zeros(bottleneck.numel(), dtype=output_gradient.dtype, device="cpu")
        gradient.scatter_add_(0, bottleneck.reshape(-1), output_gradient.reshape(-1))
        return gradient.reshape(bottleneck.shape), None, None


def rooted_reconstruction(capacity, root, backend):
    """Exact six-neighbour max-min connectivity to a fixed root on a CPU cube."""
    _validate_probability(capacity)
    root = _validate_root(root, capacity.shape[0])
    if not isinstance(backend, Backend):
        raise TypeError("backend must be returned by build_backend")
    return _RootedReconstruction.apply(capacity, root, backend)


def rooted_core(probability, width, root, backend):
    """Erode valid blocks, reconstruct connected anchors, and expand their blocks.

    Anchor (x,y,z) denotes probability[x:x+width,y:y+width,z:z+width].
    The explicit full dilation puts each anchor on precisely that block, including
    even widths and the far boundary. Each dilated value is <= its input voxel.
    """
    _validate_probability(probability)
    width = _validate_width(width, probability.shape[0])
    root = _validate_root(root, probability.shape[0] - width + 1)
    anchors = -F.max_pool3d(-probability[None, None], width, stride=1)[0, 0]
    connected = rooted_reconstruction(anchors, root, backend)
    full = F.pad(connected[None, None], (width - 1,) * 6, value=-float("inf"))
    return F.max_pool3d(full, width, stride=1)[0, 0]


def choose_root(probability, width):
    """Largest hard anchor component, deepest padded Euclidean interior, flat tie.

    Hard anchors have eroded capacity >= 0.5. Six-connectivity determines the
    component; distance to its zero boundary (including outside the cube) chooses
    the root. Equal-size components and equal depths choose first C-order index.
    Call once on detached initialization and retain this root during training.
    """
    if isinstance(probability, np.ndarray):
        if probability.dtype.kind not in "biuf":
            raise TypeError("probability must contain real numeric values")
        probability = torch.tensor(probability, dtype=torch.float64, device="cpu")
    elif isinstance(probability, torch.Tensor):
        if probability.requires_grad:
            raise ValueError("choose_root requires a detached probability tensor")
    _validate_probability(probability)
    width = _validate_width(width, probability.shape[0])
    with torch.no_grad():
        eroded = -F.max_pool3d(-probability[None, None], width, stride=1)[0, 0]
    hard = eroded.numpy() >= .5
    labels, count = ndimage.label(hard, structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        raise ValueError("no hard eroded anchor component at threshold 0.5")
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    component = labels == int(np.argmax(sizes))
    distance = ndimage.distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1, 1:-1]
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(distance)), hard.shape))
