#!/usr/bin/env python3
"""Upsample the Unitree G1 handshake reference motion from 30 Hz to 50 Hz
and export the 29-DoF reference joint trajectory as a flat float32 binary
(row-major, shape [M, 29], no header).

The C++ orchestrator fread()s this blob and infers the frame count from
the file size.
"""

import argparse
from pathlib import Path

import numpy as np

SRC_FPS = 30.0
DST_FPS = 50.0
NUM_DOF = 29

DEFAULT_SRC = Path(
    "/home/eduardot/Documents/Research/Masters research/Resources/"
    "shake_hand05_poses_100_jpos.npz"
)
DEFAULT_DST = Path(
    "/home/eduardot/Documents/Research/Masters research/Resources/"
    "shake_hand05_poses_50hz.bin"
)


def upsample(dof_positions: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    n_src, n_dof = dof_positions.shape
    duration = (n_src - 1) / src_fps
    n_dst = int(np.floor(duration * dst_fps)) + 1

    t_src = np.arange(n_src) / src_fps
    t_dst = np.arange(n_dst) / dst_fps

    out = np.empty((n_dst, n_dof), dtype=np.float32)
    for j in range(n_dof):
        out[:, j] = np.interp(t_dst, t_src, dof_positions[:, j])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = ap.parse_args()

    data = np.load(args.src)
    if "dof_positions" not in data.files:
        raise SystemExit(f"{args.src} has no 'dof_positions' array")

    dof = np.asarray(data["dof_positions"], dtype=np.float32)
    if dof.ndim != 2 or dof.shape[1] != NUM_DOF:
        raise SystemExit(f"expected (*, {NUM_DOF}), got {dof.shape}")

    file_fps = float(data["fps"][0]) if "fps" in data.files else SRC_FPS
    if abs(file_fps - SRC_FPS) > 1e-3:
        print(f"WARN: file reports {file_fps} Hz, assuming {SRC_FPS} Hz")

    ref = upsample(dof, SRC_FPS, DST_FPS)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    ref.tofile(args.dst)

    print(f"src : {dof.shape} @ {SRC_FPS:>5.1f} Hz  "
          f"({dof.shape[0] / SRC_FPS:.3f} s)")
    print(f"dst : {ref.shape} @ {DST_FPS:>5.1f} Hz  "
          f"({ref.shape[0] / DST_FPS:.3f} s)")
    print(f"wrote {args.dst} ({args.dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
