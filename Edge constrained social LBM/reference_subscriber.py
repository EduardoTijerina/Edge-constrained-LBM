#!/usr/bin/env python3
"""ZMQ SUB subscriber for the 50 Hz handshake reference stream.

The C++ orchestrator publishes raw float32[29] reference-DoF frames on
tcp://127.0.0.1:5556 (ZMQ PUB). The GentleHumanoid policy drives its
own 50 Hz control loop and calls ``ReferenceSubscriber.poll()`` on
each tick: if a new frame arrived it becomes the current target,
otherwise the last received target is held.

Usage inside the policy loop:

    sub = ReferenceSubscriber()
    while running:
        obs["reference_dof_pos"] = sub.poll()
        policy.step(obs)
        # ... sleep until next 50 Hz tick ...
    sub.close()
"""

from __future__ import annotations

import numpy as np
import zmq

NUM_DOF = 29
POLICY_ENDPOINT = "tcp://127.0.0.1:5556"
FRAME_DTYPE = np.float32


class ReferenceSubscriber:
    def __init__(
        self,
        endpoint: str = POLICY_ENDPOINT,
        num_dof: int = NUM_DOF,
        recv_timeout_ms: int = 1,
        initial: np.ndarray | None = None,
        context: zmq.Context | None = None,
    ) -> None:
        self.num_dof = num_dof
        self.frame_nbytes = num_dof * np.dtype(FRAME_DTYPE).itemsize

        self._ctx = context or zmq.Context.instance()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        # Keep only the newest frame: if the policy tick slips we jump to
        # "now" in the reference instead of replaying stale frames.
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.connect(endpoint)

        if initial is None:
            self._last = np.zeros(num_dof, dtype=FRAME_DTYPE)
        else:
            self._last = np.ascontiguousarray(initial, dtype=FRAME_DTYPE).copy()

        self._frames_received = 0
        self._have_frame = False

    @property
    def frames_received(self) -> int:
        return self._frames_received

    @property
    def has_frame(self) -> bool:
        return self._have_frame

    def poll(self) -> np.ndarray:
        """Non-blocking recv; returns the latest frame or the held target."""
        try:
            msg = self._sub.recv(flags=0)  # honours RCVTIMEO
        except zmq.Again:
            return self._last

        if len(msg) != self.frame_nbytes:
            raise RuntimeError(
                f"expected {self.frame_nbytes} bytes, got {len(msg)}"
            )
        self._last = np.frombuffer(msg, dtype=FRAME_DTYPE).copy()
        self._frames_received += 1
        self._have_frame = True
        return self._last

    def close(self) -> None:
        self._sub.close(linger=0)


def _smoke_test() -> None:
    """Run as a script to verify the stream — prints live frame stats at 50 Hz."""
    import signal
    import time

    stop = {"flag": False}

    def _on_signal(_s, _f):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    sub = ReferenceSubscriber()
    print(f"[SUB] connected to {POLICY_ENDPOINT}, waiting for frames...")

    period = 1.0 / 50.0
    next_tick = time.perf_counter()
    ticks = 0
    try:
        while not stop["flag"]:
            ref = sub.poll()
            ticks += 1
            if ticks % 50 == 0:
                print(
                    f"[SUB] t={ticks/50:5.1f}s  frames={sub.frames_received:4d}  "
                    f"ref[0:3]={ref[:3]}"
                )
            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()
    finally:
        sub.close()
        print(f"[SUB] shutting down, received {sub.frames_received} frames")


if __name__ == "__main__":
    _smoke_test()
