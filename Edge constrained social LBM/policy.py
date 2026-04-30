import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

from common.joint_mapper import create_isaac_to_real_mapper
from common.math_utils import (
    _linspace_rows,
    _remove_yaw_keep_rp_wxyz,
    _slerp,
    _yaw_component_wxyz,
    _zero_z,
)
from common.utils import DictToClass, MotionUDPServer, joint_names_23, joint_names_29
from paths import ASSETS_DIR, REAL_G1_ROOT

def benchmark_onnx(module, sample_input, runs=100, warmup=10, desc=""):
    for _ in range(warmup):
        _ = module(sample_input)

    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = module(sample_input)
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000.0)

    mean = statistics.mean(ts)
    stdev = statistics.pstdev(ts)
    p50 = np.percentile(ts, 50)
    p90 = np.percentile(ts, 90)
    p95 = np.percentile(ts, 95)
    p99 = np.percentile(ts, 99)

    print(f"[{desc}] runs={runs}, warmup={warmup}")
    print(f"mean={mean:.3f} ms, stdev={stdev:.3f} ms")
    print(f"p50={p50:.3f} ms, p90={p90:.3f} ms, p95={p95:.3f} ms, p99={p99:.3f} ms")
    return {"mean": mean, "stdev": stdev, "p50": p50, "p90": p90, "p95": p95, "p99": p99}


class ONNXModule:
    def __init__(self, path: str):
        self.ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta_path = path.replace(".onnx", ".json")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.in_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["in_keys"]]
        self.out_keys = [k if isinstance(k, str) else tuple(k) for k in self.meta["out_keys"]]

    def __call__(self, input: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        args = {
            inp.name: input[key]
            for inp, key in zip(self.ort_session.get_inputs(), self.in_keys)
            if key in input
        }
        outputs = self.ort_session.run(None, args)
        outputs = {k: v for k, v in zip(self.out_keys, outputs)}
        return outputs

# =========================================
# Policy Base
# =========================================
class Policy:
    FADE_OUT_DURATION = 2.0  # s

    def __init__(self, name: str, policy_cfg: DictToClass, controller):
        self.name = name
        self.controller = controller

        self.config = policy_cfg

        # Resolve policy path relative to repo root for robustness
        p = Path(policy_cfg.policy_path)
        self.policy_path = str(p if p.is_absolute() else (REAL_G1_ROOT / p))
        self.action_joint_names = list(policy_cfg.action_joint_names)
        self.action_scale_isaac = np.array(policy_cfg.action_scale, dtype=np.float32)
        self.alpha = float(policy_cfg.action_alpha)
        self.lowstate_alpha = float(policy_cfg.lowstate_alpha)
        self.action_clip = float(policy_cfg.action_clip)

        if hasattr(policy_cfg, "kps_real"):
            self.kps_real = np.array(policy_cfg.kps_real, dtype=np.float32)
        if hasattr(policy_cfg, "kds_real"):
            self.kds_real = np.array(policy_cfg.kds_real, dtype=np.float32)

        assert len(self.action_joint_names) == len(self.action_scale_isaac), (
            f"[{self.name}] action_joint_names ({len(self.action_joint_names)}) "
            f"!= action_scale ({len(self.action_scale_isaac)})"
        )

        self.module = ONNXModule(self.policy_path)

        self.mapper_action = create_isaac_to_real_mapper(
            self.action_joint_names,
            self.controller.config.real_joint_names
        )
        map_info = self.mapper_action.get_mapping_info()
        print(f"[Policy:{self.name}] Action mapping: {map_info['mapped_joints']}/{map_info['from_space_size']} mapped")
        if map_info['unmapped_from_joints']:
            print(f"[Policy:{self.name}] Unmapped policy action joints: {map_info['unmapped_from_joints']}")
        if map_info['unmapped_to_joints']:
            print(f"[Policy:{self.name}] Unmapped Real joints: {map_info['unmapped_to_joints']}")

        self.policy_input: Optional[Dict[str, np.ndarray]] = None
        self.applied_action_isaac = np.zeros(len(self.action_joint_names), dtype=np.float32)
        self.last_action = np.zeros(len(self.action_joint_names), dtype=np.float32)

        self._fading_deadline: Optional[float] = None
        self._active: bool = False

        self.obs_modules = []
        self.num_obs = 0
        self._build_obs_modules()

        self.policy_input = {
            "policy": np.zeros((1, self.num_obs), dtype=np.float32),
            "is_init": np.ones((1,), dtype=bool)
        }
        benchmark_onnx(self.module, self.policy_input, runs=100, warmup=200, desc="model@cuda")

    # -------- lifecycle ----------
    def fade_in(self):
        self.reset()
        self._active = True
        self._fading_deadline = None
        print(f"[Policy:{self.name}] fade_in()")

    def fade_out(self) -> float:
        self._fading_deadline = time.monotonic() + self.FADE_OUT_DURATION
        print(f"[Policy:{self.name}] fade_out() - continue until {self._fading_deadline:.3f}")
        return self._fading_deadline

    def is_fading(self) -> bool:
        return self._fading_deadline is not None

    def fading_done(self) -> bool:
        return self._fading_deadline is not None and time.monotonic() >= self._fading_deadline

    def deactivate(self):
        self._active = False
        self._fading_deadline = None
        print(f"[Policy:{self.name}] deactivated")

    # -------- abstract hooks ----------
    def _build_obs_modules(self):
        raise NotImplementedError

    def _reset_obs_modules(self):
        for m in self.obs_modules:
            if hasattr(m, "reset") and callable(m.reset):
                m.reset()

    def update_obs(self):
        obs_list = []
        for m in self.obs_modules:
            m.update()
            obs_list.append(m.compute())
        if self.policy_input is None:
            self.policy_input = {
                "policy": np.zeros((1, self.num_obs), dtype=np.float32),
                "is_init": np.ones((1,), dtype=bool),
            }
        else:
            self.policy_input["policy"][0, :] = np.concatenate(obs_list, axis=0)

    def compute_action(self) -> np.ndarray:
        try:
            out = self.module(self.policy_input)
        except Exception as e:
            print(f"[Policy:{self.name}] ONNX forward failed: {e}")
            return np.zeros(self.controller.dof_size_real, dtype=np.float32)

        if ("next", "adapt_hx") in out:
            self.policy_input["adapt_hx"][:] = out["next", "adapt_hx"]
        self.policy_input["is_init"][:] = False

        action_isaac = out["action"].copy()[0].astype(np.float32).clip( -self.action_clip, self.action_clip)
        self.last_action[:] = action_isaac
        self.applied_action_isaac[:] = action_isaac * self.action_scale_isaac

        action_real = self.mapper_action.map_action_from_to(self.applied_action_isaac)
        return action_real

    def reset(self):
        self.policy_input = None
        self.applied_action_isaac[:] = 0.0
        self.last_action[:] = 0.0
        self._reset_obs_modules()

# =========================================
# Policy Subclasses
# =========================================
def mapping_joints(data: np.ndarray, target: List[str]):
    nums = data.shape[-1]
    if nums == len(target):
        return data
    if nums == 29:
        current = joint_names_29
        print("[Mapping] from 29 to 23")
    elif nums == 23:
        current = joint_names_23
        print("[Mapping] from 23 to 29")
    else:
        raise ValueError(f"Unsupported number of joints: {nums}")

    new_data = np.zeros((data.shape[0], len(target)), dtype=np.float32)
    for i, name in enumerate(target):
        if name in current:
            new_data[:, i] = data[:, current.index(name)]
    return new_data.astype(np.float32)

class TrackingPolicy(Policy):
    def __init__(self, name: str, policy_cfg: DictToClass, controller):
        # ---- Config ---------------------------------------------------------
        self.body_name = "torso_link"
        self.transition_steps = int(getattr(policy_cfg, "transition_steps", 100))
        self.udp_enable = bool(getattr(policy_cfg, "udp_enable", True))
        self.udp_host = str(getattr(policy_cfg, "udp_host", "127.0.0.1"))
        self.udp_port = int(getattr(policy_cfg, "udp_port", 28562))

        # ---- Load motions; keep only a single body (default index=13) -------
        self.motions: Dict[str, Dict[str, np.ndarray]] = {}
        for m in policy_cfg.motions:
            mc = DictToClass(m)
            motion_name = mc.name
            mp = Path(mc.path)
            path = str(mp if mp.is_absolute() else (REAL_G1_ROOT / mp))
            t0, t1 = int(mc.start), int(mc.end)

            data = np.load(path, allow_pickle=True).item()
            # Expected keys: joint_pos(T,J), root_quat_w(T,4), root_pos_w(T,N,3)
            joint_pos = data["joint_pos"][t0:t1].astype(np.float32)
            joint_pos = mapping_joints(joint_pos, policy_cfg.dataset_joint_names)
            root_quat = data["root_quat_w"][t0:t1].astype(np.float32)
            try:
                root_pos_all = data["root_pos_w"][t0:t1].astype(np.float32)  # (T,N,3)
            except KeyError:
                breakpoint()

            root_quat_rp = _remove_yaw_keep_rp_wxyz(root_quat)      # (T,4), no yaw
            root_quat_yaw = _yaw_component_wxyz(root_quat)        # (T,4), yaw only

            self.motions[motion_name] = {
                "joint_pos": joint_pos,               # (T,J)
                "root_height": root_pos_all[:, 2:3],           # (T,1)
                "root_quat_rp": root_quat_rp,            # (T,4)  (roll+pitch only)
                "root_quat_yaw": root_quat_yaw,         # (T,4)  (yaw only)
                "root_pos": _zero_z(root_pos_all),    # (T,3) z=0
            }

        # ---- One-frame motion clips (config provided) ------------------------
        for m in policy_cfg.motion_clips:
            mc = DictToClass(m)
            motion_name = mc.name
            # joints
            joint_pos_1 = mapping_joints(
                np.asarray(mc.joint_pos, dtype=np.float32).reshape(1, -1),
                policy_cfg.dataset_joint_names
            )
            
            root_height_1 = np.full((1, 1), 0.815, dtype=np.float32)
            # quats/poses
            root_quat_1 = np.asarray(mc.root_quat, dtype=np.float32).reshape(1, 4)
            root_pos_1 = np.asarray(mc.root_pos, dtype=np.float32).reshape(1, 3)

            root_quat_rp_1 = _remove_yaw_keep_rp_wxyz(root_quat_1)
            root_quat_yaw_1 = _yaw_component_wxyz(root_quat_1)
            root_pos_flat_1 = _zero_z(root_pos_1)

            self.motions[motion_name] = {
                "joint_pos": joint_pos_1,            # (1,J)
                "root_height": root_height_1,        # (1,1)
                "root_quat_rp": root_quat_rp_1,         # (1,4)
                "root_quat_yaw": root_quat_yaw_1,        # (1,4)
                "root_pos": root_pos_flat_1,         # (1,3)
            }

        assert "default" in self.motions, "[TrackingPolicy] motions must include a 'default' clip (length==1)."

        # ---- Reference stream (updated when a motion starts) -----------------
        self.ref_joint_pos: Optional[np.ndarray] = None  # (T_ref, J)
        self.ref_root_height: Optional[np.ndarray] = None  # (T_ref, 1)
        self.ref_root_quat_rp: Optional[np.ndarray] = None  # (T_ref, 4)
        self.ref_root_quat_yaw: Optional[np.ndarray] = None  # (T_ref, 4)
        self.ref_root_pos: Optional[np.ndarray] = None   # (T_ref, 3)

        # ---- Playback state --------------------------------------------------
        self.ref_idx: int = 0
        self.ref_len: int = 0
        self.current_name: str = "default"
        self.current_done: bool = True  # boot: default done

        # ---- Misc ------------------------------------------------------------
        self.n_joints = len(policy_cfg.dataset_joint_names)

        # Optional UDP selector
        self._udp_server: Optional[MotionUDPServer] = None
        if self.udp_enable:
            try:
                self._udp_server = MotionUDPServer(self.udp_host, self.udp_port)
                self._udp_server.start()
            except Exception as e:
                print(f"[TrackingPolicy] Failed to start UDP server: {e}")

        super().__init__(name, policy_cfg, controller)
        self.init_count = 0

    # -- Lifecycle overrides ---------------------------------------------------
    def fade_in(self):
        super().fade_in()
        self._start_motion_from_current("default")

    def fade_out(self) -> float:
        self._start_motion_from_current("default")
        return super().fade_out()

    def deactivate(self):
        if self._udp_server is not None:
            self._udp_server.stop()
        self.ref_root_pos = None
        self.ref_root_height = None
        self.ref_root_quat_yaw = None
        self.ref_root_quat_rp = None
        self.ref_joint_pos = None
        super().deactivate()

    # -- Obs building ----------------------------------------------------------
    def _build_obs_modules(self):
        from observation import TrackingCommandObs, RootAngVelB, ProjectedGravityB, JointPos, PrevActions, BootIndicator
        self.obs_modules = [
            BootIndicator(),
            TrackingCommandObs(self.controller, self),
            RootAngVelB(self.controller),
            ProjectedGravityB(self.controller),
            JointPos(self.controller, pos_steps=[0, 1, 2, 3, 4, 8]),
            PrevActions(self, steps=3),
        ]
        self.num_obs = sum(m.size for m in self.obs_modules)

    # -- Public: request a motion by name -------------------------------------
    def request_motion(self, name: str) -> bool:
        """Returns True if accepted. Rule: Only when current is 'default' and done."""
        if name not in self.motions:
            print(f"[TrackingPolicy] Unknown motion '{name}'")
            return False
        if (self.current_name == "default" or name == "default") and self.current_done:
            self._start_motion_from_current(name)
            return True
        else:
            print(f"[TrackingPolicy] Reject '{name}': current='{self.current_name}', done={self.current_done}")
            return False

    # -- Update: poll UDP & advance ref ---------------------------------------
    def update_obs(self):
        if self._udp_server is not None:
            for cmd in self._udp_server.pop_all():
                if cmd == "default":
                    self.request_motion("default")
                else:
                    self.request_motion(cmd)
        if self.ref_len > 0 and self.ref_idx < self.ref_len - 1:
            self.ref_idx += 1
            if self.ref_idx == self.ref_len - 1:
                self.current_done = True
        super().update_obs()

    # -- Internals -------------------------------------------------------------
    def _read_current_state(self) -> Dict[str, np.ndarray]:
        q_real = self.controller.qj_real.copy()
        real_names = list(self.controller.config.real_joint_names)
        target_names = list(self.config.dataset_joint_names)
        name_to_idx = {n: i for i, n in enumerate(real_names)}
        q_policy = np.zeros(len(target_names), dtype=np.float32)
        for i, n in enumerate(target_names):
            j = name_to_idx.get(n, None)
            if j is not None and j < q_real.shape[0]:
                q_policy[i] = q_real[j]

        if self.ref_root_pos is not None:
            root_quat_yaw = self.ref_root_quat_yaw[self.ref_idx]
            root_quat_rp = self.ref_root_quat_rp[self.ref_idx]
            root_pos = self.ref_root_pos[self.ref_idx]
            root_height = self.ref_root_height[self.ref_idx]
        else:
            root_quat_yaw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity
            root_quat_rp = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity
            root_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            root_height = self.motions["default"]["root_height"][0].astype(np.float32).copy()
        return {
            "joint_pos": q_policy.astype(np.float32),
            "root_pos":  root_pos,    # (3,) z=0
            "root_quat_yaw": root_quat_yaw,    # (4,) yaw only (wxyz)
            "root_quat_rp": root_quat_rp,
            "root_height": root_height,
        }

    def _align_motion_to_current(self,
                                 motion: Dict[str, np.ndarray],
                                 curr: Dict[str, np.ndarray]
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p0 = motion["root_pos"][0]         # (3,), z=0
        q0 = motion["root_quat_yaw"][0]        # (4,), yaw-only
        pc = curr["root_pos"]              # (3,), z=0
        qc = curr["root_quat_yaw"]             # (4,), yaw-only

        R0 = R.from_quat(q0, scalar_first=True)
        Rc = R.from_quat(qc, scalar_first=True)
        R_delta = Rc * R0.inv()            # maps q0 -> qc

        # Align body pos: p' = R_delta*(p - p0) + pc
        root_pos_aligned = R_delta.apply(motion["root_pos"] - p0) + pc

        # Align body quat: q' = R_delta * q
        root_quat_yaw_all = R.from_quat(motion["root_quat_yaw"], scalar_first=True)
        root_quat_yaw_aligned = (R_delta * root_quat_yaw_all).as_quat(scalar_first=True)
        # Keep others as-is
        joint_pos_all = motion["joint_pos"].astype(np.float32).copy()
        root_quat_all = motion["root_quat_rp"].astype(np.float32).copy()      # rp-only, unchanged
        root_height_all = motion["root_height"].astype(np.float32).copy()

        return {
            "joint_pos": joint_pos_all,
            "root_height": root_height_all,
            "root_quat_rp": root_quat_all,
            "root_quat_yaw": root_quat_yaw_aligned.astype(np.float32),
            "root_pos": root_pos_aligned.astype(np.float32),
        }

    def _build_transition_prefix(self,
                                 curr: Dict[str, np.ndarray],
                                 tgt_first: Dict[str, np.ndarray]
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = int(self.transition_steps)
        if T <= 0:
            return (np.zeros((0, curr["joint_pos"].shape[0]), dtype=np.float32),
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0, 1), dtype=np.float32))

        joints_tr = _linspace_rows(curr["joint_pos"], tgt_first["joint_pos"], T)
        root_pos_tr = _linspace_rows(curr["root_pos"], tgt_first["root_pos"], T)
        root_quat_yaw_tr = _slerp(curr["root_quat_yaw"], tgt_first["root_quat_yaw"], T)      # yaw-only slerp
        root_quat_rp_tr = _slerp(curr["root_quat_rp"], tgt_first["root_quat_rp"], T)        # rp-only slerp

        rh_start = np.asarray(curr["root_height"], dtype=np.float32).reshape(-1)
        rh_target = np.asarray(tgt_first["root_height"], dtype=np.float32).reshape(-1)
        root_height_tr = _linspace_rows(rh_start, rh_target, T)

        return {
            "joint_pos": joints_tr,
            "root_quat_rp": root_quat_rp_tr,
            "root_quat_yaw": root_quat_yaw_tr,
            "root_pos": root_pos_tr,
            "root_height": root_height_tr,
        }

    def _start_motion_from_current(self, name: str):
        assert name in self.motions
        curr = self._read_current_state()

        # 1) Align entire motion to current *body* pose (yaw-only)
        m = self.motions[name]
        aligned_motion = self._align_motion_to_current(m, curr)

        # 2) Transition prefix towards the first aligned frame
        tgt_first = {
            "joint_pos": aligned_motion["joint_pos"][0],
            "root_quat_rp": aligned_motion["root_quat_rp"][0],   # rp-only
            "root_quat_yaw": aligned_motion["root_quat_yaw"][0],   # yaw-only
            "root_pos": aligned_motion["root_pos"][0],
            "root_height": aligned_motion["root_height"][0],
        }

        trans_motion = self._build_transition_prefix(curr, tgt_first)

        # 3) Concatenate into the reference stream
        self.ref_joint_pos = np.concatenate([trans_motion["joint_pos"], aligned_motion["joint_pos"]], axis=0)
        self.ref_root_height = np.concatenate([trans_motion["root_height"], aligned_motion["root_height"]], axis=0)
        self.ref_root_quat_rp = np.concatenate([trans_motion["root_quat_rp"], aligned_motion["root_quat_rp"]], axis=0)
        self.ref_root_pos = np.concatenate([trans_motion["root_pos"], aligned_motion["root_pos"]], axis=0)
        self.ref_root_quat_yaw = np.concatenate([trans_motion["root_quat_yaw"], aligned_motion["root_quat_yaw"]], axis=0)

        self.ref_idx = 0
        self.ref_len = int(self.ref_joint_pos.shape[0])
        self.current_name = name
        self.current_done = (self.ref_len <= 1)

        print(f"[TrackingPolicy] Start motion '{name}' | ref_len={self.ref_len}, transition={self.transition_steps}")
