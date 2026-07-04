"""Chunked multi-robot state loader for RH20T (all cfgs).

Chunks are TICK-ANCHORED: tick k = k-th camera/state timestamp of a contiguous
segment (the transformed/ streams are camera-aligned and share timestamps exactly;
verified across all 7 cfgs). No interpolation/resampling anywhere — native samples
only. Scenes contain multi-minute recording gaps, so ticks are split into segments
at gaps > GAP_MS and the last tick of each segment is dropped (no next tick to
bound its ee window).

Per chunk:
  motor      (1, N_MOTORS=8, N_CH=3)  rows 0-6 joints (row 6 masked for 6-DOF
             robots), row 7 gripper width; C = [sin q, cos q, symlog dq]
             (gripper row: ch0 = symlog width). Joint torque (KUKA-only) is ignored.
  motor_mask (8, 3) bool, True = valid
  ee         (EE_T=13, 15)  high_freq_data (100/125 Hz) samples in [tick_k, tick_k+1):
             [symlog zeroed F/T (6), symlog tcp xyz (3), tcp quat -> 6D rot (6)]
  ee_mask    (13,) bool — all False when high_freq_data is missing/empty
             (all of cfg5, ~1/3 of cfg3 scenes)
"""
import os

import numpy as np

from world_tokenizer.state import quat_to_6d, symlog

GAP_MS = 500
EE_T = 13
N_MOTORS = 8   # 7 joint rows + 1 gripper-width row
N_CH = 3       # sin q, cos q, symlog dq
EE_DIM = 15    # F/T 6 + tcp xyz 3 + tcp 6D rotation 6

ROBOT_NAMES = ["flexiv", "ur5", "franka", "kuka"]
ROBOT_OF_CFG = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 3, 7: 3}
# joint.npy vector length -> (dof, has velocity); KUKA's trailing torque is ignored
_JOINT_LAYOUT = {6: (6, False), 14: (7, True), 21: (7, True)}


class SceneChunks:
    """Tick-anchored chunks for one scene. len() = number of chunks."""

    def __init__(self, scene_dir, serial=None):
        T = os.path.join(scene_dir, "transformed")
        joint = np.load(os.path.join(T, "joint.npy"), allow_pickle=True).item()
        grip = np.load(os.path.join(T, "gripper.npy"), allow_pickle=True).item()
        self.serial = serial or sorted(set(joint) & set(grip))[0]
        self._jd, self._gd = joint[self.serial], grip[self.serial]

        vec = np.atleast_1d(self._jd[next(iter(self._jd))])
        if len(vec) not in _JOINT_LAYOUT:
            raise ValueError(f"unexpected joint vector length {len(vec)} in {scene_dir}")
        self.dof, self.has_vel = _JOINT_LAYOUT[len(vec)]

        # high-freq stream (may be absent/empty): sorted ts + entries under "base"
        self._hf_ts, self._hf = np.empty(0, dtype=np.int64), []
        hf_path = os.path.join(T, "high_freq_data.npy")
        if os.path.exists(hf_path):
            base = np.load(hf_path, allow_pickle=True).item().get("base", [])
            if len(base):
                base = sorted(base, key=lambda e: e["timestamp"])
                self._hf_ts = np.array([e["timestamp"] for e in base], dtype=np.int64)
                self._hf = base

        # ticks: timestamps present in both 10Hz streams (identical in practice),
        # split into contiguous segments at gaps, drop each segment's last tick
        ts = np.array(sorted(set(self._jd) & set(self._gd)), dtype=np.int64)
        starts, bounds = [], []
        seg_start = 0
        for i in range(1, len(ts) + 1):
            if i == len(ts) or ts[i] - ts[i - 1] > GAP_MS:
                if i - seg_start >= 2:
                    starts.append(ts[seg_start:i - 1])
                    bounds.append(ts[seg_start + 1:i])
                seg_start = i
        self.ticks = np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)
        self._next = np.concatenate(bounds) if bounds else np.empty(0, dtype=np.int64)

    def __len__(self):
        return len(self.ticks)

    def chunk(self, i):
        t0, t1 = int(self.ticks[i]), int(self._next[i])

        motor = np.zeros((1, N_MOTORS, N_CH), dtype=np.float32)
        mask = np.zeros((N_MOTORS, N_CH), dtype=bool)
        vec = np.atleast_1d(self._jd[t0]).astype(np.float64)
        q = vec[:self.dof]
        motor[0, :self.dof, 0] = np.sin(q)
        motor[0, :self.dof, 1] = np.cos(q)
        mask[:self.dof, :2] = True
        if self.has_vel:
            motor[0, :self.dof, 2] = symlog(vec[self.dof:2 * self.dof])
            mask[:self.dof, 2] = True
        motor[0, 7, 0] = symlog(self._gd[t0]["gripper_command"][0])
        mask[7, 0] = True

        ee = np.zeros((EE_T, EE_DIM), dtype=np.float32)
        ee_mask = np.zeros(EE_T, dtype=bool)
        i0, i1 = np.searchsorted(self._hf_ts, [t0, t1])
        for k, e in enumerate(self._hf[i0:i1][:EE_T]):
            tcp = np.asarray(e["tcp"], dtype=np.float64)
            ee[k, :6] = symlog(e["zeroed"])
            ee[k, 6:9] = symlog(tcp[:3])
            ee[k, 9:15] = quat_to_6d(tcp[3:7])
            ee_mask[k] = True
        return motor, mask, ee, ee_mask
