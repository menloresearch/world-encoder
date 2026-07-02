"""Robot-state loader for RH20T cfg3.

Reads transformed/*.npy directly and interpolates to a query timestamp. We do NOT use
rh20t_api's aligned getters: get_joint_angles_aligned/get_gripper have an order-dependent bug
(they only build their timestamp index if `_base_aligned_timestamps` is still None, so calling
get_ft_aligned/get_tcp_aligned first breaks them). Reading the raw dicts is order-independent.

State vector at a timestamp (base frame, one camera serial), preprocessed per modality:
  joints  (6 angles)  -> sin/cos        12
  tcp pos (3)         -> symlog          3
  tcp quat(4)         -> 6D rotation     6
  ft zeroed (6)       -> symlog          6
  gripper width (1)   -> symlog          1
                                        == 28 dims
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

STATE_DIM = 28


def symlog(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def quat_to_6d(q):
    """quaternion (x, y, z, w) -> 6D rotation (first two columns of R). Continuous, no gimbal."""
    R = Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_matrix()
    return R[:, :2].reshape(-1)


def _interp(ts_sorted, vals, t):
    """Linear interpolation of vals (N,d) at time t, clamped to the ends."""
    i = int(np.searchsorted(ts_sorted, t))
    if i <= 0:
        return vals[0]
    if i >= len(ts_sorted):
        return vals[-1]
    t0, t1 = ts_sorted[i - 1], ts_sorted[i]
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    return vals[i - 1] * (1 - w) + vals[i] * w


class SceneState:
    """Interpolatable robot state for one scene, from transformed/*.npy."""

    def __init__(self, scene_dir, serial=None):
        T = os.path.join(scene_dir, "transformed")
        joint = np.load(os.path.join(T, "joint.npy"), allow_pickle=True).item()
        tcp = np.load(os.path.join(T, "tcp_base.npy"), allow_pickle=True).item()
        ft = np.load(os.path.join(T, "force_torque_base.npy"), allow_pickle=True).item()
        grip = np.load(os.path.join(T, "gripper.npy"), allow_pickle=True).item()
        # a serial present in every stream
        self.serial = serial or sorted(set(joint) & set(tcp) & set(ft) & set(grip))[0]

        # joints: {ts: (6,)} -> sorted arrays
        jt = sorted(joint[self.serial])
        self._jt = np.array(jt)
        self._jv = np.array([joint[self.serial][t] for t in jt])            # (N,6)
        # tcp_base / ft_base: list of dicts
        self._tt, self._tv = self._from_list(tcp[self.serial], "tcp")       # (N,7)
        self._ft_t, self._ft_v = self._from_list(ft[self.serial], "zeroed")  # (N,6)
        # gripper: {ts: {"gripper_command":[w,..], "gripper_info":[..]}}
        gt = sorted(grip[self.serial])
        self._gt = np.array(gt)
        self._gv = np.array([[grip[self.serial][t]["gripper_command"][0]] for t in gt])  # (N,1)

    @staticmethod
    def _from_list(entries, key):
        entries = sorted(entries, key=lambda e: e["timestamp"])
        ts = np.array([e["timestamp"] for e in entries])
        vals = np.array([np.asarray(e[key], dtype=np.float64) for e in entries])
        return ts, vals

    def state(self, ts):
        j = _interp(self._jt, self._jv, ts)          # (6,) angles
        tcp = _interp(self._tt, self._tv, ts)         # (7,) pos+quat
        ftv = _interp(self._ft_t, self._ft_v, ts)     # (6,)
        g = _interp(self._gt, self._gv, ts)           # (1,)
        pos, quat = tcp[:3], tcp[3:7]
        return np.concatenate([
            np.sin(j), np.cos(j),      # 12
            symlog(pos),               # 3
            quat_to_6d(quat),          # 6
            symlog(ftv),               # 6
            symlog(g),                 # 1
        ]).astype(np.float32)          # 28

    def raw_ft(self, ts):
        """Interpolated zeroed force/torque (6,) in raw units — for leak-free force targets."""
        return _interp(self._ft_t, self._ft_v, ts)


# index of the F/T block inside the 28-dim state vector (exclude for leak-free force prediction)
FT_DIMS = list(range(21, 27))
