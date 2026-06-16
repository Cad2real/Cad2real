import numpy as np
import torch

# ---------- math helpers ----------
def smoothstep_5th(u):
    u = np.clip(u, 0.0, 1.0)
    # 10u^3 - 15u^4 + 6u^5
    return u*u*u*(10.0 + u*(-15.0 + 6.0*u))

def clip_range(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def resample_traj(q, T_new):
    """
    Linear time resampling to length T_new.
    q: [T, D] or [T] numpy
    """
    q = np.asarray(q, dtype=np.float32)
    if q.ndim == 1:
        q = q[:, None]
    T_old, D = q.shape
    if T_old == T_new:
        return q.copy()
    t_old = np.linspace(0.0, 1.0, T_old, dtype=np.float32)
    t_new = np.linspace(0.0, 1.0, T_new, dtype=np.float32)
    q_new = np.empty((T_new, D), dtype=np.float32)
    for d in range(D):
        q_new[:, d] = np.interp(t_new, t_old, q[:, d])
    return q_new

def pad_to(q, T_target):
    """Pad a [T, D] to T_target by repeating last row."""
    T = q.shape[0]
    if T >= T_target:
        return q[:T_target]
    pad = np.repeat(q[-1:, :], T_target - T, axis=0)
    return np.vstack([q, pad])

# ---------- dataset -> per-sample extraction ----------
def unpack_sample(sample, dt_default):
    """
    Expecting one of these common structures:
      A) {'n2': [T, Dn2], 'n5': [T, Dn5], 'rh': [T, Drh], 'lh': [T, Dlh], 'dt': float}
      B) {'traj': [T, Dtotal], 'splits': {'n2': idxs, 'n5': ..., 'rh': ..., 'lh': ...}, 'dt': float}
    Edit here if your dataset uses other keys.
    """
                              # e.g., {'n2': [0..6), 'n5': [6..12), ...}
    q_n2 = sample['nova2_position']
    q_n5 = sample['nova5_position']
    q_rh = sample['right_position']
    q_lh = sample['left_position']

    dt_src = 0.02
    return q_n2, q_n5, q_rh, q_lh, dt_src

# ---------- batching convenience ----------
def split_batch(batch):
    """
    Turn whatever your collate_fn returned into a list of per-sample dicts.
    Supports:
      - dict of arrays/lists (length B)
      - list of dicts
      - already a dict (single sample)
    """
    if isinstance(batch, list):
        assert len(batch) > 0
        if isinstance(batch[0], dict):
            return batch
        else:
            raise TypeError("Unsupported batch element type.")
    elif isinstance(batch, dict):
        # dict of lists -> list of dicts
        values = list(batch.values())
        if all(isinstance(v, (list, tuple)) for v in values):
            B = len(values[0])
            return [{k: batch[k][i] for k in batch} for i in range(B)]
        else:
            return [batch]  # already a single sample dict
    else:
        raise TypeError("Unsupported batch format from collate_fn.")