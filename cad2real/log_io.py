from __future__ import annotations

import ast
from pathlib import Path
from typing import Union

import numpy as np


def load_ros_txt_positions(txt_path: Union[str, Path], key: str = "position"):
    """
    Parse a ROS-like text dump with repeated seq/secs/nsecs/position blocks.

    Returns relative timestamps and a position matrix.
    """
    t_list, q_list = [], []
    secs, nsecs = None, None
    with Path(txt_path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("secs:"):
                try:
                    secs = int(s.split(":", 1)[1].strip())
                except ValueError:
                    secs = None
            elif s.startswith("nsecs:"):
                try:
                    nsecs = int(s.split(":", 1)[1].strip())
                except ValueError:
                    nsecs = 0
            elif s.startswith(f"{key}:"):
                arr = ast.literal_eval(s.split(":", 1)[1].strip())
                q_list.append(np.asarray(arr, dtype=np.float64))
                if secs is not None:
                    t_list.append(float(secs) + float(nsecs or 0) * 1e-9)
                else:
                    t_list.append(len(t_list) * 1.0)

    if not q_list:
        raise RuntimeError(f"No '{key}:' entries found in {txt_path}")

    t = np.asarray(t_list, dtype=np.float64)
    t -= t[0]
    q = np.vstack(q_list)
    uniq, idx = np.unique(t, return_index=True)
    return uniq, q[idx]
