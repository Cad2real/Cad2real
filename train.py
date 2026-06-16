
import argparse
import sys
import os
current_file_dir = os.path.dirname(__file__)

import glob
import cv2
workspace_dir = os.path.dirname(current_file_dir)


sys.path.append(workspace_dir)



import os
import re
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset



import torch
import torch.nn as nn
import torchvision.models as models

# ===========================
# -------- Utilities --------
# ===========================

ActorName = {
    -1: "skip",
    0: "stop",
    1: "nova2",
    2: "nova5",
    3: "LeftHand",
    4: "RightHand",
    5: "thread",
}

def _read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def _safe_float_list(s: str) -> List[float]:
    # Expects something like "0.1, 0.2, 0.3" or "[0.1, 0.2, 0.3]"
    s = s.strip()
    if not s.startswith("["):
        s = "[" + s + "]"
    return [float(x) for x in ast.literal_eval(s)]

def _clean_comment_lines(lines: List[str]) -> List[str]:
    # Remove full-line comments beginning with '#'
    out = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("#"):
            continue
        out.append(ln)
    return out

def _extract_bracketed_list(start_line_idx: int, lines: List[str]) -> Tuple[str, int]:
    """
    Given lines and an index to a line that contains '=[', collect until the
    matching closing bracket at the same nesting depth is found.
    Returns (joined_string, end_line_index_inclusive).
    """
    buff = []
    depth = 0
    started = False
    i = start_line_idx
    while i < len(lines):
        line = lines[i]
        # append the line
        buff.append(line)
        # update bracket depth
        # count '[' and ']'
        for ch in line:
            if ch == '[':
                depth += 1
                started = True
            elif ch == ']':
                depth -= 1
        if started and depth == 0:
            break
        i += 1
    return ("\n".join(buff), i)

def _literal_eval_pose_list(text: str) -> List[Any]:
    """
    text like 'pose_list=[ ..., 0 ]' -> returns Python list
    We remove "pose_list=" prefix and comment lines, then literal_eval.
    """
    # Strip inline leading up to the first '=' only
    if "pose_list" in text:
        text = text.split("=", 1)[1]
    # Remove full-line comments again (defensive)
    lines = _clean_comment_lines(text.splitlines())
    cleaned = "\n".join(lines).strip()
    # Ensure it starts with '['
    cleaned = cleaned.strip()
    if not cleaned.startswith("["):
        # Maybe was something like " [ ... ]"
        idx = cleaned.find("[")
        cleaned = cleaned[idx:]
    # literal_eval to a Python list
    return ast.literal_eval(cleaned)

def _pose_list_to_actions(flat_list: List[Any]) -> List[Dict[str, Any]]:
    """
    Turn a flat alternating sequence:
      actor_id, payload, actor_id, payload, ..., possibly terminal 0
    into structured actions.
    Payload may be:
      - list or list-of-lists of numerics
      - a string like "left"/"right"
      - absent (for terminal 0/stop)
    """
    actions: List[Dict[str, Any]] = []
    i = 0
    n = len(flat_list)
    while i < n:
        item = flat_list[i]
        if isinstance(item, int) and item in {-1, 0, 1, 2, 3, 4, 5}:
            actor = item
            if actor == 0:
                actions.append({"actor": 0, "name": ActorName[0], "payload": None})
                i += 1
                continue
            # fetch payload if present
            payload = None
            if i + 1 < n:
                payload = flat_list[i + 1]
                i += 2
            else:
                i += 1
            actions.append({
                "actor": actor,
                "name": ActorName.get(actor, f"actor_{actor}"),
                "payload": payload
            })
        else:
            # Unexpected stray item; keep it but mark unknown
            actions.append({"actor": None, "name": "unknown", "payload": item})
            i += 1
    return actions

@dataclass
class StepRecord:
    step_key: str              # e.g., 't_1'
    step_index: int            # 0-based within this episode
    D1: Optional[List[float]]  # length 7 or None
    D2: Optional[List[float]]  # length 7 or None
    actions: List[Dict[str, Any]]  # parsed pose_list actions

def parse_ex_log(path: str) -> List[StepRecord]:
    """
    Parse ex_log.txt into a list of StepRecord.
    Handles 'draw:' header, multiple t_i blocks, D1/D2 lines, and pose_list spanning multiple lines.
    """
    if not os.path.exists(path):
        return []
    raw_lines = _read_text(path)
    # Normalize indentation for robustness (we don't rely on it)
    lines = raw_lines

    # Find blocks for t_*:
    step_starts: List[Tuple[str, int]] = []
    t_pat = re.compile(r'^\s*t_(\d+)\s*:\s*$', re.IGNORECASE)
    for idx, ln in enumerate(lines):
        m = t_pat.match(ln)
        if m:
            step_starts.append((f"t_{int(m.group(1))}", idx))

    step_records: List[StepRecord] = []

    for sidx, (step_key, start_line) in enumerate(step_starts):
        # Determine end line (start of next or end of file)
        end_line = len(lines) - 1
        if sidx + 1 < len(step_starts):
            end_line = step_starts[sidx + 1][1] - 1

        block = lines[start_line:end_line + 1]
        block_text = "\n".join(block)

        # Extract D1 and D2 (single-line or bracketed)
        d1: Optional[List[float]] = None
        d2: Optional[List[float]] = None

        m1 = re.search(r'D1\s*=\s*\[([^\]]+)\]', block_text)
        if m1:
            d1 = _safe_float_list(m1.group(1))
        m2 = re.search(r'D2\s*=\s*\[([^\]]+)\]', block_text)
        if m2:
            d2 = _safe_float_list(m2.group(1))

        # Extract pose_list (may span multiple lines)
        # Find the line index containing 'pose_list'
        pose_line_idx = None
        for j, ln in enumerate(block):
            if "pose_list" in ln:
                pose_line_idx = j
                break

        actions: List[Dict[str, Any]] = []
        if pose_line_idx is not None:
            plist_str, end_idx = _extract_bracketed_list(pose_line_idx, block)
            plist_py = _literal_eval_pose_list(plist_str)
            actions = _pose_list_to_actions(plist_py)

        step_records.append(StepRecord(step_key=step_key, step_index=sidx, D1=d1, D2=d2, actions=actions))

    return step_records

@dataclass
class TimedStream:
    name: str
    seq: torch.Tensor          # (N,) int64
    t_sec: torch.Tensor        # (N,) float64 absolute epoch seconds if known
    pos: torch.Tensor          # (N, K) float32 (K varies by stream)



def _align_clip_streams_by_min_start(
    streams: Dict[str, Optional[TimedStream]],
    use_integer_secs: bool = False,
) -> Dict[str, Optional[TimedStream]]:
    """
    Align and clip multiple TimedStream objects so they:
      - share a common start time (taken from the stream with the fewest samples),
      - have the same length (the common minimum number of samples available after that start),
      - remain in strictly increasing time order.

    Args:
      streams: dict like {"left": TimedStream|None, "right": ..., "nova2": ..., "nova5": ...}
      use_integer_secs: if True, align by integer seconds (floor(secs)); otherwise use full float seconds.

    Returns:
      New dict with the same keys; each present stream is sliced to the same [start, start+L) window.
      Missing streams (None) are passed through unchanged.
    """
    # Collect valid streams
    valid: list[tuple[str, TimedStream]] = []
    for name, s in streams.items():
        if s is not None and s.pos.numel() > 0 and s.t_sec.numel() == s.pos.shape[0]:
            # Ensure time is sorted (defensive)
            if not torch.all(s.t_sec[1:] >= s.t_sec[:-1]):
                order = torch.argsort(s.t_sec)
                s = TimedStream(
                    name=s.name,
                    seq=s.seq[order],
                    t_sec=s.t_sec[order],
                    pos=s.pos[order],
                )
            valid.append((name, s))

    if len(valid) == 0:
        return streams
    if len(valid) == 1:
        # Only one stream available; nothing to align
        name, s = valid[0]
        return {**streams, name: s}

    # 1) Reference = stream with minimum length
    lengths = torch.tensor([s.pos.shape[0] for _, s in valid], dtype=torch.long)
    ref_idx = int(torch.argmin(lengths).item())
    ref_name, ref = valid[ref_idx]

    # 2) Start time t0 from the reference (optionally integer seconds)
    t0 = ref.t_sec[0].item()
    if use_integer_secs:
        t0 = float(int(t0))  # align to integer seconds if desired

    # 3) For each stream, compute first index >= t0
    search_t = torch.tensor(t0, dtype=valid[0][1].t_sec.dtype)
    starts: Dict[str, tuple[TimedStream, int]] = {}
    avail_after_start: list[int] = []
    for name, s in valid:
        i0 = int(torch.searchsorted(s.t_sec, search_t, right=False).item())
        # Clamp inside bounds
        i0 = max(0, min(i0, s.pos.shape[0]))
        starts[name] = (s, i0)
        avail_after_start.append(max(0, s.pos.shape[0] - i0))

    # 4) Common length = min available after each start
    common_len = int(torch.tensor(avail_after_start).min().item())
    if common_len <= 0:
        # Fallback: shift t0 to the maximum initial time among streams to find any common window.
        t0_alt = max(s.t_sec[0].item() for _, s in valid)
        search_t_alt = torch.tensor(float(int(t0_alt)) if use_integer_secs else t0_alt,
                                    dtype=valid[0][1].t_sec.dtype)
        starts.clear(); avail_after_start.clear()
        for name, s in valid:
            i0 = int(torch.searchsorted(s.t_sec, search_t_alt, right=False).item())
            i0 = max(0, min(i0, s.pos.shape[0]))
            starts[name] = (s, i0)
            avail_after_start.append(max(0, s.pos.shape[0] - i0))
        common_len = int(torch.tensor(avail_after_start).min().item())
        if common_len <= 0:
            # Nothing in common; return original streams unchanged
            return streams

    # 5) Slice all present streams to the common window
    out: Dict[str, Optional[TimedStream]] = {}
    for key, s in streams.items():
        if s is None or s.pos.numel() == 0:
            out[key] = s
            continue
        s0, i0 = starts.get(key, (s, 0))
        i1 = i0 + common_len
        out[key] = TimedStream(
            name=s0.name,
            seq=s0.seq[i0:i1].clone(),
            t_sec=s0.t_sec[i0:i1].clone(),
            pos=s0.pos[i0:i1].clone(),
        )
    return out

def parse_timed_positions(path: str, expected_name: str) -> Optional[TimedStream]:
    """
    Parse files like left.txt/right.txt/nova2.txt/nova5.txt
    Blocks of:
      seq: 0
      secs: 1755682654
      nsecs: 406556367
      position: [254.0, 0.0, ...]
    """
    if not os.path.exists(path):
        return None

    lines = _read_text(path)
    seqs: List[int] = []
    secs: List[int] = []
    nsecs: List[int] = []
    positions: List[List[float]] = []

    cur_seq = None
    cur_secs = None
    cur_nsecs = None
    cur_pos = None

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("seq:"):
            cur_seq = int(ln.split(":", 1)[1].strip())
        elif ln.startswith("secs:"):
            cur_secs = int(ln.split(":", 1)[1].strip())
        elif ln.startswith("nsecs:"):
            cur_nsecs = int(ln.split(":", 1)[1].strip())
        elif ln.startswith("position:"):
            # everything after ':' is a Python list
            val = ln.split(":", 1)[1].strip()
            cur_pos = [float(x) for x in ast.literal_eval(val)]

            if cur_seq is None or cur_secs is None or cur_nsecs is None:
                # malformed entry; skip
                cur_seq = cur_secs = cur_nsecs = None
                cur_pos = None
                continue

            seqs.append(cur_seq)
            secs.append(cur_secs)
            nsecs.append(cur_nsecs)
            positions.append(cur_pos)

            # reset
            cur_seq = cur_secs = cur_nsecs = None
            cur_pos = None

    if not seqs:
        return None

    t = torch.tensor(secs, dtype=torch.float64) + torch.tensor(nsecs, dtype=torch.float64) * 1e-9
    return TimedStream(
        name=expected_name,
        seq=torch.tensor(seqs, dtype=torch.long),
        t_sec=t,
        pos=torch.tensor(positions, dtype=torch.float32)
    )

# ===========================
# ----- Video Backends ------
# ===========================

class _VideoBackend:
    def __init__(self, path: str):
        self.path = path
        self.num_frames = None
        self.fps = None
        self.width = None
        self.height = None
        self.duration = None

    def get_frame(self, frame_idx: int) -> torch.Tensor:
        raise NotImplementedError

class _TorchvisionVideoReader(_VideoBackend):
    def __init__(self, path: str):
        super().__init__(path)
        from torchvision.io import VideoReader
        self.reader = VideoReader(path, "video")
        meta = self.reader.get_metadata()["video"]
        # fps may be a list of frame rates; take the first
        self.fps = float(meta["fps"][0] if isinstance(meta["fps"], list) else meta["fps"])
        # Determine num_frames by iterating once (metadata can be missing)
        # We avoid consuming the reader permanently; so we open a temp reader.
        tmp = VideoReader(path, "video")
        n = 0
        for _ in tmp:
            n += 1
        self.num_frames = n
        # Re-open the main reader after counting (safer for some versions)
        self.reader = VideoReader(path, "video")
        self.duration = self.num_frames / self.fps if self.fps and self.num_frames else None
        # width/height not always in metadata; probe first frame
        first = next(self.reader)
        frame = first["data"]  # (H, W, C), uint8
        self.height, self.width = int(frame.shape[0]), int(frame.shape[1])
        # reset to beginning
        self.reader.seek(0.0)

    def get_frame(self, frame_idx: int) -> torch.Tensor:
        # seek by time
        t = max(frame_idx, 0) / max(self.fps, 1.0)
        self.reader.seek(t)
        fr = next(self.reader)
        frame = fr["data"]  # HWC uint8
        # to CHW float32 [0,1]
        return frame.permute(2, 0, 1).to(torch.float32) / 255.0

class _OpenCVVideoReader(_VideoBackend):
    def __init__(self, path: str):
        super().__init__(path)
        import cv2
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.num_frames / self.fps if self.fps and self.num_frames else None

    def get_frame(self, frame_idx: int) -> torch.Tensor:
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame_bgr = self.cap.read()
        if not ok:
            # fallback: last frame
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, self.num_frames - 1)
            ok, frame_bgr = self.cap.read()
            if not ok:
                raise RuntimeError("Failed to decode frame.")
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        # torch.from_numpy expects numpy; convert safely
        import numpy as np
        arr = np.ascontiguousarray(frame_rgb)
        t = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32) / 255.0
        return t

class _TorchvisionReadVideo(_VideoBackend):
    def __init__(self, path: str):
        super().__init__(path)
        from torchvision.io import read_video
        # This loads entire video into memory; use only as last resort
        video, audio, info = read_video(path, pts_unit="sec")
        # video: (T, H, W, C) uint8
        self.buffer = video  # keep in memory
        self.num_frames = int(video.shape[0])
        self.height = int(video.shape[1])
        self.width = int(video.shape[2])
        self.fps = float(info.get("video_fps", 30.0))
        self.duration = self.num_frames / self.fps if self.fps and self.num_frames else None

    def get_frame(self, frame_idx: int) -> torch.Tensor:
        idx = min(max(int(frame_idx), 0), self.num_frames - 1)
        frame = self.buffer[idx]  # HWC uint8
        return frame.permute(2, 0, 1).to(torch.float32) / 255.0

def make_video_backend(path: str, preferred: Optional[str] = None) -> Optional[_VideoBackend]:
    if not os.path.exists(path):
        return None
    # preferred: "torchvision", "opencv", "read_video"
    tried = []
    def _try_tv_vr():
        from torchvision.io import VideoReader  # noqa: F401
        return _TorchvisionVideoReader(path)
    def _try_cv2():
        import cv2  # noqa: F401
        return _OpenCVVideoReader(path)
    def _try_tv_read():
        from torchvision.io import read_video  # noqa: F401
        return _TorchvisionReadVideo(path)

    candidates = {
        "torchvision": _try_tv_vr,
        "opencv": _try_cv2,
        "read_video": _try_tv_read,
    }
    order = [preferred] if preferred in candidates else []
    order += [k for k in ["torchvision", "opencv", "read_video"] if k not in order]

    for key in order:
        try:
            return candidates[key]()
        except Exception as e:
            tried.append((key, str(e)))
            continue
    # No backend available
    return None

# ===========================
# --------- Dataset ---------
# ===========================

@dataclass
class EpisodeData:
    episode_name: str
    episode_dir: str
    video_path: Optional[str]
    video: Optional[_VideoBackend]
    steps: List[StepRecord]
    left: Optional[TimedStream]
    right: Optional[TimedStream]
    nova2: Optional[TimedStream]
    nova5: Optional[TimedStream]
    # Optional absolute epoch (secs) for the video frame 0 (if known)
    video_epoch0: Optional[float] = None

class DrawDataset(Dataset):
    """
    A PyTorch Dataset for episodes under an experiment folder like:

        draw/
          draw_t1/
            draw_t1.mp4
            ex_log.txt
            left.txt
            right.txt
            nova2.txt
            nova5.txt
          draw_t2/
            ...

    Indexing modes:
      - index_mode="event": each sample = one ex_log step (t_i). We map step index -> representative frame
                            and -> seq indices of timed streams via proportional mapping by default.
      - index_mode="frame": each sample = one video frame (decode or just index). We map frame index
                            -> nearest ex_log step (by proportional mapping) and seq indices similarly.

    True time alignment:
      If you pass episode_time_offsets={"draw_t1": 1_755_682_654.406556, ...} and
      align_mode="time", the dataset aligns frames to timed streams by nearest absolute epoch
      (frame_time = video_epoch0 + frame_idx / fps). When not provided, it falls back to
      robust proportional mapping (align_mode="relative").

    Returns a dictionary with:
      - 'episode', 'index_in_episode', 'global_index'
      - 'frame' (FloatTensor CxHxW in [0,1]) if decode_video=True, else None
      - 'frame_idx' (int), 'fps', 'num_frames'
      - 'D1_pose' (7,), 'D2_pose' (7,) or None
      - 'pose_actions': list of {actor:int|0|-1|None, name:str, payload: list|str|None}
      - streams for 'left'/'right'/'nova2'/'nova5': {'seq': int, 'position': FloatTensor(K,)} or None

    Collation:
      Use DrawDataset.collate_fn to stack tensors and keep variable-length parts as lists.
    """
    def __init__(
        self,
        root: str,
        index_mode: str = "event",
        decode_video: bool = True,
        video_backend_preference: Optional[str] = None,  # "torchvision"|"opencv"|"read_video"
        transform = None,  # optional image transform to apply to frame (expects CxHxW FloatTensor)
        align_mode: str = "relative",  # "relative" or "time"
        episode_time_offsets: Optional[Dict[str, float]] = None,  # epoch seconds for frame0 per episode
    ):
        super().__init__()
        assert index_mode in {"event", "frame", "stream"}
        assert align_mode in {"relative", "time"}
        self.root = root
        self.index_mode = index_mode
        self.decode_video = decode_video
        self.video_backend_preference = video_backend_preference
        self.transform = transform
        self.align_mode = align_mode
        self.episode_time_offsets = episode_time_offsets or {}

        self.episodes: List[EpisodeData] = []
        self.index: List[Dict[str, Any]] = []  # flattened index over episodes

        self._build_index()

    # ---------- public helpers ----------

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate variable-length 'pose_actions' and optional streams gracefully.
        Frames are stacked if present and same size (ensure resize in transform if batching).
        """
        out: Dict[str, Any] = {}
        # scalar metadata
        out["episode"] = [b["episode"] for b in batch]
        out["index_in_episode"] = torch.tensor([b["index_in_episode"] for b in batch], dtype=torch.long)
        out["global_index"] = torch.tensor([b["global_index"] for b in batch], dtype=torch.long)
        out["frame_idx"] = torch.tensor([b["frame_idx"] for b in batch], dtype=torch.long)
        out["fps"] = torch.tensor([b["fps"] for b in batch], dtype=torch.float32)
        out["num_frames"] = torch.tensor([b["num_frames"] for b in batch], dtype=torch.long)

        # frame tensor (if any)
        if batch[0].get("frame") is not None:
            # Verify shapes match; if not, raise to encourage resize in transform
            shapes = [tuple(b["frame"].shape) for b in batch if b.get("frame") is not None]
            if len(set(shapes)) > 1:
                raise ValueError(
                    "Frames in a batch have different shapes. "
                    "Provide a transform that resizes them uniformly."
                )
            out["frame"] = torch.stack([b["frame"] for b in batch], dim=0)
        else:
            out["frame"] = None

        # D1/D2 (may be None)
        def stack_optional(key: str):
            vals = [b[key] for b in batch]
            if any(v is None for v in vals):
                out[key] = vals  # keep as list if any None
            else:
                out[key] = torch.stack(vals, dim=0)
        stack_optional("D1_pose")
        stack_optional("D2_pose")

        # pose_actions: keep as list of lists
        out["pose_actions"] = [b["pose_actions"] for b in batch]

        # timed streams
        for stream_name in ["left", "right", "nova2", "nova5"]:
            seqs: List[Optional[int]] = []
            pos_list: List[Optional[torch.Tensor]] = []
            for b in batch:
                st = b.get(stream_name)
                if st is None:
                    seqs.append(None); pos_list.append(None)
                else:
                    seqs.append(int(st["seq"]))
                    pos_list.append(st["position"])
            # keep as list if any None or sizes vary
            if any(x is None for x in pos_list) or len({p.shape if p is not None else None for p in pos_list}) > 1:
                out[f"{stream_name}_seq"] = seqs
                out[f"{stream_name}_position"] = pos_list
            else:
                out[f"{stream_name}_seq"] = torch.tensor(seqs, dtype=torch.long)
                out[f"{stream_name}_position"] = torch.stack(pos_list, dim=0)

        return out

    # ---------- core dataset API ----------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        si = self.index[idx]
        ep: EpisodeData = self.episodes[si["episode_idx"]]
        step: Optional[StepRecord] = si.get("step")  # may be None in frame mode
        frame_idx: int = si["frame_idx"]

        # Decode frame if requested
        frame_tensor = None
        if self.decode_video and ep.video is not None:
            frame_tensor = ep.video.get_frame(frame_idx)  # FloatTensor CxHxW in [0,1]
            if self.transform is not None:
                frame_tensor = self.transform(frame_tensor)

        # D1/D2 come from step in event mode; in frame mode we map to nearest step
        D1_tensor = None
        D2_tensor = None
        actions = []
        if step is not None:
            if step.D1 is not None:
                D1_tensor = torch.tensor(step.D1, dtype=torch.float32)
            if step.D2 is not None:
                D2_tensor = torch.tensor(step.D2, dtype=torch.float32)
            actions = step.actions

        # Streams: pick aligned sample
        left = self._pick_stream(ep.left, si)
        right = self._pick_stream(ep.right, si)
        nova2 = self._pick_stream(ep.nova2, si)
        nova5 = self._pick_stream(ep.nova5, si)

        sample = {
            "episode": ep.episode_name,
            "index_in_episode": si["index_in_episode"],
            "global_index": idx,
            "frame": frame_tensor,
            "frame_idx": frame_idx,
            "fps": float(si["fps"]),
            "num_frames": int(si["num_frames"]),
            "D1_pose": D1_tensor,
            "D2_pose": D2_tensor,
            "pose_actions": actions,
            "left": left,
            "right": right,
            "nova2": nova2,
            "nova5": nova5,
        }
        return sample

    # ---------- internal helpers ----------

    def _build_index(self) -> None:
        # Discover episodes
        candidates = sorted(
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d)) and re.match(r"^draw_t\d+$", d)
        )
        ep_counter = 0
        for ep_name in candidates:
            ep_dir = os.path.join(self.root, ep_name)
            video_path = os.path.join(ep_dir, f"{ep_name}.mp4")
            ex_log_path = os.path.join(ep_dir, "ex_log.txt")
            left_path = os.path.join(ep_dir, "left.txt")
            right_path = os.path.join(ep_dir, "right.txt")
            nova2_path = os.path.join(ep_dir, "nova2.txt")
            nova5_path = os.path.join(ep_dir, "nova5.txt")

            # Parse logs
            steps = parse_ex_log(ex_log_path)

            # Timed streams
            # Timed streams (parse)
            left  = parse_timed_positions(left_path,  "left")
            right = parse_timed_positions(right_path, "right")
            nova2 = parse_timed_positions(nova2_path, "nova2")
            nova5 = parse_timed_positions(nova5_path, "nova5")

            # ---- Align & clip so all four share the same start time and length ----
            streams = _align_clip_streams_by_min_start(
                {"left": left, "right": right, "nova2": nova2, "nova5": nova5},
                use_integer_secs=False  # set True if you want to align by integer secs only
            )
            left, right, nova2, nova5 = streams["left"], streams["right"], streams["nova2"], streams["nova5"]


            # Video
            video = None
            fps = 30.0
            num_frames = 0
            if os.path.exists(video_path):
                video = make_video_backend(video_path, self.video_backend_preference)
                if video is not None and video.fps:
                    fps = float(video.fps)
                if video is not None and video.num_frames:
                    num_frames = int(video.num_frames)

                    
            def _stream_len_or_zero(*ss):
                for s in ss:
                    if s is not None and s.pos.numel() > 0:
                        return int(s.pos.shape[0])
                return 0

            stream_len = _stream_len_or_zero(left, right, nova2, nova5)

            # Video meta as you already compute above: (video, fps, num_frames)
            # steps already parsed above.

            ep = EpisodeData(
                episode_name=ep_name,
                episode_dir=ep_dir,
                video_path=video_path if os.path.exists(video_path) else None,
                video=video,
                steps=steps,
                left=left, right=right, nova2=nova2, nova5=nova5,
                video_epoch0=self.episode_time_offsets.get(ep_name)
            )
            ep_idx = len(self.episodes)
            self.episodes.append(ep)

            # ---------- NEW: stream indexing ----------
            if self.index_mode == "stream":
                if stream_len <= 0:
                    # nothing usable; skip indexing this episode
                    continue
                for i in range(stream_len):
                    # Map this stream index to a representative ex_log step (so D1/D2/actions are available)
                    sidx = self._map_stream_to_step(i, stream_len, len(steps))
                    step_obj = steps[sidx] if (steps and 0 <= sidx < len(steps)) else None

                    # If you want frames decoded, pick the nearest video frame for this stream index
                    frame_idx_video = self._map_stream_to_frame(i, stream_len, num_frames)

                    self.index.append({
                        "episode_idx": ep_idx,
                        "index_in_episode": i,       # stream index within this episode
                        "step": step_obj,            # keep steps so D1/D2/actions work downstream
                        "frame_idx": frame_idx_video,# decode this frame if decode_video=True
                        "fps": fps,
                        "num_frames": num_frames,    # actual video frames (not stream_len)
                        "stream_idx": i,             # <-- used for choosing the exact row from streams
                        "stream_len": stream_len,
                    })
                continue  # prevent falling through to event/frame indexing

            if self.index_mode == "event":
                if not steps:
                    continue
                for i, step_obj in enumerate(steps):
                    self.index.append({
                        "episode_idx": ep_idx,
                        "index_in_episode": i,
                        "step": step_obj,
                        "frame_idx": self._map_step_to_frame(i, len(steps), num_frames),
                        "fps": fps,
                        "num_frames": num_frames,
                    })
                continue

            if self.index_mode == "frame":
                if num_frames <= 0:
                    continue
                for frame_idx in range(num_frames):
                    step_idx = self._map_frame_to_step(frame_idx, num_frames, len(steps))
                    step_obj = steps[step_idx] if steps and 0 <= step_idx < len(steps) else None
                    self.index.append({
                        "episode_idx": ep_idx,
                        "index_in_episode": frame_idx,
                        "step": step_obj,
                        "frame_idx": frame_idx,
                        "fps": fps,
                        "num_frames": num_frames,
                    })

    @staticmethod
    def _map_step_to_frame(step_index: int, num_steps: int, num_frames: int) -> int:
        if num_frames <= 0:
            return 0
        if num_steps <= 1:
            return 0
        r = step_index / max(num_steps - 1, 1)
        return int(round(r * (num_frames - 1)))

    @staticmethod
    def _map_frame_to_step(frame_idx: int, num_frames: int, num_steps: int) -> int:
        if num_steps <= 0:
            return 0
        if num_frames <= 1:
            return 0
        r = frame_idx / max(num_frames - 1, 1)
        return int(round(r * (num_steps - 1)))
    
    @staticmethod
    def _map_stream_to_step(stream_idx: int, stream_len: int, num_steps: int) -> int:
        if num_steps <= 0 or stream_len <= 1:
            return 0
        r = stream_idx / max(stream_len - 1, 1)
        return int(round(r * (num_steps - 1)))

    @staticmethod
    def _map_stream_to_frame(stream_idx: int, stream_len: int, num_frames: int) -> int:
        if num_frames <= 0 or stream_len <= 1:
            return 0
        r = stream_idx / max(stream_len - 1, 1)
        return int(round(r * (num_frames - 1)))

    def _pick_stream(self, stream: Optional[TimedStream], si: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if stream is None or stream.pos.numel() == 0:
            return None

        # Choose seq index either by relative mapping (default) or true time alignment if available
        if self.align_mode == "time":
            # need episode video epoch0 and fps
            ep = self.episodes[si["episode_idx"]]
            if ep.video is None or ep.video_epoch0 is None or not ep.video.fps:
                # fall back to relative
                seq_idx = self._relative_seq_index(stream, si)
            else:
                frame_idx = si["frame_idx"]
                t_frame = ep.video_epoch0 + frame_idx / float(ep.video.fps)
                # nearest in absolute epoch
                idx = torch.argmin(torch.abs(stream.t_sec - t_frame)).item()
                seq_idx = int(idx)
        else:
            seq_idx = self._relative_seq_index(stream, si)

        # clip bounds
        seq_idx = max(0, min(seq_idx, stream.pos.shape[0] - 1))
        return {"seq": int(stream.seq[seq_idx].item()), "position": stream.pos[seq_idx]}


    def _relative_seq_index(self, stream: TimedStream, si: Dict[str, Any]) -> int:
        N = stream.pos.shape[0]

        if self.index_mode == "stream":
            if N <= 1:
                return 0
            # pick the exact synchronized row
            return max(0, min(int(si.get("stream_idx", 0)), N - 1))

        if self.index_mode == "event":
            step: StepRecord = si["step"]
            if N <= 1 or step is None:
                return 0
            r = step.step_index / max(len(self.episodes[si["episode_idx"]].steps) - 1, 1)
            return int(round(r * (N - 1)))

        # frame mode (unchanged)
        if N <= 1 or si["num_frames"] <= 1:
            return 0
        r = si["frame_idx"] / max(si["num_frames"] - 1, 1)
        return int(round(r * (N - 1)))
    

# ===========================
# --------- Usage -----------
# ===========================

# train.py
from typing import Dict, Any, Tuple, Optional
import re
import torch
import torch.nn as nn
from model import MultiModalActTransformer
from torch.utils.data import Sampler


class PerEpisodeStreamBatchSampler(Sampler):
    """
    Yields one batch per episode, where the batch contains all indices for that episode.
    Requires DrawDataset(index_mode='stream').
    """
    def __init__(self, dataset: DrawDataset, drop_empty: bool = True):
        if getattr(dataset, "index_mode", None) != "stream":
            raise ValueError("PerEpisodeStreamBatchSampler requires DrawDataset(index_mode='stream').")
        self.dataset = dataset
        # Group dataset indices by episode
        from collections import defaultdict
        by_ep = defaultdict(list)
        for i, si in enumerate(dataset.index):
            by_ep[ si["episode_idx"] ].append(i)
        # Preserve episode order
        self._batches = []
        for ep_idx in range(len(dataset.episodes)):
            idxs = by_ep.get(ep_idx, [])
            if idxs or not drop_empty:
                self._batches.append(idxs)

    def __iter__(self):
        for batch in self._batches:
            if batch:     # skip empties if requested
                yield batch

    def __len__(self):
        # number of non-empty episodes
        return sum(1 for b in self._batches if b)
# ----------------- Normalization helpers -----------------

def scale_pre_hand(x: torch.Tensor) -> torch.Tensor:
    # Hands come in 0..255
    return x / 255.0

def scale_pre_arm(x: torch.Tensor) -> torch.Tensor:
    # Arm joint degrees -> roughly scale to [-1,1] by dividing by 180
    return x / 180.0

def scale_object_pose(x: torch.Tensor) -> torch.Tensor:
    # D1/D2 pose: [x,y,z, qw,qx,qy,qz] already ~[-1,1] for quats, metric for xyz.
    # Light scale xyz to a reasonable range if necessary (here: pass-through).
    return x

def scale_targets(actor: str, x: torch.Tensor) -> torch.Tensor:
    if actor in ("LeftHand", "RightHand"):
        return x / 255.0
    elif actor in ("nova2", "nova5"):
        return x / 180.0
    return x


# ----------------- Task parsing -----------------

class TaskVocab:
    def __init__(self):
        self._tok2id = {}
        self._id2tok = []

    def encode(self, episode_names) -> torch.Tensor:
        # episode like "draw_t1", "draw_t10"
        ids = []
        for name in episode_names:
            m = re.match(r"([A-Za-z]+)_t(\d+)", name)
            task = m.group(1) if m else name
            if task not in self._tok2id:
                self._tok2id[task] = len(self._id2tok)
                self._id2tok.append(task)
            ids.append(self._tok2id[task])
        return torch.tensor(ids, dtype=torch.long)

    @property
    def size(self):
        return len(self._id2tok) if self._id2tok else 1


# ----------------- Targets from pose_actions -----------------

THREAD_MAP = {"none": 0, "left": 1, "right": 2}

def _last_numeric_payload(rows: list, expected_dim: int) -> Optional[torch.Tensor]:
    """
    rows is something like [[...]] or [[...],[...]] — we pick the last row.
    Returns (expected_dim,) tensor or None if not found/shape mismatch.
    """
    try:
        if not rows:
            return None
        last = rows[-1]
        if isinstance(last, list) and len(last) == expected_dim:
            return torch.tensor(last, dtype=torch.float32)
        # Sometimes payload itself is [[...],[...]] (nested)
        if isinstance(rows[0], list) and isinstance(rows[0][0], (int, float)):
            # rows is list of vectors already
            if len(rows[-1]) == expected_dim:
                return torch.tensor(rows[-1], dtype=torch.float32)
        return None
    except Exception:
        return None

def build_targets_from_pose_actions(pose_actions_batch: list) -> Dict[str, torch.Tensor]:
    """
    pose_actions_batch: list length B; each element is a list of dicts with keys:
      {'actor': int, 'name': 'nova2'|'nova5'|'LeftHand'|'RightHand'|'thread'|'stop', 'payload': ...}
    We pick the **last** payload for each actor (per sample).
    Returns targets (tensors) with NaNs where missing, and classification labels for thread.
    """
    B = len(pose_actions_batch)
    # Initialize with NaNs / ignore labels
    tgt = {
        "nova2":      torch.full((B, 6),  float("nan")),
        "nova5":      torch.full((B, 6),  float("nan")),
        "LeftHand":   torch.full((B,10),  float("nan")),
        "RightHand":  torch.full((B,10),  float("nan")),
        "thread":     torch.full((B,),    -100, dtype=torch.long),  # ignore_index
        "stop":       torch.full((B,1),   float("nan")),
    }

    for b, events in enumerate(pose_actions_batch):
        last_numeric = {"nova2": None, "nova5": None, "LeftHand": None, "RightHand": None}
        last_thread: Optional[int] = None
        last_stop: Optional[float] = None

        for ev in events:
            name = ev.get("name")
            payload = ev.get("payload")

            if name in ("nova2", "nova5"):
                # payload like [[...], [...]]; take last vector of length 6
                if isinstance(payload, list):
                    # some samples: [[-48.9, ...], [-48.98, ...]]
                    flat = payload if isinstance(payload[0], list) else [payload]
                    vec = _last_numeric_payload(flat, 6)
                    if vec is not None:
                        last_numeric[name] = vec

            elif name in ("LeftHand", "RightHand"):
                # payload like [[255,0,255,...]] (10 dims)
                if isinstance(payload, list):
                    flat = payload if isinstance(payload[0], list) else [payload]
                    vec = _last_numeric_payload(flat, 10)
                    if vec is not None:
                        last_numeric[name] = vec

            elif name == "thread":
                # payload is 'left' or 'right'
                if isinstance(payload, str):
                    last_thread = THREAD_MAP.get(payload.lower(), THREAD_MAP["none"])

            elif name == "stop":
                # payload is None; just set flag 1.0
                last_stop = 1.0

        # Assign if found
        for k, v in last_numeric.items():
            if v is not None:
                tgt[k][b] = v
        if last_thread is not None:
            tgt["thread"][b] = last_thread
        if last_stop is not None:
            tgt["stop"][b, 0] = last_stop

    return tgt


# ----------------- Loss -------------------------------------------------------

class MultiHeadLoss(nn.Module):
    def __init__(self, thread_ignore_index: int = -100,
                 w_arm: float = 1.0, w_hand: float = 1.0,
                 w_thread: float = 0.5, w_stop: float = 0.25):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.ce  = nn.CrossEntropyLoss(ignore_index=thread_ignore_index)
        self.bce = nn.BCEWithLogitsLoss()

        self.w_arm = w_arm
        self.w_hand = w_hand
        self.w_thread = w_thread
        self.w_stop = w_stop

    def forward(self, preds: Dict[str, torch.Tensor], tgts: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}

        # --- Arms (MSE with NaN masking) ---
        total = 0.0
        for name in ("nova2", "nova5"):
            y = tgts[name]                    # (B,6)
            p = preds[name]                   # (B,6)
            mask = ~torch.isnan(y)
            if mask.any():
                mse = self.mse(p[mask], y[mask]).mean()
                loss_dict[name] = mse.item()
                total = total + self.w_arm * mse
            else:
                loss_dict[name] = 0.0

        # --- Hands (MSE with NaN masking) ---
        for name in ("LeftHand", "RightHand"):
            y = tgts[name]                    # (B,10)
            p = preds[name]                   # (B,10)
            mask = ~torch.isnan(y)
            if mask.any():
                mse = self.mse(p[mask], y[mask]).mean()
                loss_dict[name] = mse.item()
                total = total + self.w_hand * mse
            else:
                loss_dict[name] = 0.0

        # --- Thread (CE) ---
        y_thr = tgts["thread"]                # (B,)
        thr_loss = self.ce(preds["thread_logits"], y_thr)
        loss_dict["thread"] = thr_loss.item()
        total = total + self.w_thread * thr_loss

        # --- Stop (BCE) with NaN masking ---
        y_stop = tgts["stop"]                 # (B,1)
        p_stop = preds["stop_logit"]          # (B,1)
        mask_stop = ~torch.isnan(y_stop)
        if mask_stop.any():
            bce = self.bce(p_stop[mask_stop], y_stop[mask_stop])
            loss_dict["stop"] = bce.item()
            total = total + self.w_stop * bce
        else:
            loss_dict["stop"] = 0.0

        loss_dict["total"] = float(total.item())
        return total, loss_dict


# ----------------- Batch -> model tensors -------------------------------------

def batch_to_model_inputs(batch: Dict[str, Any], task_vocab: TaskVocab, device: torch.device):
    """
    Maps your loader's dict to properly scaled tensors for the model.
    Returns:
      inputs: dict of tensors for model.forward
      targets: dict of appropriately scaled targets
    """
    # (B,)
    task_id = task_vocab.encode(batch["episode"]).to(device)

    # (B,7) object poses
    d1 = scale_object_pose(batch["D1_pose"].to(device).float())
    d2 = scale_object_pose(batch["D2_pose"].to(device).float())

    # Pre-timestamp signals
    pre_left  = scale_pre_hand(batch["left_position"].to(device).float())     # (B,10)
    pre_right = scale_pre_hand(batch["right_position"].to(device).float())    # (B,10)
    pre_n2    = scale_pre_arm(batch["nova2_position"].to(device).float())     # (B,6)
    pre_n5    = scale_pre_arm(batch["nova5_position"].to(device).float())     # (B,6)

    img = batch["frame"].to(device).float() if batch.get("frame") is not None else None

    # Targets
    tgt_raw = build_targets_from_pose_actions(batch["pose_actions"])
    # Scale numeric targets to the same scale the model sees
    for k in ("nova2", "nova5", "LeftHand", "RightHand"):
        y = tgt_raw[k]
        # Leave NaNs as-is; scale valid entries
        mask = ~torch.isnan(y)
        if mask.any():
            y_scaled = scale_targets(k, y.clone())
            y[mask] = y_scaled[mask]
        tgt_raw[k] = y
    # thread already mapped to class ids; stop is 0/1 (NaN mask)

    # Device
    for k in ("nova2", "nova5", "LeftHand", "RightHand", "stop"):
        tgt_raw[k] = tgt_raw[k].to(device)
    tgt_raw["thread"] = tgt_raw["thread"].to(device)

    inputs = dict(
        task_id=task_id,
        d1_pose=d1, d2_pose=d2,
        pre_nova2=pre_n2, pre_nova5=pre_n5,
        pre_left=pre_left, pre_right=pre_right,
        img=img
    )
    return inputs, tgt_raw


# ----------------- Train step -------------------------------------------------

def train_one_epoch(
    model: MultiModalActTransformer,
    optimizer: torch.optim.Optimizer,
    dataloader,               # yields your batch dict (like the one you pasted)
    device: torch.device,
    task_vocab: TaskVocab,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
):
    model.train()
    loss_fn = MultiHeadLoss()
    for it, batch in enumerate(dataloader):
        inputs, targets = batch_to_model_inputs(batch, task_vocab, device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            preds = model(**inputs)
            loss, loss_dict = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                preds = model(**inputs)
                loss, loss_dict = loss_fn(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if it % 10 == 0:
            # Lightweight log
            msg = "it {:05d} | total {:.4f} | n2 {:.4f} n5 {:.4f} L {:.4f} R {:.4f} thr {:.4f} stop {:.4f}".format(
                it, loss_dict["total"], loss_dict["nova2"], loss_dict["nova5"],
                loss_dict["LeftHand"], loss_dict["RightHand"], loss_dict["thread"], loss_dict["stop"]
            )
            print(msg)


# ----------------- Putting it together ---------------------------------------

def build_model_and_optimizer(use_images: bool, task_vocab_size: int, lr: float = 3e-4):
    model = MultiModalActTransformer(
        d_model=256, nhead=8, num_layers=6, dim_feedforward=1024,
        dropout=0.1, num_tasks=max(task_vocab_size, 1), use_images=use_images
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    return model, optimizer

"""
Legacy inline training example kept for reference only. The active CLI starts at
parse_args() below.

if False and __name__ == "__main__":
    Example usage:

    root = "draw"  # experiment folder containing draw_t1, draw_t2, ...
    ds = DrawDataset(
        root=root,
        index_mode="event",      # or "frame"
        decode_video=True,       # set False to skip decoding and only get frame indices
        video_backend_preference=None,  # "torchvision"|"opencv"|"read_video"|None
        transform=None,          # e.g., torchvision transforms expecting CxHxW FloatTensor
        align_mode="relative",   # "time" if you provide per-episode epoch0
        episode_time_offsets={   # only needed for align_mode="time"
            # "draw_t1": 1755682654.406556,  # secs + nsecs*1e-9 at video frame 0
        }
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=DrawDataset.collate_fn)

    for batch in loader:
        # batch["frame"]: (B, C, H, W) or None
        # batch["D1_pose"]: (B,7) or list with Nones
        # batch["pose_actions"]: list of length B, each a list of actions
        # batch["left_position"]: (B,K) or list of Tensors if K differs/missing
        print(batch["episode"][:2], batch["frame_idx"][:2])
        break
    root = "exp_1/draw"  # experiment folder containing draw_t1, draw_t2, ...
    ds = DrawDataset(
        root=root,
        index_mode="event",      # or "frame"
        decode_video=True,       # set False to skip decoding and only get frame indices
        video_backend_preference=None,  # "torchvision"|"opencv"|"read_video"|None
        transform=None,          # e.g., torchvision transforms expecting CxHxW FloatTensor
        align_mode="relative",   # "time" if you provide per-episode epoch0
        episode_time_offsets={   # only needed for align_mode="time"
            # "draw_t1": 1755682654.406556,  # secs + nsecs*1e-9 at video frame 0
        }
    )

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=10, shuffle=False, num_workers=0, collate_fn=DrawDataset.collate_fn)

# Example usage (pseudo):
    num_epochs=100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task_vocab = TaskVocab()
    model, optimizer = build_model_and_optimizer(use_images=False, task_vocab_size=task_vocab.size)
    model.to(device)
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, loader, device, task_vocab)





#     root = "exp_1/draw"  # experiment folder containing draw_t1, draw_t2, ...
#     ds = DrawDataset(
#         root=root,
#         index_mode="stream",      # or "frame"
#         decode_video=True,       # set False to skip decoding and only get frame indices
#         video_backend_preference=None,  # "torchvision"|"opencv"|"read_video"|None
#         transform=None,          # e.g., torchvision transforms expecting CxHxW FloatTensor
#         align_mode="relative",   # "time" if you provide per-episode epoch0
#         episode_time_offsets={   # only needed for align_mode="time"
#             # "draw_t1": 1755682654.406556,  # secs + nsecs*1e-9 at video frame 0
#         }
#     )

#     from torch.utils.data import DataLoader
#     loader = DataLoader(ds, batch_size=10, shuffle=False, num_workers=0, collate_fn=DrawDataset.collate_fn)
#     batch_sampler = PerEpisodeStreamBatchSampler(ds)
#     loader = DataLoader(ds, batch_sampler=batch_sampler, num_workers=0,
#                     collate_fn=DrawDataset.collate_fn)

# # Example usage (pseudo):
#     num_epochs=100
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     task_vocab = TaskVocab()
#     model, optimizer = build_model_and_optimizer(use_images=False, task_vocab_size=task_vocab.size)
#     model.to(device)
#     for epoch in range(num_epochs):
#         train_one_epoch(model, optimizer, loader, device, task_vocab)

"""


def parse_args():
    parser = argparse.ArgumentParser(description="Train the cad2real multi-modal action model.")
    parser.add_argument("--data-root", default="dataset/draw", help="Task dataset folder containing draw_t* episodes.")
    parser.add_argument("--index-mode", choices=("event", "frame", "stream"), default="event")
    parser.add_argument("--decode-video", action="store_true", help="Decode video frames and include them in samples.")
    parser.add_argument("--video-backend", choices=("torchvision", "opencv", "read_video"), default=None)
    parser.add_argument("--align-mode", choices=("relative", "time"), default="relative")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-images", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--save-every", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    ds = DrawDataset(
        root=args.data_root,
        index_mode=args.index_mode,
        decode_video=args.decode_video,
        video_backend_preference=args.video_backend,
        transform=None,
        align_mode=args.align_mode,
        episode_time_offsets={},
    )
    if len(ds) == 0:
        raise RuntimeError("No samples found in {} with index_mode={}".format(args.data_root, args.index_mode))

    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(args.index_mode != "stream"),
        num_workers=args.num_workers,
        collate_fn=DrawDataset.collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_vocab = TaskVocab()
    task_vocab.encode([ep.episode_name for ep in ds.episodes])
    model, optimizer = build_model_and_optimizer(
        use_images=args.use_images,
        task_vocab_size=task_vocab.size,
        lr=args.lr,
    )
    model.to(device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print("epoch {}/{}".format(epoch + 1, args.epochs))
        train_one_epoch(model, optimizer, loader, device, task_vocab)
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = checkpoint_dir / "policy_epoch_{:04d}.pt".format(epoch + 1)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "task_vocab": {
                        "tok2id": task_vocab._tok2id,
                        "id2tok": task_vocab._id2tok,
                    },
                    "args": vars(args),
                },
                ckpt_path,
            )
            print("saved checkpoint: {}".format(ckpt_path))


if __name__ == "__main__":
    main()
