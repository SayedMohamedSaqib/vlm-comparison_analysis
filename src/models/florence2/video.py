import time
import cv2
import numpy as np
from collections import deque
from typing import List, Any
import logging
import torch

import state
import utils
import config
from ultralytics.trackers.byte_tracker import STrack
from ultralytics.engine.results import Boxes

logger = logging.getLogger(__name__)
_label2id = {}

def _xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = xyxy[:,0], xyxy[:,1], xyxy[:,2], xyxy[:,3]
    w = np.clip(x2 - x1, a_min=1.0, a_max=None)
    h = np.clip(y2 - y1, a_min=1.0, a_max=None)
    cx = x1 + 0.5*w
    cy = y1 + 0.5*h
    return np.stack([cx, cy, w, h], axis=1).astype(np.float32)

def _build_tracker_input(dets: List[dict], frame) -> Boxes:
    """
    Convert detections into Ultralytics Boxes for BYTETracker.
    """
    if not dets:
        return Boxes(torch.zeros((0, 6), dtype=torch.float32), frame)

    xyxy = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32)
    conf = torch.tensor([float(d.get("score", 0.0)) for d in dets], dtype=torch.float32).unsqueeze(1)
    cls = torch.tensor([
        _label2id.setdefault(str(d.get("label", "object")), len(_label2id))
        for d in dets
    ], dtype=torch.float32).unsqueeze(1)

    data = torch.cat([xyxy, conf, cls], dim=1)  # [N,6]
    return Boxes(data, frame)

def video_loop(cap, tracker):
    fps_times = deque(maxlen=30)
    frame_idx = 0
    logger.info("Video loop started")

    while not state.stop_event.is_set():
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_idx += 1
        fps_times.append(time.time())
        fps = 0.0 if len(fps_times)<2 else (len(fps_times)-1)/(fps_times[-1]-fps_times[0])

        with state.frame_lock:
            state.latest_frame = frame.copy()
        overlay = frame.copy()

        with state.detections_lock:
            dets, ts = state.latest_detections, state.detections_ts
        use_dets = dets is not None and (time.time()-ts)<=config.VALID_DETECTION_AGE

        tracks = []
        if use_dets:
            try:
                inp = _build_tracker_input(dets, overlay)
                raw = tracker.update(inp, overlay)
                tracks = [t for t in raw if isinstance(t, STrack)]
            except Exception as e:
                logger.error(f"Tracker update failed: {e}")

        draw_tracks(overlay, tracks)
        draw_overlay(overlay, fps)

        cv2.imshow("Webcam Feed", overlay)
        if cv2.waitKey(1)&0xFF==ord('q'):
            state.stop_event.set()
            break

        if frame_idx%300==0:
            elapsed = (time.time()-start)*1000
            logger.info(f"Frame {frame_idx}, FPS={fps:.1f}, Avg frame time={elapsed:.1f}ms")

def draw_tracks(frame, tracks: List[Any]):
    colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]
    for t in tracks:
        try:
            x,y,w,h = t.tlwh
            tid = t.track_id
            score = getattr(t,'score',0.0)
            x1,y1,x2,y2 = map(int,(x,y,x+w,y+h))
            col=colors[tid%len(colors)]
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            label=f"ID:{tid} {score:.2f}"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.6,1)
            cv2.rectangle(frame,(x1,max(0,y1-th-6)),(x1+tw+6,y1),col,-1)
            cv2.putText(frame,label,(x1+3,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1)
        except Exception as e:
            logger.error(f"Error drawing track: {e}")

def draw_overlay(frame, fps: float):
    h,w=frame.shape[:2]
    if state.latest_caption:
        lines = utils.wrap_text(state.latest_caption,50)
        for i,line in enumerate(lines):
            (tw,th),_ = cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            cv2.rectangle(frame,(8,25+30*i),(8+tw+8,50+30*i),(0,0,0),-1)
            cv2.putText(frame,line,(10,45+30*i),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    status=[f"FPS:{fps:.1f}"]
    if state.processing_event.is_set(): status.append("Processing...")
    with state.detections_lock:
        if state.latest_detections: status.append(f"Objects:{len(state.latest_detections)}")
    st=" | ".join(status)
    (tw,th),_ = cv2.getTextSize(st,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
    cv2.rectangle(frame,(8,h-th-16),(8+tw+8,h-8),(0,0,0),-1)
    cv2.putText(frame,st,(10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
