import numpy as np
import logging
import argparse
from typing import Optional
from ultralytics.trackers.byte_tracker import BYTETracker

logger = logging.getLogger(__name__)

class ByteTrackArgs(argparse.Namespace):
    """
    Default ByteTrack hyperparameters with improved configuration.
    Subclasses argparse.Namespace to be fully compatible with Ultralytics.
    Includes all required attributes for BYTETracker.
    """
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 50,
        match_thresh: float = 0.8,
        aspect_ratio_thresh: float = 10.0,
        min_box_area: float = 1.0,
        mot20: bool = False,
        fuse_score: float = 0.0,
        new_track_thresh: float = 0.4
    ):
        super().__init__()
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
        self.mot20 = mot20

        # Newly added fields required by Ultralytics
        self.fuse_score = fuse_score
        self.new_track_thresh = new_track_thresh

        # Aliases for Ultralytics BYTETracker expected fields
        self.track_low_thresh = self.track_thresh
        self.track_high_thresh = self.match_thresh

        self._validate_parameters()

    def _validate_parameters(self):
        if not 0.0 <= self.track_thresh <= 1.0:
            raise ValueError(f"track_thresh must be between 0 and 1, got {self.track_thresh}")
        if self.track_buffer < 1:
            raise ValueError(f"track_buffer must be positive, got {self.track_buffer}")
        if not 0.0 <= self.match_thresh <= 1.0:
            raise ValueError(f"match_thresh must be between 0 and 1, got {self.match_thresh}")
        if self.aspect_ratio_thresh <= 0:
            raise ValueError(f"aspect_ratio_thresh must be positive, got {self.aspect_ratio_thresh}")
        if self.min_box_area < 0:
            raise ValueError(f"min_box_area must be non-negative, got {self.min_box_area}")

def init_tracker(frame_rate: int = 30, **kwargs) -> BYTETracker:
    if frame_rate <= 0:
        raise ValueError(f"Frame rate must be positive, got {frame_rate}")
    if frame_rate > 120:
        logger.warning(f"Unusually high frame rate: {frame_rate} FPS")
    elif frame_rate < 10:
        logger.warning(f"Low frame rate may affect tracking performance: {frame_rate} FPS")

    try:
        args = ByteTrackArgs(**kwargs)
        tracker = BYTETracker(args, frame_rate=frame_rate)
        logger.info("ByteTracker initialized successfully:")
        logger.info(f" Frame rate: {frame_rate} FPS")
        logger.info(f" Track threshold: {args.track_thresh}")
        logger.info(f" Match threshold: {args.match_thresh}")
        logger.info(f" Track buffer: {args.track_buffer}")
        logger.info(f" Fuse score: {args.fuse_score}")
        return tracker
    except Exception as e:
        logger.error(f"Failed to initialize BYTETracker: {e}")
        raise

def create_optimized_tracker(use_case: str = "general", frame_rate: int = 30) -> BYTETracker:
    configs = {
        "general": {"track_thresh": 0.5, "track_buffer": 50, "match_thresh": 0.8,
                    "aspect_ratio_thresh": 10.0, "min_box_area": 1.0},
        "fast": {"track_thresh": 0.4, "track_buffer": 30, "match_thresh": 0.7,
                 "aspect_ratio_thresh": 15.0, "min_box_area": 1.0},
        "precise": {"track_thresh": 0.6, "track_buffer": 70, "match_thresh": 0.85,
                    "aspect_ratio_thresh": 8.0, "min_box_area": 4.0},
        "crowded": {"track_thresh": 0.55, "track_buffer": 40, "match_thresh": 0.75,
                    "aspect_ratio_thresh": 12.0, "min_box_area": 2.0},
    }
    if use_case not in configs:
        logger.warning(f"Unknown use case '{use_case}', using 'general'")
        use_case = "general"
    config_kwargs = configs[use_case]
    logger.info(f"Creating optimized tracker for '{use_case}' use case")
    return init_tracker(frame_rate=frame_rate, **config_kwargs)
