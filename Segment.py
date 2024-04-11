import sys
sys.path.append("..")
sys.path.append("./sam")
from sam.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tool.segmentor import Segmentor

import numpy as np


class Segment():
    def __init__(self, segment_args, sam_args) -> None:
        """
        Initialize SAM.
        """
        self.sam = Segmentor(sam_args)
        self.detector = Detector(self.sam.device)
        self.sam_gap = segment_args['sam_gap']
        self.min_area = segment_args['min_area']
        self.max_obj_num = segment_args['max_obj_num']

        self.origin_merged_mask = None
        self.first_frame_mask = None



