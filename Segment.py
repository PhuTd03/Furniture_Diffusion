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
        self.reference_objs_list = []
        self.object_idx = 1
        self.curr_idx = 1
        self.origin_merged_mask = None # init by segment-everything or update
        self.first_frame_mask = None

        #debug
        self.everything_points = []
        self.everything_labels = []
        print("Segment has been initialized")


    def seg(self, frame):
        '''
        Arguments:
            frame: numpy array (h,w,3)
        Return:
            origin_merged_mask: numpy array (h,w)
        '''
        frame = frame[:, :, ::-1]
        anns = self.sam.everything_generator.generate(frame)

        # anns is a list recording all predictions in an image
        if len (anns) == 0:
            return
        
        # merge all predictions into one mask (h,w)
        # note that the merged mask may lost some objects due to the overlapping
        self.origin_merged_mask = np.zeros(anns[0]['segmention'].shape, dtype=np.uint8)
        idx = 1
        for ann in anns:    # ann is a dict containing all information of a prediction
            if ann['area'] > self.min_area:
                m = ann['segmentation']
                self.origin_merged_mask[m==1] = idx     # m==1 is the mask of the object
                idx += 1
                self.everything_points.append(ann["point_coords"][0])
                self.everything_labels.append(1) 

        obj_ids = np.unique(self.origin_merged_mask)
        obj_ids = obj_ids[obj_ids!=0]

        self.obj_idx = 1
        for id in obj_ids:
            if np.sum(self.origin_merged_mask==id) < self.min_area:
                self.origin_merged_mask[self.origin_merged_mask==id] = 0
            else:
                self.origin_merged_mask[self.origin_merged_mask==id] = self.obj_idx
                self.obj_idx += 1

        self.first_frame_mask == self.origin_merged_mask
        return self.origin_merged_mask

    def update_origin_merged_mask(self, origin_merged_mask):
        self.origin_merged_mask = origin_merged_mask

    def reset_origin_merged_mask(self, mask, id):
        self.origin_merged_mask = mask
        self.curr_idx = id

    def seg_acc_bbox(self, origin_frame: np.array, bbox: np.array, ):
        ''''
        Use bbox-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            bbox: [[x0, y0], [x1, y1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.get_interactive_mask(origin_frame, bbox)
        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw bbox
        masked_frame = draw_bbox(bbox, masked_frame)

        return refined_merged_mask, masked_frame



    def seg_acc_click(self, origin_frame: np.ndarray, coords: np.ndarray, modes: np.ndarray, multimask=True):
        '''
        Use point-prompt to get mask
        Parameters:
            origin_frame: H, W, C
            coords: nd.array [[x, y]]
            modes: nd.array [[1]]
        Return:
            refined_merged_mask: numpy array (h, w)
            masked_frame: numpy array (h, w, c)
        '''
        # get interactive_mask
        interactive_mask = self.sam.segment_with_click(origin_frame, coords, modes, multimask)

        refined_merged_mask = self.add_mask(interactive_mask)

        # draw mask
        masked_frame = draw_mask(origin_frame.copy(), refined_merged_mask)

        # draw points
        # self.everything_labels = np.array(self.everything_labels).astype(np.int64)
        # self.everything_points = np.array(self.everything_points).astype(np.int64)

        masked_frame = draw_points(coords, modes, masked_frame)

        # draw outline
        masked_frame = draw_outline(interactive_mask, masked_frame)

        return refined_merged_mask, masked_frame


    def add_mask(self, interactive_mask: np.array):
        '''
        Arguments:
            mask: numpy array (h, w)
        Return:
            refined_merged_mask: numpy array (h, w)
        '''
        if self.origin_merged_mask is None:
            self.origin_merged_mask = np.zeros(interactive_mask.shape, dtype=np.uint8)

        refined_merged_mask = self.origin_merged_mask.copy()
        refined_merged_mask[interactive_mask > 0] = self.curr_idx
        return refined_merged_mask
        
    def seg_and_dec():
        pass

if __name__ == "__main__":
    

