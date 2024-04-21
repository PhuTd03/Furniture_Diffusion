# Argument for SegTracker
sam_args = {
    "sam_checkpoint": "ckpt/sam_vit_b_01ec64.pth",
    "model_type": "vit_b",
    "generator_args": {
        "points_per_side": 16,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh":0.9,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 200,
    },
    "gpu_id": None,
}

segment_args = {
    "sam_gap": 10,
    "min_area": 200,
    "max_obj_num": 255,
}