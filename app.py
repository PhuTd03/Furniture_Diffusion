import numpy as np
import gradio as gr
from PIL.ImageOps import colorize, scale
from PIL import Image

import torch
import os
import matplotlib.pyplot as plt
import json
import sys
import cv2
import gc

from utils.Segment import Segment
from utils.seg_anything import draw_mask
from utils.model_args import sam_args, segment_args


def clean():
    return ([[],[]]), None, None, "", None

def get_meta_from_img_seq(img):
    if img is None:
        print("Input image is None.")
        return None, None, None, None, ""

    print("Getting meta information from image sequence.")
    try:
        origin_img = Image.open(img).convert("RGB")
        origin_img = origin_img.resize((512, 512))
    except Exception as e:
        print("Error loading image:", e)
        return None, None, "Error loading image."

    return origin_img, origin_img, origin_img, origin_img, ""


def Segment_add_first_frame(Segment_in, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset first frame's mask
        frame_idx = 0
        Segment_in.restart_tracker()
        Segment_in.add_reference(origin_frame, predicted_mask, frame_idx)
        Segment_in.first_frame_mask = predicted_mask
    
    return Segment_in

def create_placeholder_image():
    # Create a black image
    placeholder = np.zeros((512, 512, 3), dtype=np.uint8)
    # Convert the NumPy array to a PIL Image
    placeholder = Image.fromarray(placeholder)
    return placeholder

def init_Segment_all(sam_gap, points_per_side, max_obj_num, origin_frame):
    if origin_frame is None:
        placeholder = create_placeholder_image()
        return placeholder, origin_frame, [[], []], ""

    # reset sam args
    sam_args["generator_args"]["points_per_side"] = points_per_side
    segment_args["sam_gap"] = sam_gap
    segment_args["max_obj_num"] = max_obj_num
    
    Segment_in = Segment(segment_args, sam_args)

    return Segment_in, origin_frame, origin_frame, origin_frame, [[], []], ""

def init_Segment(sam_gap, points_per_side, max_obj_num, origin_frame):
    if origin_frame is None:
        placeholder = create_placeholder_image()
        return placeholder, origin_frame, [[], []], ""

    # reset sam args
    sam_args["generator_args"]["points_per_side"] = points_per_side
    segment_args["sam_gap"] = sam_gap
    segment_args["max_obj_num"] = max_obj_num
    
    Segment_in = Segment(segment_args, sam_args)

    return Segment_in, origin_frame, [[], []], ""

def undo_click_state_and_refine_seg(Segment_in, origin_frame, click_state, sam_gap, max_obj_num, points_per_side):
    if Segment_in is None:
        return Segment, origin_frame, [[], []]

    print("Undo!")
    if len(click_state[0]) > 0:
        click_state[0] = click_state[0][: -1]
        click_state[1] = click_state[1][: -1]

    if len(click_state[0]) > 0:
        prompt = {
            "points_coord": click_state[0],
            "points_mode": click_state[1],
            "multimask": "True"
        }

        masked_frame = seg_acc_click(Segment_in, prompt, origin_frame)
        return Segment_in, masked_frame, click_state
    else:
        return Segment_in, origin_frame, [[], []]

def segment_everything(Segment_in, origin_frame, sam_gap, points_per_side, max_obj_num):
    if Segment_in is None:
        Segment_in, _, _, _ = init_Segment(sam_gap, points_per_side, max_obj_num, origin_frame)

    print("Segment Everything")
    frame_idx = 0

    with torch.cuda.amp.autocast():
        pred_mask = Segment_in.seg(origin_frame)
        torch.cuda.empty_cache()
        gc.collect()
        Segment_in.add_reference(origin_frame, pred_mask, frame_idx)
        Segment_in.first_frame_mask = pred_mask

    masked_frame = draw_mask(origin_frame.copy(), pred_mask)

    return Segment_in, masked_frame

def get_click_prompt(click_state, point):
    click_state[0].append(point["coords"])
    click_state[1].append(point["mode"])

    prompt = {
        "point_coord": click_state[0],
        "mode": click_state[1],
        "multimask": "True",
    }

    return prompt

def seg_acc_click(Segment_in, prompt, origin_frame):
    # Seg acc to click 
    predicted_mask, masked_frame = Segment_in.seg_acc_click(
                                                    origin_frame=origin_frame,
                                                    coords=np.array(prompt["point_coord"]),
                                                    modes=np.array(prompt["mode"]),
                                                    multimask=prompt["multimask"])

    Segment_in = Segment_add_first_frame(Segment_in, origin_mask, predicted_mask)
    
    return masked_frame


def sam_click(Segment_in, origin_frame, point_mode, click_state: list, sam_gap, max_obj_num, points_per_side, evt:gr.SelectData):
    """
    Args:
        origin_frame: np.ndarray
        click_state: [[coordinate], [point_mode]]
    """
    print("sam_click")
    # import pdb; pdb.set_trace()
    if point_mode == "Positive":
        point = {"coords": [evt.index[0], evt.index[1]], "mode": 1}
    else:
        # TODO: add everything positive points
        point = {"coords": [evt.index[0], evt.index[1]], "mode": 0}

    if Segment_in is None:
        Segment_in, _, _, _ = init_Segment(sam_gap, points_per_side, max_obj_num, origin_frame)

    # get_click_prompt for sam to predict the mask
    click_prompt = get_click_prompt(click_state, point)

    # refine acc to prompt
    masked_frame = seg_acc_click(Segment_in, click_prompt, origin_frame)

    return Segment_in, masked_frame, click_state

def gd_detect(Segment_in, origin_frame, prompt_text, box_threshold, text_threshold, sam_gap, max_obj_num, points_per_side):
    if Segment_in is None:
        Segment_in, _, _, _ = init_Segment(sam_gap, points_per_side, max_obj_num, origin_frame)

    print("Detecting")
    predicted_mask, annotated_frame = Segment_in.seg_and_dec(origin_frame, prompt_text, box_threshold, text_threshold)

    masked_frame = draw_mask(annotated_frame, predicted_mask)

    return Segment_in, masked_frame, origin_frame

def app():

    app = gr.Blocks()

    #################
    ### Front-end ###
    #################

    with app:
        gr.Markdown(
            "Furniture Diffusion"
        )

        click_state = gr.State([[], []])
        origin_img = gr.State(None)
        # segment_img = gr.State(None)
        output_img = gr.State(None)
        segment_img_ev = gr.State(None)
        segment_img_cli = gr.State(None)
        segment_img_te = gr.State(None)
        prompt_text = gr.State("")
        Segment_in = gr.State(None)

        sam_gap = gr.State(None)
        points_per_side = gr.State(None)
        max_obj_num = gr.State(None)

        with gr.Row(scale=0.6):
            with gr.Column():
                #input image
                input_img = gr.Image(type="filepath", label="Input Image")

                # Segmentation Tab
                tab_everything = gr.Tab(label="Everything")
                with tab_everything:
                    with gr.Column():
                        segment_img_ev = gr.Image(label="Segmented Image", interactive=False)
                        with gr.Row():
                            seg_every_button = gr.Button(label="Segmentation", interactive=True, value="segment everything")
                            with gr.Column():    
                                point_mode = gr.Radio(
                                    choices=["Positive"],
                                    label="Point Prompt",
                                    value="Positive",
                                    interactive=True
                                )

                                undo_every_button = gr.Button(label="Undo", interactive=True, value="undo")

                tab_click = gr.Tab(label="Click")
                with tab_click:
                    with gr.Column():
                        segment_img_cli = gr.Image(label="Segmented Image", interactive=True)
                        with gr.Row():
                            click_mode = gr.Radio(
                                choices=["Positive", "Negative"],
                                label="Click Prompt",
                                value="Positive",
                                interactive=True
                            )
                            undo_click_button = gr.Button(label="Undo", interactive=True, value="undo")
                
                tab_text = gr.Tab(label="Text")
                with tab_text:
                    with gr.Column():
                        segment_img_te = gr.Image(label="Segmented Image", interactive=False)
                        with gr.Row():
                            prompt_text = gr.Textbox(label="prompt")
                            with gr.Accordion("Advance option", open=False):
                                with gr.Row():
                                    with gr.Column(scale=0.5):
                                        box_threshold = gr.Slider(minimum=0, maximum=1, step=0.001, value=0.5, label="Box Threshold")
                                    with gr.Column(scale=0.5):
                                        text_threshold = gr.Slider(minimum=0, maximum=1, step=0.001, value=0.5, label="Text Threshold")
                            with gr.Column():
                                seg_text_button = gr.Button(label="Segmentation", interactive=True, value="segmentation")
                                # undo_text_button = gr.Button(label="Undo", interactive=True, value="undo")

                with gr.Row():
                    with gr.Accordion("Advance option", open=True):
                        # args for tracking in video do segment-everthing
                            points_per_side = gr.Slider(
                                label = "points_per_side",
                                minimum= 1,
                                step = 1,
                                maximum=100,
                                value=16,
                                interactive=True
                            )

                            sam_gap = gr.Slider(
                                label='sam_gap',
                                minimum = 1,
                                step=1,
                                maximum = 9999,
                                value=100,
                                interactive=True,
                            )

                            max_obj_num = gr.Slider(
                                label='max_obj_num',
                                minimum = 50,
                                step=1,
                                maximum = 300,
                                value=255,
                                interactive=True
                            )

                    reset_button = gr.Button(label="Reset", interactive=True, value="reset")

            output_img = gr.Image(label="Output Image", interactive=False, show_download_button=True)



    #################
    ### Back-end ###
    #################
    
        # import img and get the image to the right position
        input_img.change(
            fn=get_meta_from_img_seq,
            inputs=[input_img],
            outputs=[segment_img_cli, segment_img_ev, segment_img_te, origin_img, prompt_text]
        )

        # --------Clean the state---------
        reset_button.click(
            fn=init_Segment_all,
            inputs=[sam_gap, points_per_side, max_obj_num, origin_img],
            outputs=[Segment_in, segment_img_ev, segment_img_cli, segment_img_te, click_state, prompt_text],
            queue=False,
            show_progress=False
        )

        # -------Undo button--------------
        undo_every_button.click(
            fn=undo_click_state_and_refine_seg,
            inputs=[Segment_in, origin_img, click_state,
            sam_gap,
            max_obj_num,
            points_per_side],
            outputs=[Segment_in, origin_img, click_state]
        )

        undo_click_button.click(
            fn=undo_click_state_and_refine_seg,
            inputs=[Segment_in, origin_img, click_state,
            sam_gap,
            max_obj_num,
            points_per_side],
            outputs=[Segment_in, origin_img, click_state]
        )

        # undo_text_button.click(
        #     fn=clean,
        #     inputs=[],
        #     outputs=[click_state, origin_img, segment_img, prompt_text]
        # )

        # -------------init the segment----------------
        tab_everything.select(
            fn=init_Segment,
            inputs=[sam_gap, points_per_side, max_obj_num, origin_img],
            outputs=[Segment_in, segment_img_ev, click_state, prompt_text],
            queue=False
        )

        tab_click.select(
            fn=init_Segment,
            inputs=[sam_gap, points_per_side, max_obj_num, origin_img],
            outputs=[Segment_in, segment_img_cli, click_state, prompt_text],
            queue=False
        )

        tab_text.select(
            fn=init_Segment,
            inputs=[sam_gap, points_per_side, max_obj_num, origin_img],
            outputs=[Segment_in, segment_img_te, click_state, prompt_text],
            queue=False
        )
        
        # --------SEGMENTATION---------------
        # Segment everything
        seg_every_button.click(
            fn=segment_everything,
            inputs=[Segment_in, origin_img, sam_gap, points_per_side, max_obj_num],
            outputs=[Segment_in, segment_img_ev]
        )
        # Segment with click to get mask
        segment_img_cli.select(
            fn=sam_click,
            inputs=[Segment_in, origin_img, point_mode, click_state, sam_gap, max_obj_num, points_per_side],
            outputs=[Segment_in, segment_img_cli, click_state]
        )
        # Segment with text prompt
        seg_text_button.click(
            fn=gd_detect,
            inputs=[Segment_in, origin_img, prompt_text, box_threshold, text_threshold, sam_gap, max_obj_num, points_per_side],
            outputs=[Segment_in, segment_img_te, origin_img]
        )         

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    app()