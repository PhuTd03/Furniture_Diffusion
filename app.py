import numpy as np
import gradio as gr
from PIL.ImageOps import colorize, scale
from PIL import Image
import os
import matplotlib.pyplot as plt
import json
import sys
import cv2

from Segment import Segment


def clean():
    return ([[],[]]), None, None, ""

def get_meta_from_img_seq(img):
    if img is not None:
        return ([[],[]]), None, None, ""

    print("get meta information from img seq")
    origin_img = cv2.imread(img)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

    return origin_img, origin_img, ""

# def init_Segment(sam_gap, ):










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
        segment_img = gr.State(None)
        prompt_text = gr.State("")

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
                    with gr.Row():
                        seg_every_button = gr.Button(label="Segmentation", interactive=True, value="segment everything")
                        with gr.Column():    
                            point_mode = gr.Radio(
                                choices=["Positive"],
                                label="Point Prompt",
                                value="Positive",
                                interactive=True
                            )

                            every_undo_button = gr.Button(label="Undo", interactive=True, value="undo")

                tab_click = gr.Tab(label="Click")
                with tab_click:
                    with gr.Row():
                        click_mode = gr.Radio(
                            choices=["Positive", "Negative"],
                            label="Click Prompt",
                            value="Positive",
                            interactive=True
                        )
                        click_undo_button = gr.Button(label="Undo", interactive=True, value="undo")
                
                tab_text = gr.Tab(label="Text")
                with tab_text:
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
                            undo_text_button = gr.Button(label="Undo", interactive=True, value="undo")

                with gr.Row():
                    with gr.Accordion("Advance option", open=False):
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

            segment_img = gr.Image(type="pil", label="Segmented Image")


    #################
    ### Back-end ###
    #################
    
        # import img and get the image to the right position
        input_img.change(
            fn=get_meta_from_img_seq,
            inputs=[input_img],
            outputs=[segment_img, origin_img, prompt_text]
        )

        # clean the state
        tab_everything.select(
            fn=clean,
            inputs=[],
            outputs=[click_state, origin_img, segment_img, prompt_text]
        )

        tab_click.select(
            fn=clean,
            inputs=[],
            outputs=[click_state, origin_img, segment_img, prompt_text]
        )

        tab_text.select(
            fn=clean,
            inputs=[],
            outputs=[click_state, origin_img, segment_img, prompt_text]
        )

        ### Undo button ###
        every_undo_button.click(
            fn=clean,
            inputs=[],
            outputs=[click_state, origin_img, segment_img, prompt_text]
        )

        # click_undo_button.click(
        #     fn=clean,
        #     inputs=[],
        #     outputs=[click_state, origin_img, segment_img, prompt_text]
        # )

        undo_text_button.click(
            fn=clean,
            inputs=[],
            outputs=[click_state, origin_img, segment_img, prompt_text]
        )

        ###init the segment###


    


                        

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    app()



