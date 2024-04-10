import numpy as np

import gradio as gr
from PIL.ImageOps import colorize, scale
from PIL import Image

import os
import matplotlib.pyplot as plt
import json
import sys

def clean():
    return ([[],[]]), None, None, ""

def get_meta_from_img_seq(img):
    if img is not None:
        return ([[],[]]), None, None, ""

    print("get meta information from img seq")
    origin_img = cv2.imread(img)
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

    return origin_img, origin_img, ""



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



        with gr.Row():
            with gr.Column():
                with gr.Row():
                    #input image
                    input_img = gr.Image(type="filepath", label="Input Image")

                    segment_img = gr.Image(type="pil", label="Segmented Image")

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
                        prompt_text = gr.Textbox(label="prompt", placeholder="Name of the furniture")
                        with gr.Accordion("Advance option", open=False):
                            with gr.Row():
                                with gr.Column(scale=0.5):
                                    box_threshold = gr.Slider(minimum=0, maximum=1, step=0.001, value=0.5, label="Box Threshold")
                                with gr.Column(scale=0.5):
                                    text_threshold = gr.Slider(minimum=0, maximum=1, step=0.001, value=0.5, label="Text Threshold")

                        seg_text_button = gr.Button(label="Segmentation", interactive=True, value="segmentation")


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

    


                        

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    app()



