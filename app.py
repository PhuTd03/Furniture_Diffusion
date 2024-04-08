import numpy as np

import gradio as gr
from PIL.ImageOps import colorize, scale
from PIL import Image

import os
import matplotlib.pyplot as plt
import json
import sys


def app():

    app = gr.Blocks()

    #################
    ### Front-end ###
    #################

    with app:
        gr.Markdown(
            "Furniture Diffusion"
        )

        # click_state = gr.State([], [])
        # origin_img = gr.State(None)

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    #input image
                    origin_img = gr.Image(type="filepath", label="Input Image")

                    segment_img = gr.Image(type="pil", label="Segmented Image")

                # Segmentation Tab
                tab_everything = gr.Tab(label="Everything")
                with tab_everything:
                    with gr.Row():
                        seg_every_button = gr.Button(label="Segmentation", interactive=True, value="segment everything")
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

    

                        

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    app()



