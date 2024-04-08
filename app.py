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
                        seg_every_img = gr.Button(label="Segmentation", interactive=True, value="segment everything")
                            

    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    app()



