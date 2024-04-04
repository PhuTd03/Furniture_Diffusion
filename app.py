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

        click_stack = gr.State([[],[]])
        origin_frame = gr.State(None)
        Seg_Tracker = gr.State(None)

        current_frame_num = gr.State(None)
        refine_idx = gr.State(None)
        frame_num = gr.State(None)

        aot_model = gr.State(None)
        sam_gap = gr.State(None)
        points_per_side = gr.State(None)
        max_obj_num = gr.State(None)

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



