import gradio as gr
import gradio.components as comp
from scripts.full_pipeline_descent import get_image


demo = gr.Interface(
    fn=get_image,
    inputs=["number", "number", "text"],
    outputs=["number", comp.Image()],
)


demo.launch()