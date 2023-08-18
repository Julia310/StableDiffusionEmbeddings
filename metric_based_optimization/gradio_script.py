import gradio as gr
from metric_based_optimization.full_pipeline_descent import get_image

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                seed = gr.Number(label="Seed")
                iterations = gr.Number(label="Iterations")
            with gr.Row():
                metric = gr.Dropdown(
                    choices=["LAION-Aesthetics V2", "Bluriness", "Sharpness", "Saturation increase"], label="Metric"
                )
                optimizer = gr.Dropdown(
                    choices=["SGD", "Adam"], label="Optimizer"
                )
            prompt = gr.Textbox(label="Prompt")

        with gr.Column():
            with gr.Row():
                initial_score = gr.Number(label="Initial Score")
                score = gr.Number(label="Score")
            with gr.Row():
                initial_image = gr.Image(label="Initial Image")
                image = gr.Image(label="Image")
    btn = gr.Button("Generate")
    btn.click(get_image, inputs=[seed, iterations, prompt, optimizer, metric], outputs=[initial_score, score, initial_image, image])

demo.launch()