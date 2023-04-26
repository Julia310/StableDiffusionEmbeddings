import gradio as gr
import torch
from optimizer.adam_on_lion import AdamOnLion
from ldm.stable_diffusion import StableDiffusion
from utils.image_generation import create_random_prompts, create_prompts

ldm = StableDiffusion(device='cuda')
gd = None
current_gd_image = None
previous_gd_image = None

image_choice1 = None
image_choice2 = None
image_choice3 = None

condition1 = None
condition2 = None
condition3 = None

optimizer = None


class GradientDescent(torch.nn.Module):

    def __init__(self, prompt, optimizer, seed):
        super().__init__()
        condition = ldm.text_enc([prompt])
        self.condition = torch.nn.Parameter(condition)
        self.condition.requires_grad = True
        self.uncondition = ldm.text_enc([""], condition.shape[1])
        self.optimizer = self.get_optimizer(0.01, optimizer)
        self.seed = seed
        self.target_condition = None

    def init_target(self, target_condition):
        self.target_condition = target_condition

    def forward(self):
        distance = torch.dist(
            self.target_condition.flatten(start_dim=1, end_dim=-1).to(torch.float64),
            self.condition.flatten(start_dim=1, end_dim=-1).to(torch.float64)
        )
        return distance * 100

    def get_optimizer(self, eta, optim='SGD'):
        if optim == 'SGD':
            return torch.optim.SGD(
                self.parameters(),
                lr=eta,
                momentum=0.95,
                nesterov=True
            )
        elif optim == 'AdamTorch':
            return torch.optim.Adam(
                self.parameters(),
                lr=eta
                # eps=0.00000001
            )
        else:
            return AdamOnLion(
                params=self.parameters(),
                lr=eta,
                eps=0.0001,
            )


def init_pipeline_params(prompt, gd_optimizer, seed):
    global gd, optimizer, current_gd_image
    global condition1, condition2, condition3
    global image_choice1, image_choice2, image_choice3

    optimizer = gd_optimizer

    gd = GradientDescent(prompt, gd_optimizer, seed)

    condition1 = ldm.text_enc([prompt + ' hyperrealistic'])
    condition2 = ldm.text_enc([prompt + ' Modernist'])
    condition3 = ldm.text_enc([prompt + ' pop-art'])

    embedding1 = torch.cat([gd.uncondition, condition1])
    embedding2 = torch.cat([gd.uncondition, condition2])
    embedding3 = torch.cat([gd.uncondition, condition3])

    image1 = ldm.embedding_2_img('', embedding1, seed=seed, keep_init_latents=False, return_pil=True, save_img=False)
    image2 = ldm.embedding_2_img('', embedding2, seed=seed, keep_init_latents=False,  return_pil=True, save_img=False)
    image3 = ldm.embedding_2_img('', embedding3, seed=seed, keep_init_latents=False,  return_pil=True, save_img=False)

    current_gd_image = ldm.embedding_2_img(
        '',
        torch.cat([gd.uncondition, gd.condition]),
        seed=seed,
        return_pil=True,
        save_img=False
    )

    image_choice1 = image1
    image_choice2 = image2
    image_choice3 = image3

    return image1, image2, image3, current_gd_image


def update_image_embedding(choice, iterations):
    global condition1, condition2, condition3
    global gd, current_gd_image, previous_gd_image
    global image_choice1, image_choice2, image_choice3

    gd.optimizer = gd.get_optimizer(0.001, optimizer)

    match choice:
        case "Image1":
            gd.init_target(condition1)
            previously_chosen = image_choice1.copy()
        case "Image2":
            gd.init_target(condition2)
            previously_chosen = image_choice2.copy()
        case "Image3":
            gd.init_target(condition3)
            previously_chosen = image_choice3.copy()

    condition1 = condition2 = condition3 = None

    for i in range(int(iterations)):
        gd.optimizer.zero_grad()
        score = gd.forward()
        print(score)
        loss = score
        loss.backward(retain_graph=True)
        gd.optimizer.step()

    rand_cond_list = list()
    prompt_list1 = create_random_prompts(3, random_prompt_len=True)
    prompt_list2 = create_prompts(3)
    for i in range(len(prompt_list1)):
        rand_cond_list.append(ldm.text_enc([prompt_list1[i] + prompt_list2[i]]))

    condition1 = torch.clone(gd.condition) * 0.7 + rand_cond_list[0] * 0.3
    condition2 = torch.clone(gd.condition) * 0.7 + rand_cond_list[1] * 0.3
    condition3 = torch.clone(gd.condition) * 0.7 + rand_cond_list[2] * 0.3

    print(torch.dot(gd.condition.flatten(start_dim=1, end_dim=-1), condition1.flatten(start_dim=1, end_dim=-1)))
    print(torch.dot(gd.condition.flatten(start_dim=1, end_dim=-1), condition2.flatten(start_dim=1, end_dim=-1)))
    print(torch.dot(gd.condition.flatten(start_dim=1, end_dim=-1), condition3.flatten(start_dim=1, end_dim=-1)))

    embedding1 = torch.cat([gd.uncondition, condition1])
    embedding2 = torch.cat([gd.uncondition, condition2])
    embedding3 = torch.cat([gd.uncondition, condition3])

    image1 = ldm.embedding_2_img('', embedding1, seed=gd.seed, return_pil=True, save_img=False)
    image2 = ldm.embedding_2_img('', embedding2, seed=gd.seed, return_pil=True, save_img=False)
    image3 = ldm.embedding_2_img('', embedding3, seed=gd.seed, return_pil=True, save_img=False)

    previous_gd_image = current_gd_image.copy()
    current_gd_image = ldm.embedding_2_img(
        '',
        torch.cat([gd.uncondition, gd.condition]),
        seed=gd.seed,
        return_pil=True,
        save_img=False
    )

    image_choice1 = image1
    image_choice2 = image2
    image_choice3 = image3

    return image1, image2, image3, previously_chosen, previous_gd_image, current_gd_image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                seed = gr.Number(label="Seed")
                optimizer = gr.Dropdown(
                    choices=["SGD", "Adam"], label="Optimizer"
                )
            with gr.Row():
                prompt = gr.Textbox(label="Prompt")
            with gr.Row():
                btn_init = gr.Button("Initialize Pipeline")
        with gr.Column():
            with gr.Row():
                gr_image1 = gr.Image(label="Image1", interactive=True)
                gr_image2 = gr.Image(label="Image2", interactive=True)
                gr_image3 = gr.Image(label="Image3", interactive=True)
            with gr.Row():
                choice = gr.Radio(["Image1", "Image2", "Image3"])
                iterations = gr.Number(label="Iterations")
            with gr.Row():
                btn_select = gr.Button("Select")

    with gr.Row():
        previous_choice = gr.Image(label="Previous Selection", interactive=True)
        previous_image = gr.Image(label="Previous Image", interactive=True)
        image = gr.Image(label="Current Condition", interactive=True)

    btn_init.click(
        init_pipeline_params,
        inputs=[prompt, optimizer, seed],
        outputs=[gr_image1, gr_image2, gr_image3, image]
    )

    btn_select.click(
        update_image_embedding,
        inputs=[choice, iterations],
        outputs=[gr_image1, gr_image2, gr_image3, previous_choice, previous_image, image]
    )

demo.launch()
