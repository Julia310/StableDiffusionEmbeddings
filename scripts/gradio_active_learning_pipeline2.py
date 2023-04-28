import gradio as gr
import torch
from ldm.stable_diffusion import StableDiffusion
from utils.image_generation import create_random_prompts, create_prompts
import numpy as np

ldm = StableDiffusion(device='cuda')
uncondition = None
target_condition = None
current_condition = None

current_gd_image = None
previous_gd_image = None

global_prompt = None

image_list = list()
condition_list = list()


def init_pipeline_params(prompt, seed):
    global uncondition, current_gd_image, current_condition, condition_list
    global image_list, global_prompt

    global_prompt = prompt

    current_condition = ldm.text_enc([prompt])
    uncondition = ldm.text_enc([""], current_condition.shape[1])

    condition_list = [
        ldm.text_enc([prompt + ' unreal engine']),
        ldm.text_enc([prompt + ' digital painting']),
        ldm.text_enc([prompt + ' pop-art']),
        ldm.text_enc([prompt + ' cartoon']),
        ldm.text_enc([prompt + ' art nouveau'])
    ]

    image_list = list()
    for i in range(len(condition_list)):
        embedding = torch.cat([uncondition, condition_list[i]])
        image_list.append(ldm.embedding_2_img('', embedding, seed=seed, keep_init_latents=False, return_pil=True,
                                     save_img=False))

    current_gd_image = ldm.embedding_2_img(
        '',
        torch.cat([uncondition, current_condition]),
        seed=seed,
        return_pil=True,
        save_img=False
    )

    return image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], current_gd_image


def compute_dot(cond_A, cond_B):
    #low_norm = torch.mean(cond_A, dim=1, keepdim=True)
    #high_norm = torch.mean(cond_B, dim=1, keepdim=True)
    low_norm = cond_A[:,-1]
    low_norm = low_norm / torch.norm(low_norm, dim=-1, keepdim=True)
    high_norm = cond_B[:,-1]
    high_norm = high_norm / torch.norm(high_norm, dim=-1, keepdim=True)

    dot = (low_norm * high_norm).sum()
    return dot


def update_image_embedding(choice, selection_effect):
    global uncondition, condition_list, target_condition, current_condition
    global image_list, current_gd_image, previous_gd_image

    idx = int(choice.split('Img')[1]) - 1
    target_condition = torch.clone(condition_list[idx])
    previously_chosen = image_list[idx].copy()

    current_condition = ldm.lerp(current_condition, target_condition, selection_effect)

    N = len(condition_list) * 100
    p = len(condition_list)


    prompt_list1 = create_random_prompts(N, random_prompt_len=True)
    prompt_list2 = create_prompts(N, prompt_len=3)
    prompt_list3 = create_prompts(len(condition_list), prompt_len=3)
    condition_list = []


    temp_condition_list = list()
    d = np.empty((N, N))

    #https://stackoverflow.com/questions/48925086/choosing-subset-of-farthest-points-in-given-set-of-points/60955896#60955896
    with torch.no_grad():

        for i in range(N):
            temp_condition_list.append(ldm.text_enc([prompt_list1[i] + prompt_list2[i]]))
        for i in range(N):
            for j in range(i, N):
                d[i,j] = d[j, i] = 1 - compute_dot(temp_condition_list[i], temp_condition_list[j])
    #d = (d + d.T) / 2  # Make the matrix symmetric

    print("Finding initial edge...")
    maxdist = 0
    bestpair = ()
    for i in range(N):
        for j in range(i + 1, N):
            if d[i, j] > maxdist:
                maxdist = d[i, j]
                bestpair = (i, j)

    P = set()
    P.add(bestpair[0])
    P.add(bestpair[1])

    print("Finding optimal set...")
    while len(P) < p:
        print("P size = {0}".format(len(P)))
        maxdist = 0
        vbest = None
        for v in range(N):
            if v in P:
                continue
            for vprime in P:
                if d[v, vprime] > maxdist:
                    maxdist = d[v, vprime]
                    vbest = v
        P.add(vbest)

    print(d[list(P)][:,list(P)])

    prompt_list1 = [prompt_list1[i] for i in P]
    prompt_list2 = [prompt_list2[i] for i in P]


    for i in range(len(prompt_list1)):
        # a*((1-faktor)*a+faktor*b) = const
        const = 0.85
        cond_A = current_condition
        cond_B = ldm.text_enc([prompt_list1[i] + prompt_list2[i]])

        val = (1-const)/(1 - compute_dot(cond_A, cond_B))
        print(f'val: {val}')

        print(f'shape: {compute_dot(cond_A, cond_B).shape}')
        print(f'cond_A * cond_A: {compute_dot(cond_A, cond_A)}')
        cond = ldm.lerp(cond_A, cond_B, val)

        cond_A = cond
        cond_B = ldm.text_enc([global_prompt + prompt_list3[i]])
        val = (1 - const) / (1 - compute_dot(cond_A, cond_B))
        print(val)
        condition_list.append(
            ldm.lerp(cond_A, cond_B, val)
        )
        embedding = torch.cat([uncondition, condition_list[i]])
        image_list[i] = ldm.embedding_2_img('', embedding, return_pil=True, save_img=False)

    previous_gd_image = current_gd_image.copy()
    current_gd_image = ldm.embedding_2_img(
        '',
        torch.cat([uncondition, current_condition]),
        return_pil=True,
        save_img=False
    )

    return image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], \
        previously_chosen, previous_gd_image, current_gd_image


with gr.Blocks() as demo:
    with gr.Row():
        gr_image1 = gr.Image(label="Image1", interactive=True)
        gr_image2 = gr.Image(label="Image2", interactive=True)
        gr_image3 = gr.Image(label="Image3", interactive=True)
        gr_image4 = gr.Image(label="Image4", interactive=True)
        gr_image5 = gr.Image(label="Image5", interactive=True)
    with gr.Row():
        with gr.Column():
            with gr.Row():
                seed = gr.Number(label="Seed")
                prompt = gr.Textbox(label="Prompt")
        with gr.Column():
            with gr.Row():
                choice = gr.Radio(["Img1", "Img2", "Img3", "Img4", "Img5"], label="Select an Image")
                selection_effect = gr.Slider(label="Interpolation Value", minimum=0.0, maximum=1.0)
    with gr.Row():
        btn_init = gr.Button("Initialize Pipeline")
        btn_select = gr.Button("Select")

    with gr.Row():
        previous_image = gr.Image(label="Previous Image", interactive=True)
        previous_choice = gr.Image(label="Previous Selection", interactive=True)
        image = gr.Image(label="Current Condition", interactive=True)

    btn_init.click(
        init_pipeline_params,
        inputs=[prompt, seed],
        outputs=[gr_image1, gr_image2, gr_image3, gr_image4, gr_image5, image]
    )

    btn_select.click(
        update_image_embedding,
        inputs=[choice, selection_effect],
        outputs=[gr_image1, gr_image2, gr_image3, gr_image4, gr_image5, previous_choice, previous_image, image]
    )

demo.launch()
