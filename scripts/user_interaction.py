import gradio as gr
import torch
from ldm.stable_diffusion import StableDiffusion
from utils.image_generation import create_random_prompts, create_prompts
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image, ImageDraw
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

ldm = StableDiffusion(device='cuda')
uncondition = None
target_condition = None
current_condition = None

current_image = None
previous_image = None

global_prompt = None
global_seed = None

image_list = [None] * 5
image_history = [[None, None, None]] * 5
image_history_interpolation_valie = [[None, None]] * 5
interpolation_val = [[None, None, None]] * 5
condition_list = [None] * 5


def init_pipeline_params(prompt, seed):
    global uncondition, current_gd_image, current_condition, condition_list
    global image_list, global_prompt, global_seed, current_image

    global_prompt = prompt
    global_seed = seed

    current_condition = ldm.text_enc([prompt])
    uncondition = ldm.text_enc([""], current_condition.shape[1])

    current_image = ldm.embedding_2_img(
        '',
        torch.cat([uncondition, current_condition]),
        seed=seed,
        return_pil=True,
        save_img=False,
        keep_init_latents=False
    )

    get_images_for_selection()

    text = 'Initialization completed. Switch to Image Selection Tab.'

    return image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], \
        current_image, text, get_tsne_image()


def compute_dot(cond_A, cond_B):
    low_norm = cond_A[:,-1]
    low_norm = low_norm / torch.norm(low_norm, dim=-1, keepdim=True)
    high_norm = cond_B[:,-1]
    high_norm = high_norm / torch.norm(high_norm, dim=-1, keepdim=True)

    dot = (low_norm * high_norm).sum()
    return dot


def get_random_conditions():
    N = 5 * 100
    p = 5

    prompt_list1 = create_random_prompts(N, random_prompt_len=True)
    prompt_list2 = create_prompts(N, prompt_len=3)

    temp_condition_list = list()
    d = np.empty((N, N))

    # https://stackoverflow.com/questions/48925086/choosing-subset-of-farthest-points-in-given-set-of-points/60955896#60955896
    with torch.no_grad():

        for i in range(N):
            temp_condition_list.append(ldm.text_enc([prompt_list1[i] + prompt_list2[i]]))
            #print(prompt_list1[i] + prompt_list2[i])
        for i in range(N):
            for j in range(i, N):
                d[i, j] = d[j, i] = 1 - compute_dot(temp_condition_list[i], temp_condition_list[j])
    # d = (d + d.T) / 2  # Make the matrix symmetric

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

    print(d[list(P)][:, list(P)])

    return [temp_condition_list[i] for i in P]


def update_user_prompt(choice, selection_effect):
    global current_condition

    idx = int(choice.split('Img')[1]) - 1
    target_condition = torch.clone(condition_list[idx])

    current_condition = ldm.lerp(current_condition, target_condition, selection_effect)

    return image_list[idx].copy()


def add_to_history(previously_chosen, previous_image, current_image, val):
    global image_history
    for i in range(len(image_history) - 1):
        image_history[len(image_history) - 1 - i] = image_history[len(image_history) - 2 - i]
        interpolation_val[len(image_history) - 1 - i] = interpolation_val[len(image_history) - 2 - i]
    image_history[0] = [previously_chosen, previous_image, current_image]
    interpolation_val[0] = [round(val, 2), round(1 - val, 2)]
    print('')


def get_tsne_image():
    #current_condition_flattened = [current_condition[:, :, -1, :].flatten()]
    current_condition_flattened = [current_condition.flatten()]
    flattened_conditions = [tensor.flatten() for tensor in condition_list] + current_condition_flattened
    concatenated_tensor = torch.stack(flattened_conditions, dim=0)
    numpy_array = concatenated_tensor.detach().cpu().numpy()

    # Adjust the perplexity value
    perplexity = numpy_array.shape[0] - 1

    # Perform t-SNE on the NumPy array
    # https://umap-learn.readthedocs.io/en/latest/
    tsne = TSNE(n_components=2, perplexity=perplexity)
    embedded_array = tsne.fit_transform(numpy_array)

    # Separate the first element from the rest
    first_element = embedded_array[-1]
    other_elements = embedded_array[:-1]

    # Create the scatter plot
    fig, ax = plt.subplots()
    ax.scatter(first_element[0], first_element[1], color='red', label='Current')

    # Add blue numbers to the plot
    for i, point in enumerate(other_elements):
        ax.text(point[0], point[1], str(i + 1), color='blue', ha='center', va='center')

    # Set the plot limits to ensure the numbers are visible
    ax.set_xlim(np.min(embedded_array[:, 0]) - 10, np.max(embedded_array[:, 0]) + 10)
    ax.set_ylim(np.min(embedded_array[:, 1]) - 10, np.max(embedded_array[:, 1]) + 10)

    # Add labels and legend
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend()

    # Render the plot onto a canvas
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    # Convert the canvas to a PIL image
    plot_image = np.array(canvas.renderer.buffer_rgba())
    pil_image = Image.fromarray(plot_image)

    return pil_image


def update_images(choice, selection_effect):
    global image_list, current_image, previous_image

    previously_chosen = update_user_prompt(choice, selection_effect)

    previous_image = current_image.copy()
    current_image = ldm.embedding_2_img(
        '',
        torch.cat([uncondition, current_condition]),
        return_pil=True,
        save_img=False
    )

    image_list = [None] * 5

    get_images_for_selection()

    add_to_history(previously_chosen, previous_image, current_image, selection_effect)
    return image_list[0], image_list[1], image_list[2], image_list[3], image_list[4], current_image, \
        image_history[0][0], image_history[0][1], image_history[0][2], \
        f'<p style="text-align: center;">{interpolation_val[0][0]}</p>', f'<p style="text-align: center;">{interpolation_val[0][1]}</p>', \
        image_history[1][0], image_history[1][1], image_history[1][2], \
        f'<p style="text-align: center;">{interpolation_val[1][0]}</p>', f'<p style="text-align: center;">{interpolation_val[1][1]}</p>', \
        image_history[2][0], image_history[2][1], image_history[2][2], \
        f'<p style="text-align: center;">{interpolation_val[2][0]}</p>', f'<p style="text-align: center;">{interpolation_val[2][1]}</p>', \
        image_history[3][0], image_history[3][1], image_history[3][2], \
        f'<p style="text-align: center;">{interpolation_val[3][0]}</p>', f'<p style="text-align: center;">{interpolation_val[3][1]}</p>', \
        image_history[4][0], image_history[4][1], image_history[4][2], \
        f'<p style="text-align: center;">{interpolation_val[4][0]}</p>', f'<p style="text-align: center;">{interpolation_val[4][1]}</p>', \
        get_tsne_image()



def get_images_for_selection():
    global uncondition, condition_list
    global global_prompt, image_list

    temp_condition_list = get_random_conditions()
    prompt_list = create_prompts(5, prompt_len=3)

    for i in range(5):
        # a*((1-faktor)*a+faktor*b) = const
        const = 0.75
        cond_A = current_condition
        cond_B = temp_condition_list[i]

        val = (1-const)/(1 - compute_dot(cond_A, cond_B))

        cond = ldm.lerp(cond_A, cond_B, val)

        cond_A = cond
        cond_B = ldm.text_enc([global_prompt + prompt_list[i]])
        val = (1 - const) / (1 - compute_dot(cond_A, cond_B))
        print(val)
        condition_list[i] = ldm.lerp(cond_A, cond_B, val)
        image_list[i] = ldm.embedding_2_img('', torch.cat([uncondition, condition_list[i]]), return_pil=True, save_img=False)




with gr.Blocks() as demo:
    with gr.Tab("1. Initialization"):
        with gr.Row():
            seed = gr.Number(label="Seed")
            prompt = gr.Textbox(label="Prompt")
        with gr.Row():
            btn_init = gr.Button("Initialize")
        with gr.Row():
            with gr.Column():
                pass
            with gr.Column():
                text = gr.Textbox(label=" ")
            with gr.Column():
                pass

    with gr.Tab("2. Image Selection"):
        with gr.Row():
            gr_image1 = gr.Image(label="Image1", interactive=True)
            gr_image2 = gr.Image(label="Image2", interactive=True)
            gr_image3 = gr.Image(label="Image3", interactive=True)
            gr_image4 = gr.Image(label="Image4", interactive=True)
            gr_image5 = gr.Image(label="Image5", interactive=True)
        with gr.Row():
            choice = gr.Radio(["Img1", "Img2", "Img3", "Img4", "Img5"], label="Select an Image")
            btn_select = gr.Button("Select")
            selection_effect = gr.Slider(label="Interpolation Value", minimum=0.0, maximum=1.0)
        with gr.Row():
            curr_image = gr.Image(label="Current", interactive=True)
            image_tsne = gr.Image(label="TSNE", interactive=True)


    with gr.Tab("3. History"):
        with gr.Row():
            gr.Markdown(
                """
                # Selected
                """)
            gr.Markdown(
                """
                # Previous 
                """)
            gr.Markdown(
                """
                # Updated
                """)
        with gr.Row():
            previous_image_1 = gr.Image(interactive=True)
            previous_choice_1 = gr.Image(interactive=True)
            image_1 = gr.Image(interactive=True)
        with gr.Row():
            markdown1 = gr.Markdown()
            markdown2 = gr.Markdown()
            markdown3 = gr.Markdown()
        with gr.Row():
            previous_image2 = gr.Image(interactive=True)
            previous_choice2 = gr.Image(interactive=True)
            image2 = gr.Image(interactive=True)
        with gr.Row():
            markdown4 = gr.Markdown()
            markdown5 = gr.Markdown()
            markdown6 = gr.Markdown()
        with gr.Row():
            previous_image3 = gr.Image(interactive=True)
            previous_choice3 = gr.Image(interactive=True)
            image3 = gr.Image(interactive=True)
        with gr.Row():
            markdown7 = gr.Markdown()
            markdown8 = gr.Markdown()
            markdown9 = gr.Markdown()
        with gr.Row():
            previous_image4 = gr.Image(interactive=True)
            previous_choice4 = gr.Image(interactive=True)
            image4 = gr.Image(interactive=True)
        with gr.Row():
            markdown10 = gr.Markdown()
            markdown11 = gr.Markdown()
            markdown12 = gr.Markdown()
        with gr.Row():
            previous_image5 = gr.Image(interactive=True)
            previous_choice5 = gr.Image(interactive=True)
            image5 = gr.Image(interactive=True)
        with gr.Row():
            markdown13 = gr.Markdown()
            markdown14 = gr.Markdown()
            markdown15 = gr.Markdown()


    #with gr.Accordion("Open for More!"):
    #    gr.Markdown("Look at me...")


    btn_init.click(
        init_pipeline_params,
        inputs=[prompt, seed],
        outputs=[gr_image1, gr_image2, gr_image3, gr_image4, gr_image5, curr_image, text, image_tsne]
    )

    btn_select.click(
        update_images,
        inputs=[choice, selection_effect],
        outputs=[gr_image1, gr_image2, gr_image3, gr_image4, gr_image5, curr_image,
                 previous_image_1, previous_choice_1, image_1,
                 markdown1, markdown2,
                 previous_image2, previous_choice2, image2,
                 markdown4, markdown5,
                 previous_image3, previous_choice3, image3,
                 markdown7, markdown8,
                 previous_image4, previous_choice4, image4,
                 markdown10, markdown11,
                 previous_image5, previous_choice5, image5,
                 markdown13, markdown14,
                 image_tsne
                 ]
    )

demo.launch()
