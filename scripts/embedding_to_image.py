from ldm.stable_diffusion import StableDiffusion
import torch
import os


ldm = StableDiffusion(device='cuda')
seeds = [9373462, 2229467, 471454, 79942, 90214]


def create_next_directory(directory):
    numerical_directories = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.isdigit():
            numerical_directories.append(int(item))

    if numerical_directories:
        next_number = max(numerical_directories) + 1
    else:
        next_number = 1

    new_directory_name = str(next_number)
    new_directory_path = os.path.join(directory, new_directory_name)
    os.mkdir(new_directory_path)

    return next_number

def cond2img(input_file, output_dir):
    condition = torch.load(input_file)
    uncond = ldm.text_enc([""])
    embedding = torch.cat([uncond, condition])

    img_dir = create_next_directory(output_dir)

    output_dir = os.path.join(output_dir, str(img_dir))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for seed in seeds:
        pil_img = ldm.embedding_2_img('', embedding, seed=seed, return_pil=True, keep_init_latents=False,
                                      save_img=False)

        filename = f'{os.path.basename(input_file).split("_")[0]}_{seed}.jpg'
        pil_img.save(os.path.join(output_dir, filename))

input_file = "/mnt/ceph/storage/data-tmp/current/deckersn/stablediffusionembeddings-deployment-ui-output/3_scary and horror old house/results/cond_binary/17_tensor.pt"

output_dir = '/mnt/ceph/storage/data-tmp/current/wd83zadi/output/UI/results'
cond2img(input_file, output_dir)