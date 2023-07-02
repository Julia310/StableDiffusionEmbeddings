from ldm.stable_diffusion import StableDiffusion
import torch
from PIL import Image


seed = 417016
target_seed = 510675
target_seed = 683395

ldm = StableDiffusion(device='cuda')

prompt = 'Single Color Ball'
prompt2 = 'Blue Single Color Ball'

initial_condition = ldm.text_enc([prompt])[:, 1:, :]
target_condition = ldm.text_enc([prompt2])[:, 1:, :]
uncondition = ldm.text_enc([""])
condition_row = uncondition[:, 0, :]


def create_combined_image(image_list):
    # Get the dimensions of the input images
    width, height = image_list[0][0].size

    # Calculate the dimensions of the combined image
    combined_width = width * len(image_list[0])
    combined_height = height * len(image_list)

    # Create a new blank image with the combined dimensions
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the images into the combined image
    for i, row in enumerate(image_list):
        for j, image in enumerate(row):
            combined_image.paste(image, (j * width, i * height))

    return combined_image



image_list = list()

with torch.no_grad():
    ldm.embedding_2_img('', ldm.get_embedding([prompt])[0], save_img=False,
                        seed=target_seed, return_pil=False,
                        return_latents=True, keep_init_latents=False)

    target_init_latents = torch.clone(ldm.initial_latents)
    ldm.embedding_2_img('', ldm.get_embedding([prompt])[0], seed=seed, return_latents=True,
                        keep_init_latents=False)
    latents = torch.clone(ldm.initial_latents)

for alpha in range(-8, 9):
    image_list_row = []
    condition = ldm.slerp(target_condition, initial_condition, torch.sigmoid(torch.tensor(alpha)))
    condition = torch.cat((condition_row.unsqueeze(dim=1), condition), dim=1)
    embedding = torch.cat([uncondition, condition])
    for i in range(21):
        beta = i * 0.05
        initial_latents = ldm.slerp(latents, target_init_latents, beta)
        ldm.initial_latents = initial_latents
        pil_image = ldm.embedding_2_img('', embedding,
                                        return_pil=True,
                                        keep_init_latents=True,
                                        save_img=False
                                        )
        image_list_row.append(pil_image)
    image_list.append(image_list_row)
combined_image = create_combined_image(image_list)
combined_image.save(f'{seed}_{target_seed}.png')

