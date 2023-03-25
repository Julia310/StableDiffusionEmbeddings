from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
from utils.image_generation import get_random_seeds

seed = 61582
seed_list = [417016, 683395, 724839]
dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)



if __name__ == '__main__':
    img_path = r"D:\StableDiffusionEmbeddings\output\Cute small squi_Cute small fox\4_Cute small squi_Cute small fox .jpg"
    pil = ldm.load_image(img_path)
    embedding = ldm.image_to_embedding(pil)
    print('')


