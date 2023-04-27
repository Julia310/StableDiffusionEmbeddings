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
    clip_condition = ldm.text_enc(["A squirrel programming in Typescript on the moon"], maxlen=77)
    clip_condition2 = ldm.text_enc(["Quokka in Interstellar movie"], maxlen=77)

    uncond = ldm.text_enc([""], clip_condition.shape[1])
    condition = ldm.slerp(clip_condition, clip_condition2, 0.9)
    emb = torch.cat([uncond, condition])

    pil_image = ldm.embedding_2_img('', emb, save_img=False)
    pil_image.save(f'output/test_{"this was a test"}_.jpg')

    # img_path = r"D:\StableDiffusionEmbeddings\output\Cute small squi_Cute small fox\4_Cute small squi_Cute small fox .jpg"
    # #pil = ldm.load_image(img_path)
    # #embedding = ldm.image_to_embedding(pil)
    # #print('')



