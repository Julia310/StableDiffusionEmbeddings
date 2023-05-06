from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
from utils.image_generation import get_random_seeds

seed = 61582
seed_list = [417016, 683395, 724839]
dim = 512


prompts = ["beautiful mountain landscape, lake, snow, oil painting 8 k hd ",
           "a beautiful and highly detailed matte painting of the epic mountains of avalon, intricate details, epic "
           "scale, insanely complex, 8 k, sharp focus, hyperrealism, very realistic, by caspar friedrich, albert "
           "bierstadt, james gurney, brian froud, "]


seed = 824331

device = 'cuda'

ldm = StableDiffusion(device=device)

def get_shifted_embedding(text_embedding, default_std, default_mean):
    shifted_text_embedding = text_embedding / (torch.std(text_embedding)) * default_std
    shifted_text_embedding = shifted_text_embedding - torch.mean(shifted_text_embedding) + default_mean
    return shifted_text_embedding

if __name__ == '__main__':
    clip_condition = ldm.text_enc([prompts[0]], maxlen=77)
    clip_condition2 = ldm.text_enc([prompts[1]], maxlen=77)
    prompt = 'A squirrel_Quokka in'

    uncond = ldm.text_enc([""], clip_condition.shape[1])

    print(f'0.00. std: {torch.std(clip_condition)} mean: {torch.mean(clip_condition)}')

    for i in range(49):
        alpha = (i + 1) * 0.02
        cond = ldm.combine_embeddings(clip_condition, clip_condition2, alpha)
        cond = cond - torch.mean(cond) + 0.5 * (torch.mean(clip_condition) + torch.mean(torch.mean(clip_condition2)))
        print(f'{alpha}. std: {torch.std(cond)} mean: {torch.mean(cond)}')

        emb = torch.cat([uncond, cond])

        pil_image = ldm.embedding_2_img('', emb, dim=dim, seed=seed, return_pil=True,  save_img=False)

        # pil_image = ldm.latents_to_image(combined_latents)[0]
        pil_image.save(f'output/{i + 1}_{prompt}.jpg')

    print(f'1.00. std: {torch.std(clip_condition2)} mean: {torch.mean(clip_condition2)}')


    #uncond = ldm.text_enc([""], clip_condition.shape[1])
    #condition = ldm.slerp(clip_condition, clip_condition2, 0.9)
    #emb = torch.cat([uncond, condition])
#
    #pil_image = ldm.embedding_2_img('', emb, save_img=False)
    #pil_image.save(f'output/test_{"this was a test"}_.jpg')
#
    # img_path = r"D:\StableDiffusionEmbeddings\output\Cute small squi_Cute small fox\4_Cute small squi_Cute small fox .jpg"
    # #pil = ldm.load_image(img_path)
    # #embedding = ldm.image_to_embedding(pil)
    # #print('')



