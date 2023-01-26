import random
from stable_diffusion import StableDiffusion
from aesthetic_predictor.simple_inference import AestheticPredictor
from aesthetic_predictor.gradient_ascent import Gradient_Ascent
import os
from utils import get_random_seeds, write_to_csv, make_dir, create_random_prompts, create_boxplot

prompt = "realistic portrait of a beautiful fox in a fairy wood, 8k, ultra realistic, atmosphere, glow, detailed, " \
         "intricate, full of colour, trending on artstation, masterpiece"

seed = 3027230121
def main():
    seeds = [100]
    ldm = StableDiffusion()
    gradient_ascent = Gradient_Ascent()
    emb = ldm.get_embedding([prompt])[0]
    #emb[:,:36] = 0
    #emb[:,36:] = emb[:,36:37]
    #emb = emb[:, -10:]
    #make_dir(f'../output/improve_prompt', seed)
    perturbed_emb = gradient_ascent.get_gradient(prompt)
    emb[1] = perturbed_emb
    pil_image = ldm.embedding_2_img(prompt, emb, seed=seed, save_int=False)
    pil_image.save(f'../output/{prompt[0:30]}.jpg')


if __name__ == "__main__":
    main()


