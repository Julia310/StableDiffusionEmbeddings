from ldm.stable_diffusion import StableDiffusion
from aesthetic_predictor.gradient_ascent import Gradient_Ascent

prompt = "realistic portrait of a beautiful fox in a fairy wood, 8k, ultra realistic, atmosphere, glow, detailed, " \
         "intricate, full of colour, trending on artstation, masterpiece"

seed = 3027230121
def main():
    ldm = StableDiffusion()
    gradient_ascent = Gradient_Ascent()
    emb = ldm.get_embedding([prompt])[0]
    #emb[:,:36] = 0
    #emb[:,36:] = emb[:,36:37]
    #emb = emb[:, -10:]
    #make_dir(f'../output/improve_prompt', seed)
    #perturbed_emb = gradient_ascent.get_gradient(prompt, image_input=False)
    #emb[1] = perturbed_emb
    pil_image = ldm.embedding_2_img(prompt, emb, seed=seed, save_int=False)
    gradient_ascent.get_gradient(pil_image)
    pil_image.save(f'../output/{prompt[0:30]}.jpg')


if __name__ == "__main__":
    main()


