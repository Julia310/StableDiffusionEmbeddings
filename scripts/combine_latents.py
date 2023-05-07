from ldm.stable_diffusion import StableDiffusion
import torch

device = 'cuda'
seed = 61582
seed = 100
seed = 476336
#seed = 615399
#seed2 = 100
#seed = 91687
#seed2 = 476337
#seed2 = 61594
seed2 = 476335
steps = 70

prompt2 = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
         'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'

prompt2 = "ethiopian landscape, highly detailed, digital painting, concept art, sharp focus,  cinematic lighting, diffuse lighting, fantasy, intricate, elegant, lifelike, photorealistic,  illustration,  smooth"

#prompt2 = "treehouse designed by mc escher in lush forest, better homes and garden, surreal"

#prompt2 = 'a painting of a tree with pink flowers, a detailed matte painting by shin yun - bok, cgsociety, photorealism, matte painting, detailed painting, matte drawing '

#prompt2 = "beautiful mountain landscape, lake, snow, oil painting 8 k hd "

#prompt2 = "a cybernetic samoyed and beagle, concept art, detailed face and body, detailled decor, fantasy, highly detailed, cinematic lighting, digital art painting, winter, nature, running"

#prompt2 = "Golden treehouse in lush forest, better homes and hardens magazine, big glass windows, intricate woodworking, polaroid"

prompt2 = 'pixar style, blue cartoon hummingbird mascot with adorable eyes, friendly, waving to the camera, cinematic lighting'

seed = 797248
seed2 = 253808

dim = 512

ldm = StableDiffusion(device=device)

if __name__ == '__main__':
    #img = ldm.load_image(p=image_path)
    #latents_target = ldm.pil_to_latents(img)

    embedding = ldm.get_embedding([prompt2])[0]
    latents = ldm.embedding_2_img('', embedding, dim=dim, seed=seed, return_pil=False, steps=steps, keep_init_latents=False)
    init_latents1 = torch.clone(ldm.initial_latents)
    latents_target = ldm.embedding_2_img('', embedding, dim=dim, seed=seed2, return_pil=False, steps=steps, keep_init_latents=False)
    init_latents2 = torch.clone(ldm.initial_latents)
    pil_image = ldm.latents_to_image(latents)[0]
    pil_image.save(f'output/0_{prompt2[0:45]}.jpg')
    pil_image = ldm.latents_to_image(latents_target)[0]
    pil_image.save(f'output/50_{prompt2[0:45]}.jpg')

    for i in range(49):
        alpha = (i+1) * 0.02
        combined_init_latents = ldm.slerp(init_latents1, init_latents2, alpha)
        ldm.initial_latents = combined_init_latents
        pil_image = ldm.embedding_2_img('', embedding, dim=dim, seed=seed, return_pil=True, steps=steps,
                                        keep_init_latents=True, save_img=False)
        pil_image.save(f'output/{i + 1}_{prompt2[0:45]}.jpg')
