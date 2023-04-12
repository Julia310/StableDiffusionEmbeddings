#from scripts.full_pipeline_descent import GradientDescent
from ldm.stable_diffusion import StableDiffusion
from utils.image_generation import get_random_seeds
from PIL import Image
import torch

device = 'cuda'
seed = 61582
#seed = 100
#seed = 300
#seed=37
seed = 816782
#seed = 243635
seed = 824331
steps = 70
dim = 512
seed = 500403
#seed = 151299
seed = 824641
#seed = 151299

seed = 824331

#seed=38
#seed=37
seed=3490347565


ldm = StableDiffusion(device=device)

prompts = ['Cute small squirrel sitting in a movie theater eating popcorn watching a movie ,unreal engine, cozy indoor lighting, artstation, detailed, digital painting,cinematic,character design by mark ryden and pixar and hayao miyazaki, unreal 5, daz, hyperrealistic, octane render',
            'Cute small fox sitting in a movie theater eating popcorn watching a movie ,unreal engine, cozy indoor lighting, artstation, detailed, digital painting,cinematic,character design by mark ryden and pixar and hayao miyazaki, unreal 5, daz, hyperrealistic, octane render']

prompts = ['a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, '
           'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful',
           "epic landscape with a lake, golden hour, misty ground, rocky ground, distant mountains, hazy, foggy, "
           "atmospheric perspective, altostratus clouds,, cinematic, 1000mm lens, anamorphic lens flare, photographic, "
           "octane render, cinematography by roger deakins, in the style of ansel adams, low details,"]

prompts = ['A picture of a cat on a cushion, studio image photography', 'A picture of a dog on a cushion, studio image photography']

prompts = ["beautiful mountain landscape, lake, snow, oil painting 8 k hd ",
           "a beautiful and highly detailed matte painting of the epic mountains of avalon, intricate details, epic scale, insanely complex, 8 k, sharp focus, hyperrealism, very realistic, by caspar friedrich, albert bierstadt, james gurney, brian froud, "]

prompts = ['full body image of a norwegian forest cat of white and ginger fur, by dan mumford, yusuke murata and '
           'makoto shinkai, 8k, cel shaded, unreal engine, featured on artstation, pixiv',
           'fluffy dog']

prompts = ['cat sleeping in a circle with its face hidden inside of a christmas snowglobe', 'dog sleeping in a circle with its face hidden inside of a christmas snowglobe']

prompts = ['majestic fluffy white cat, studio image photography', 'majestic fluffy white dog, studio image photography']


if __name__ == '__main__':
    seeds = get_random_seeds(2)
    embedding1 = ldm.get_embedding([prompts[0]])[0]
    print(f'std: {torch.std(embedding1)}, mean: {torch.mean(embedding1)}')
    embedding2 = ldm.get_embedding([prompts[1]])[0]
    print(f'std: {torch.std(embedding2)}, mean: {torch.mean(embedding2)}')

    prompt = prompts[0][0:15] + '_' + prompts[1][0:15]
    prompt = prompt[0:30]

    pil_image = ldm.embedding_2_img('', embedding1, dim=dim, seed=seed, return_pil=True, steps=steps, save_img=False)
    pil_image.save(f'output/0_{prompt}.jpg')
    pil_image = ldm.embedding_2_img('', embedding2, dim=dim, seed=seed, return_pil=True, steps=steps, save_img=False)
    pil_image.save(f'output/50_{prompt}.jpg')

    for i in range(49):
        alpha = (i+1) * 0.02
        #print(alpha)

        combined_embedding = ldm.combine_embeddings(embedding1, embedding2, alpha)
        pil_image = ldm.embedding_2_img('', combined_embedding, dim=dim, seed=seed, return_pil=True, steps=steps, save_img=False)

        #pil_image = ldm.latents_to_image(combined_latents)[0]
        pil_image.save(f'output/{i + 1}_{prompt}.jpg')