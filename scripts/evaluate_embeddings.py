from ldm.stable_diffusion import StableDiffusion
import torch
from aesthetic_predictor.simple_inference import AestheticPredictor
from torchvision.transforms import CenterCrop, Resize, Normalize, InterpolationMode
from optimizer.adam_on_lion import AdamOnLion
from torch.nn import functional as F
from time import time
import os

seeds = [222261, 1332]
prompts = [
    'highly detailed photoreal eldritch biomechanical rock monoliths, stone obelisks, aurora borealis, psychedelic',
    'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
    'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'
]

dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)
aesthetic_predictor = AestheticPredictor(device=device)


def preprocess(rgb):
    rgb = Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None)(rgb)
    rgb = CenterCrop(size=(224, 224))(rgb)
    rgb = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(rgb)
    return rgb

for prompt in prompts:
    for seed in seeds:
        if not os.path.exists(f'output/metric_based/{prompt[0:45].strip()}/image_{seed}'):
            os.mkdir(f'output/metric_based/{prompt[0:45].strip()}/image_{seed}')
        for i in range(300):
            embedding = torch.load(f'./output/metric_based/{prompt[0:45].strip()}/embeddings/{i}_{prompt[0:45].strip()}.pt')

            latents = ldm.embedding_2_img('', embedding, dim=dim, seed=seed, return_pil=False, keep_init_latents=False)
            image = ldm.latents_to_image(latents, return_pil=False)

            image = preprocess(image)
            image_embedding = aesthetic_predictor.clip.encode_image(image).float()
            image_embedding = aesthetic_predictor.get_features(image_embedding, image_input=False)
            score = aesthetic_predictor.mlp(image_embedding).squeeze()
            pil_image = ldm.latents_to_image(latents)[0]
            pil_image.save(
                f'output/metric_based/{prompt[0:45].strip()}/image_{seed}/{i}_{prompt[0:45].strip()}_{round(score.item(), 4)}.jpg')