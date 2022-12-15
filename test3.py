from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler, AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
from datetime import datetime


scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
scheduler.set_timesteps(51)
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to("cuda")
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to("cuda")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to("cuda")


def text_enc(prompts, maxlen=None):
    '''
    A function to take a texual promt and convert it into embeddings
    '''
    if maxlen is None: maxlen = tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    return text_encoder(inp.input_ids.to("cuda"))[0].half()


def latents_to_pil(latents):
    '''
    Function to convert latents to images
    '''
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def get_embedding(prompts):
    embedding_list = list()
    for text in prompts:
        text_encoded = text_enc([text])

        uncond = text_enc([""], text_encoded.shape[1])
        emb = torch.cat([uncond, text_encoded])
        embedding_list.append(emb)

    return embedding_list

def intermediate_embeddings(embedding1, embedding2, noise):
    latents_a = embedding1
    latents_b = embedding2
    return (latents_a * (1 - noise) + latents_b * noise) / (
            torch.sqrt(torch.std(latents_a * (1 - noise)) ** 2 + torch.std(latents_b * noise) ** 2) + 1e-14)


def prompt_2_img(prompt, emb, g=7.5, seed=37, steps=70, dim=512, save_int=True):
    """
    Diffusion process to convert prompt to image
    """
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    # Setting the seed
    if seed: torch.manual_seed(seed)

    # Initiating random noise
    latents = torch.randn((1, unet.in_channels, dim // 8, dim // 8))

    # Setting number of steps in scheduler
    scheduler.set_timesteps(steps)

    # Adding noise to the latents
    latents = latents.to("cuda").half() * scheduler.init_noise_sigma

    # Iterating through defined steps
    for i, ts in enumerate(tqdm(scheduler.timesteps)):
        # We need to scale the i/p latents to match the variance
        inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)

        # Predicting noise residual using U-Net
        with torch.no_grad():
            u, t = unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)

        # Performing Guidance
        pred = u + g * (t - u)

        # Conditioning  the latents
        latents = scheduler.step(pred, ts, latents).prev_sample

        # Saving intermediate images
    if save_int:
        if not os.path.exists(f'./steps'):
            os.mkdir(f'./steps')
        latents_to_pil(latents)[0].save(f'output/{prompt[0:45]}.jpeg')



noise=0.5
prompts = ['full body image of a norwegian forest cat of white and ginger fur, by dan mumford, yusuke murata and '
           'makoto shinkai, 8k, cel shaded, unreal engine, featured on artstation, pixiv',
           'fluffy dog']

prompts = ['futuristic spiderman in style of wolf ,highly detailed, 4k, HDR, award-winning, artstation, octane render',
           'happy pepe, artstation, marvel, anfas, futuristic, cyberpunk hyper detailed, transformers ']

prompts = ['khajit tabaxi catfolk humanoid with maine coon features with an eye patch on the left eye and black fur '
           'holding two shortswords cloaked in shadow and wearing hooded leather armor toned muscle, dungeons and dragons, '
           'pure white background, fantasy, tarot card style, half body portrait, high detail, hyper realistic ',
           'profile of a tan maine coon bold natural colors masterpiece trending on artstation photograph ']

prompts = ['walter white is sherlock holmes, high quality illustration, trending on artstation, octane render, 4 k, pixar rendering, ',
           'cinematic still of will smith in Blade (2001), XF IQ4, f/1.4, ISO 200, 1/160s, 8K, RAW, dramatic lighting, '
           'heisymmetrical balance, in-frame, highly accurate facial features']

prompts = ['a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, '
           'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful',
           "epic landscape with a lake, golden hour, misty ground, rocky ground, distant mountains, hazy, foggy, "
           "atmospheric perspective, altostratus clouds,, cinematic, 1000mm lens, anamorphic lens flare, photographic, "
           "octane render, cinematography by roger deakins, in the style of ansel adams, low details,"]

prompts = ['a painting of a tree with pink flowers, a detailed matte painting by shin yun - bok, cgsociety, photorealism, '
           'matte painting, detailed painting, matte drawing ',
           "Beautiful hyperrealistic detailed matte painting of a landscape of a landscape of wheat fields on which "
           "stands a large wooden toilet with a heart cut out of lost vibes in the foreground and a dark forest in the "
           "background during autumn, afternoon, winter, by andreas rocha and john howe, and Martin Johnson Heade, "
           "featured on artstation, featured on behance, golden ratio, ultrawide angle, f32, well composed, cohesive"]

prompts = ["beautiful mountain landscape, lake, snow, oil painting 8 k hd ",
           "a beautiful and highly detailed matte painting of the epic mountains of avalon, intricate details, epic scale, "
           "insanely complex, 8 k, sharp focus, hyperrealism, very realistic, by caspar friedrich, albert bierstadt, james gurney, brian froud, "]



# Converting textual prompts to embedding
emb_list = get_embedding(prompts)
emb = intermediate_embeddings(emb_list[0], emb_list[1], noise)
emb_list.append(emb)
prompts.append(f'{prompts[0][0:20]}_{prompts[1][0:20]}_{noise}')

for i in range(len(prompts)):
    prompt_2_img(prompts[i], emb_list[i])