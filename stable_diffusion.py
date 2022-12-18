from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler, AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image


class SrtableDiffusion():

    def __init__(self):

        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                              num_train_timesteps=1000)
        self.scheduler.set_timesteps(51)
        self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet",
                                                         torch_dtype=torch.float16).to("cuda")
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",
                                                 torch_dtype=torch.float16).to(
            "cuda")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                                          torch_dtype=torch.float16).to("cuda")

    def text_enc(self, prompts, maxlen=None):
        '''
        A function to take a texual promt and convert it into embeddings
        '''
        if maxlen is None: maxlen = self.tokenizer.model_max_length
        inp = self.tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        return self.text_encoder(inp.input_ids.to("cuda"))[0].half()

    def latents_to_pil(self, latents):
        '''
        Function to convert latents to images
        '''
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def get_embedding(self, prompts):
        embedding_list = list()
        for text in prompts:
            text_encoded = self.text_enc([text])

            uncond = self.text_enc([""], text_encoded.shape[1])
            emb = torch.cat([uncond, text_encoded])
            embedding_list.append(emb)

        return embedding_list

    def combine_embeddings(aelf, embedding1, embedding2, noise):
        return (embedding1 * (1 - noise) + embedding2 * noise) / (
                torch.sqrt(torch.std(embedding1 * (1 - noise)) ** 2 + torch.std(embedding2 * noise) ** 2) + 1e-14)

    def embedding_2_img(self, prompt, emb, g=7.5, seed=37, steps=70, dim=512, save_int=True):
        """
        Diffusion process to convert prompt to image
        """
        if seed: torch.manual_seed(seed)
        latents = torch.randn((1, self.unet.in_channels, dim // 8, dim // 8))

        # Setting number of steps in scheduler
        self.scheduler.set_timesteps(steps)

        # Adding noise to the latents
        latents = latents.to("cuda").half() * self.scheduler.init_noise_sigma

        # Iterating through defined steps
        for i, ts in enumerate(tqdm(self.scheduler.timesteps)):
            # We need to scale the i/p latents to match the variance
            inp = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)

            # Predicting noise residual using U-Net
            with torch.no_grad():
                u, t = self.unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)

            # Performing Guidance
            pred = u + g * (t - u)

            # Conditioning  the latents
            latents = self.scheduler.step(pred, ts, latents).prev_sample

            # Saving image
        if save_int:
            if not os.path.exists(f'./steps'):
                os.mkdir(f'./steps')
            self.latents_to_pil(latents)[0].save(f'output/{prompt[0:45]}.jpeg')
