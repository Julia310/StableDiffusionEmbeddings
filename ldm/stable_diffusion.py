from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import UNet2DConditionModel, LMSDiscreteScheduler, AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as tfms
from utils import sample_noise
import clip

class StableDiffusion:

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

    def pil_to_latents(self, image):
        '''
        Function to convert image to latents
        '''
        init_image = tfms.ToTensor()(image).unsqueeze(0) * 2.0 - 1.0
        init_image = init_image.to(device="cuda", dtype=torch.float16)
        init_latent_dist = self.vae.encode(init_image).latent_dist.sample() * 0.18215
        return init_latent_dist

    def load_image(self, p):
        '''
        Function to load images from a defined path
        '''
        return Image.open(p).convert('RGB').resize((512, 512))

    def get_embedding(self, prompts):
        embedding_list = list()
        for text in prompts:
            text_encoded = self.text_enc([text])

            uncond = self.text_enc([""], text_encoded.shape[1])
            emb = torch.cat([uncond, text_encoded])
            embedding_list.append(emb)

        return embedding_list

    def random_embedding(self, shape, std, mean, num):
        embedding_list = list()
        for i in range(num):
            text_encoded = sample_noise(std, mean, shape)

            uncond = self.text_enc([""], shape)
            emb = torch.cat([uncond, text_encoded])
            embedding_list.append(emb)

        return embedding_list


    def get_cov(self, X, Y):
        mean_X = torch.mean(X)
        mean_Y = torch.mean(Y)

        X_flat = torch.flatten(X)
        Y_flat = torch.flatten(Y)

        covariance = sum((X_flat - mean_X) * (Y_flat - mean_Y)) / (len(X_flat) - 1)

        return covariance

    def combine_embeddings(self, embedding1, embedding2, noise):
        X = embedding1 * (1 - noise)
        Y = embedding2 * noise

        cov = self.get_cov(X, Y)

        #return X + Y

        return (X + Y) / (
                torch.sqrt(torch.std(X) ** 2 + torch.std(Y) ** 2) + 1e-14 + 2 * cov)

    def embedding_2_img(self, prompt, emb, g=7.5, seed=61582, steps=70, dim=512, save_int=True):
        """
        Diffusion process to convert input to image
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
        pil_image = self.latents_to_pil(latents)[0]
        if save_int:
            if not os.path.exists(f'./steps'):
                os.mkdir(f'./steps')
            pil_image.save(f'output/{prompt[0:45]}.jpeg')
        return pil_image
