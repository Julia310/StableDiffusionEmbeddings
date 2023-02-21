from ldm.stable_diffusion import StableDiffusion
import torch
from tqdm import tqdm
from aesthetic_predictor.simple_inference import AestheticPredictor
from torchvision.transforms import CenterCrop, Resize, Normalize, InterpolationMode


seed=61582
dim=512
ldm = StableDiffusion()
if seed: torch.manual_seed(seed)
initial_latents = torch.randn((1, ldm.unet.in_channels, dim // 8, dim // 8))
initial_latents = initial_latents.to("cuda").half() * ldm.scheduler.init_noise_sigma
aesthetic_predictor = AestheticPredictor()
prompt = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
               'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'


def preprocess(rgb):
    rgb = Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None)(rgb)
    rgb = CenterCrop(size=(224, 224))(rgb)
    rgb = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(rgb)
    return rgb

class GradientDescent(torch.nn.Module):

    def get_image(self, text_embedding):
        pil_image = ldm.embedding_2_img('', text_embedding, save_int=False)

    def forward(self, text_embedding, g=7.5, steps=70):
        latents = torch.clone(initial_latents)


        # Setting number of steps in scheduler
        ldm.scheduler.set_timesteps(steps)

        # Adding noise to the latents

        # Iterating through defined steps
        for i, ts in enumerate(tqdm(ldm.scheduler.timesteps)):
            # We need to scale the i/p latents to match the variance
            inp = ldm.scheduler.scale_model_input(torch.cat([latents] * 2), ts)

            # Predicting noise residual using U-Net
            u, t = ldm.unet(inp, ts, encoder_hidden_states=text_embedding).sample.chunk(2)

            # Performing Guidance
            pred = u + g * (t - u)

            # Conditioning  the latents
            latents = ldm.scheduler.step(pred, ts, latents).prev_sample
        latents = (1 / 0.18215) * latents

        image = ldm.vae.decode(latents).sample
        print(image)

        image = image.clamp(-1, 1)
        #image = (image / 2 + 0.5).clamp(0, 1)
        #image = image.permute(0, 2, 3, 1)
        #image = (image * 255)

        image = preprocess(image)#.unsqueeze(0)
        image_embedding = aesthetic_predictor.clip.encode_image(image).float()
        score = aesthetic_predictor.mlp(image_embedding).squeeze()
        print(score)

        return score


if __name__ == '__main__':
    eta = 0.01
    gradient_descent = GradientDescent()
    with torch.no_grad():
        text_embedding = ldm.get_embedding([prompt])[0]
    #text_embedding = torch.Tensor(ldm.get_embedding([prompt])[0].cpu().detach().numpy()).to("cuda")
    text_embedding.requires_grad = True
    #with torch.no_grad():
    loss = gradient_descent.forward(text_embedding, steps=30)
    #loss = torch.sum(text_embedding)
    with torch.no_grad():
        loss.backward()
        grad = text_embedding.grad.data
        text_embedding += eta * grad # gradient ascent




