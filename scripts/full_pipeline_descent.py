from ldm.stable_diffusion import StableDiffusion
import torch
from aesthetic_predictor.simple_inference import AestheticPredictor
from torchvision.transforms import CenterCrop, Resize, Normalize, InterpolationMode


seed=61582
dim=256

ldm = StableDiffusion()
aesthetic_predictor = AestheticPredictor()

prompt = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
               'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'


def preprocess(rgb):
    rgb = Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None)(rgb)
    rgb = CenterCrop(size=(224, 224))(rgb)
    rgb = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(rgb)
    return rgb


class GradientDescent(torch.nn.Module):

    def forward(self, text_embedding, g=7.5, steps=70):
        latents = ldm.embedding_2_img('', text_embedding, dim=dim, seed=seed, return_pil=False, g=g, steps=steps)
        latents = (1 / 0.18215) * latents

        image = ldm.vae.decode(latents).sample
        #print(image)

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
    text_embedding.requires_grad = True
    loss = gradient_descent.forward(text_embedding, steps=50)
    loss.backward()
    grad = text_embedding.grad.data
    print(grad)
    text_embedding = text_embedding + eta * grad # gradient ascent




