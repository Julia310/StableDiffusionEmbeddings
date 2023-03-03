from ldm.stable_diffusion import StableDiffusion
import torch
from aesthetic_predictor.simple_inference import AestheticPredictor
from torchvision.transforms import CenterCrop, Resize, Normalize, InterpolationMode
from optimizer.adam_on_lion import AdamOnLion

seed = 61582
dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)
aesthetic_predictor = AestheticPredictor(device=device)

prompt1 = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
         'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'


prompt2 = "ugly meme, funniest thing ever"
prompt3 = "a dad angry at missing his flight from prague to nyc, the dad is drunk "

prompt4 = "Cute small humanoid bat sitting in a movie theater eating popcorn watching a movie ,unreal engine, cozy " \
         "indoor lighting, artstation, detailed, digital painting,cinematic,character design by mark ryden and pixar " \
         "and hayao miyazaki, unreal 5, daz, hyperrealistic, octane render"

#prompts = [prompt1, prompt2, prompt3]


def preprocess(rgb):
    rgb = Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None)(rgb)
    rgb = CenterCrop(size=(224, 224))(rgb)
    rgb = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(rgb)
    return rgb


class GradientDescent(torch.nn.Module):
    def __init__(self, text_embedding):
        super().__init__()
        self.text_embedding = torch.nn.Parameter(text_embedding)
        self.text_embedding.requires_grad = True
        self.latents = None

    def forward(self, g=7.5, steps=70):
        latents = ldm.embedding_2_img('', self.text_embedding, dim=dim, seed=seed, return_pil=False, g=g, steps=steps)
        self.latents = latents
        image = ldm.latents_to_image(latents, return_pil=False)

        image = preprocess(image)
        image_embedding = aesthetic_predictor.clip.encode_image(image).float()
        image_embedding = aesthetic_predictor.get_features(image_embedding, image_input=False)
        score = aesthetic_predictor.mlp(image_embedding).squeeze()
        print(score)

        return score

    def get_optimizer(self, eta, optim='SGD'):
        if optim == 'SGD':
            return torch.optim.SGD(
                self.parameters(),
                lr=eta,
                momentum=0.95,
                nesterov=True
            )
        elif optim == 'AdamTorch':
            return torch.optim.Adam(
                self.parameters(),
                lr=eta
                #eps=0.00000001
            )
        else:
            return AdamOnLion(
                params=gradient_descent.parameters(),
                lr=eta,
                eps=0.001
            )


if __name__ == '__main__':
    prompt = prompt4
    gradient_descent = GradientDescent(ldm.get_embedding([prompt])[0])

    eta = 0.01
    num_images = 1000

    optimizer = gradient_descent.get_optimizer(eta, 'AdamOnLion')

    for i in range(num_images):
        optimizer.zero_grad()
        output = gradient_descent.forward(steps=70)
        loss = -output
        loss.backward()
        optimizer.step()
        pil_image = ldm.latents_to_image(gradient_descent.latents)[0]
        pil_image.save(f'output/{i}_{prompt[0:45]}_{output.item()}.jpg')
