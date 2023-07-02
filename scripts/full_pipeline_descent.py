from ldm.stable_diffusion import StableDiffusion
import torch
from aesthetic_predictor.simple_inference import AestheticPredictor
from torchvision.transforms import CenterCrop, Resize, Normalize, InterpolationMode
from optimizer.adam_on_lion import AdamOnLion
from torch.nn import functional as F
from time import time

dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)
aesthetic_predictor = AestheticPredictor(device=device)

prompt1 = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
          'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'


def compute_blurriness(image):
    # Convert the image to grayscale
    gray_image = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]

    # Compute the Laplacian filter
    laplacian_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_filter = laplacian_filter.to(gray_image.device).to(gray_image.dtype)

    # Apply the Laplacian filter to the grayscale image
    filtered_image = F.conv2d(gray_image.unsqueeze(0), laplacian_filter.unsqueeze(0).unsqueeze(0))

    # Compute the variance of the Laplacian filter response
    variance = torch.var(filtered_image)

    return variance * 100


def compute_metric(image):
    # Convert the image to grayscale
    gray_image = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]

    # Compute the Laplacian filter
    # laplacian_filter = torch.tensor([[1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9], [1.0/9, 1.0/9, 1.0/9]])
    laplacian_filter = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    laplacian_filter = laplacian_filter.to(gray_image.device).to(gray_image.dtype)

    # Apply the Laplacian filter to the grayscale image
    filtered_image = F.conv2d(gray_image.unsqueeze(0), laplacian_filter.unsqueeze(0).unsqueeze(0))

    # Compute the variance of the Laplacian filter response
    variance = torch.var(filtered_image)

    return variance


def grayscale(image):
    score = - torch.mean(torch.std(image, dim=1)) * 100
    return score


def compute_colorfulness(image):
    # Compute the mean and standard deviation of the RGB channels
    mean = torch.mean(image, dim=(2, 3))
    std = torch.std(image, dim=(2, 3))

    # Compute the colorfulness metric
    colorfulness = torch.sum(std) / (3 * torch.mean(mean))

    return colorfulness


def preprocess(rgb):
    rgb = Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None)(rgb)
    rgb = CenterCrop(size=(224, 224))(rgb)
    rgb = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(rgb)
    return rgb


def laion_aesthetic(image):
    image = preprocess(image)
    image_embedding = aesthetic_predictor.clip.encode_image(image).float()
    image_embedding = aesthetic_predictor.get_features(image_embedding, image_input=False)
    return aesthetic_predictor.mlp(image_embedding).squeeze()


def get_shifted_embedding(text_embedding, default_std, default_mean):
    shifted_text_embedding = text_embedding / (torch.std(text_embedding)) * default_std
    shifted_text_embedding = shifted_text_embedding - torch.mean(shifted_text_embedding) + default_mean
    return shifted_text_embedding


class GradientDescent(torch.nn.Module):
    def __init__(self, condition):
        super().__init__()
        self.condition_row = condition[:, 0, :]
        self.condition = torch.nn.Parameter(condition[:, 1:, :])
        self.condition.requires_grad = True
        self.uncondition = torch.nn.Parameter(ldm.text_enc([""], condition.shape[1])[:, 1:, :])
        self.uncondition.requires_grad = True
        self.default_cond_std = torch.std(condition[:, 1:, :])
        self.default_cond_mean = torch.mean(condition[:, 1:, :])
        self.default_uncond_std = torch.std(self.uncondition)
        self.default_uncond_mean = torch.mean(self.uncondition)

    def get_text_embedding(self):
        cond = torch.cat((self.condition_row.unsqueeze(dim=1), self.condition), dim=1)
        uncond = torch.cat((self.condition_row.unsqueeze(dim=1), self.uncondition), dim=1)
        return torch.cat([uncond, cond])

    def forward(self, metric, seed=61582, g=7.5, steps=70):
        latents = ldm.embedding_2_img(
            '',
            self.get_text_embedding(),
            dim=dim,
            seed=seed,
            keep_init_latents=False,
            return_pil=False,
            g=g,
            steps=steps
        )

        image = ldm.latents_to_image(latents, return_pil=False)

        if metric == "LAION-Aesthetics V2":
            score = laion_aesthetic(image)
        elif metric == "Sharpness":
            score = compute_blurriness(image)
        elif metric == "Bluriness":
            score = -compute_blurriness(image) * 10
        elif metric == "Saturation increase":
            score = compute_colorfulness(image)
        elif metric == "Saturation decrease":
            score = -compute_colorfulness(image) / 100.0

        print(score)

        return score

    def shift_embedding(self, eta = 0.01, optim='AdamOnLion'):
        self.condition = torch.nn.Parameter(
            get_shifted_embedding(
                self.condition,
                self.default_cond_std,
                self.default_cond_mean
            )
        )
        self.condition.requires_grad = True
        self.uncondition = torch.nn.Parameter(
            get_shifted_embedding(
                self.uncondition,
                self.default_uncond_std,
                self.default_uncond_mean
            )
        )
        self.uncondition.requires_grad = True
        return self.get_optimizer(eta, optim)

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
                # eps=0.00000001
            )
        else:
            return AdamOnLion(
                params=self.parameters(),
                lr=eta,
                eps=0.001,
            )


# data/rdm/searchers/openimages

def get_image(seed, iterations, prompt, optimizer, metric):
    max_score = -1000.0
    max_embedding = None

    gradient_descent = GradientDescent(ldm.text_enc([prompt]))
    initial_embedding = torch.clone(gradient_descent.get_text_embedding())
    initial_score = 0
    optimizer = gradient_descent.get_optimizer(0.01, optimizer)
    for i in range(int(iterations)):
        optimizer.zero_grad()
        score = gradient_descent.forward(metric, seed=int(seed), steps=70)
        if initial_score == 0:
            initial_score = round(score.item(), 4)
        if score > max_score:
            max_score = round(score.item(), 4)
            max_embedding = torch.clone(gradient_descent.get_text_embedding())
        loss = -score
        loss.backward(retain_graph=True)
        optimizer.step()

        if metric == 'LAION-Aesthetics V2':
            if (i + 1) % 150:
                optimizer = gradient_descent.shift_embedding()
        elif (i + 1) % 20:
            optimizer = gradient_descent.shift_embedding()

    max_image = ldm.embedding_2_img('', max_embedding, dim=dim, seed=seed, return_pil=True, steps=70, save_img=False)
    initial_image = ldm.embedding_2_img('', initial_embedding, dim=dim, seed=seed, return_pil=True, steps=70,
                                        save_img=False)
    return initial_score, max_score, initial_image, max_image


if __name__ == '__main__':
    initial_score, score, initial_image, image = get_image(61582, 7, "cat", "AdamOnLion", "Sharpness")
