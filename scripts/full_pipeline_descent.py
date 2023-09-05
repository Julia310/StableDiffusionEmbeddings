from ldm.stable_diffusion import StableDiffusion
import torch
from aesthetic_predictor.simple_inference import AestheticPredictor
from torchvision.transforms import CenterCrop, Resize, Normalize, InterpolationMode
from optimizer.adam_on_lion import AdamOnLion
from torch.nn import functional as F
from time import time
from utils.file_utils import make_dir
import os
seed = 61582
#seed = 9373462
dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)
aesthetic_predictor = AestheticPredictor(device=device)

prompt1 = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
         'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'





def create_next_directory(directory):
    image_directories = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.startswith("image"):
            image_num = item.replace("image", "")
            if image_num.isdigit():
                image_directories.append(int(image_num))

    if image_directories:
        next_number = max(image_directories) + 1
    else:
        next_number = 1

    new_directory_name = "image" + str(next_number)
    new_directory_path = os.path.join(directory, new_directory_name)
    os.mkdir(new_directory_path)

    return new_directory_name

#with open('./scripts/prompts.txt', 'r', encoding='utf-8') as file:
#    prompts = file.readlines()
#    prompts = [line.strip() for line in prompts]  # Remove leading/trailing whitespace and newlines

prompts = ['highly detailed photoreal eldritch biomechanical rock monoliths, stone obelisks, aurora borealis, psychedelic']



def compute_blurriness(image):
    # Convert the image to grayscale
    gray_image = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]

    # Compute the Laplacian filter
    #laplacian_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian_filter = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    laplacian_filter = laplacian_filter.to(gray_image.device).to(gray_image.dtype)

    # Apply the Laplacian filter to the grayscale image
    filtered_image = F.conv2d(gray_image.unsqueeze(0), laplacian_filter.unsqueeze(0).unsqueeze(0))

    # Compute the variance of the Laplacian filter response
    variance = torch.var(filtered_image)

    return variance * 10


def laion_aesthetic(image):
    image = preprocess(image)
    image_embedding = aesthetic_predictor.clip.encode_image(image).float()
    image_embedding = aesthetic_predictor.get_features(image_embedding, image_input=False)
    return aesthetic_predictor.mlp(image_embedding).squeeze()


def get_shifted_embedding(text_embedding, default_std, default_mean):
    shifted_text_embedding = text_embedding / (torch.std(text_embedding)) * default_std
    shifted_text_embedding = shifted_text_embedding - torch.mean(shifted_text_embedding) + default_mean
    return shifted_text_embedding


def preprocess(rgb):
    rgb = Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None)(rgb)
    rgb = CenterCrop(size=(224, 224))(rgb)
    rgb = Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(rgb)
    return rgb


class GradientDescent(torch.nn.Module):
    def __init__(self, condition):
        super().__init__()
        self.condition_row = condition[:, 0, :]
        self.condition = torch.nn.Parameter(condition[:, 1:, :])
        self.uncondition = torch.nn.Parameter(ldm.text_enc([""], condition.shape[1])[:, 1:, :])
        self.condition.requires_grad = True
        self.uncondition.requires_grad = True
        self.latents = None
        self.default_cond_std = torch.std(condition[:, 1:, :])
        self.default_cond_mean = torch.mean(condition[:, 1:, :])
        self.default_uncond_std = torch.std(self.uncondition)
        self.default_uncond_mean = torch.mean(self.uncondition)



    def get_text_embedding(self):
        cond = torch.cat((self.condition_row.unsqueeze(dim=1), self.condition), dim=1)
        uncond = torch.cat((self.condition_row.unsqueeze(dim=1), self.uncondition), dim=1)
        return torch.cat([uncond, cond])


    def forward(self, g=7.5, steps=70):
        latents = ldm.embedding_2_img(self.get_text_embedding(), dim=dim, seed=seed, return_pil=False, g=g, steps=steps)
        self.latents = latents
        image = ldm.latents_to_image(latents, return_pil=False)

        score = compute_blurriness(image)

        print(score)

        return score

    def get_optimizer(self, eta):
        return AdamOnLion(
            params=self.parameters(),
            lr=eta,
            eps=0.0001,
        )


if __name__ == '__main__':

    start = time()

    eta = 0.01
    num_images = 50
    metric = 'Sharpness'
    for prompt in prompts:
        print(prompt)
        gradient_descent = GradientDescent(ldm.text_enc([prompt]))
        os.makedirs(f'./output/metric_optimization/{metric}/{prompt[0:45].strip()}/embeddings', exist_ok=True)
        os.makedirs(f'./output/metric_optimization/{metric}/{prompt[0:45].strip()}/images', exist_ok=True)
        optimizer = gradient_descent.get_optimizer(eta)
        score_list = list()
        max_score = 100000
        max_latents = None

        for i in range(num_images):
            optimizer.zero_grad()
            score = gradient_descent.forward(steps=70)
            score_list.append(score.item())
            torch.save(gradient_descent.get_text_embedding(),
                       f'./output/metric_optimization/{metric}/{prompt[0:45].strip()}/embeddings/{i}_{prompt[0:45].strip()}.pt')

            pil_image = ldm.latents_to_image(gradient_descent.latents)[0]
            pil_image.save(
                f'./output/metric_optimization/{metric}/{prompt[0:45].strip()}/images/{i}_{prompt[0:45].strip()}_{round(score.item(), 4)}.jpg')
            if score.item() < max_score:
                max_score = score.item()
                max_latents = torch.clone(gradient_descent.latents)
            loss = -score

            if i == 0:
                pil_image = ldm.latents_to_image(gradient_descent.latents)[0]
                pil_image.save(
                    f'./output/metric_optimization/{metric}/{prompt[0:45].strip()}/initial_{prompt[0:45].strip()}_{round(max_score, 4)}.jpg')

            loss.backward(retain_graph=True)
            optimizer.step()

            if (i + 1) % 150 == 0:
                gradient_descent.condition = torch.nn.Parameter(
                    get_shifted_embedding(
                        gradient_descent.condition,
                        gradient_descent.default_cond_std,
                        gradient_descent.default_cond_mean
                    )
                )
                gradient_descent.uncondition = torch.nn.Parameter(
                    get_shifted_embedding(
                        gradient_descent.uncondition,
                        gradient_descent.default_uncond_std,
                        gradient_descent.default_uncond_mean
                    )
                )
                gradient_descent.condition.requires_grad = True
                gradient_descent.uncondition.requires_grad = True
                optimizer = gradient_descent.get_optimizer(eta)

        with open(f'./output/metric_optimization/{metric}/{prompt[0:45].strip()}/{round(max_score, 4)}_output.txt', 'w') as file:
            for item in score_list:
                file.write(str(item) + '\n')


        gradient_descent.latents = max_latents
        pil_image = ldm.latents_to_image(gradient_descent.latents)[0]
        pil_image.save(f'./output/metric_optimization/{metric}/{prompt[0:45].strip()}/{prompt[0:45].strip()}_{round(max_score, 4)}.jpg')



    print((time() - start)/60.0)

