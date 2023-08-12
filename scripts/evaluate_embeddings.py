from ldm.stable_diffusion import StableDiffusion
import torch
from aesthetic_predictor.simple_inference import AestheticPredictor
from torchvision.transforms import CenterCrop, Resize, Normalize, InterpolationMode
import os
import re
import numpy as np
import matplotlib.pyplot as plt

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

def embeddings_to_images():
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



def extract_numbers_from_dirs(base_dir: str, window_size=10) -> dict:
    """
    Extract numbers from directories and files in a given directory.

    Args:
    - base_dir (str): Path to the base directory.

    Returns:
    - dict: A dictionary where keys are the integers extracted from
            directory names and values are lists of floats extracted
            from file names inside each directory.
    """
    # Regular expression pattern to match the directory name structure
    dir_pattern = re.compile(r'image_(\d+)')
    line_cnt = 0

    # Create 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 20), sharex=True)

    results = {}

    dirs = os.listdir(base_dir)

    # Iterate over items in the base directory
    for item in dirs:
        full_path = os.path.join(base_dir, item)

        # Check if it's a directory and if its name matches the directory structure
        if os.path.isdir(full_path) and dir_pattern.match(item):
            dir_number = int(dir_pattern.match(item).group(1))

            results[dir_number] = []

            # Iterate over items inside the directory
            files = os.listdir(full_path)
            sorted_files = sorted(files, key=lambda x: int(x.split('_')[0]))
            for sub_item in sorted_files:
                file_number = float(sub_item.split('_')[-1].split('.jpg')[0])
                results[dir_number].append(file_number)

            lines = results[dir_number]
            smoothed_scores = np.convolve(lines, np.ones(window_size) / window_size, mode='valid')

            # Plot on the respective subplot with label as dir_number
            axs[line_cnt % 3].plot(range(window_size - 1, len(lines)), smoothed_scores, label=f'{dir_number}')
            axs[line_cnt % 3].legend()  # Display the legend

            line_cnt += 1

            # Adjust labels
            if line_cnt % 3 == 0:
                axs[(line_cnt // 3) % 3].set_ylabel('Score')
                axs[(line_cnt // 3) % 3].set_xlabel('Iterations')

            # Save and clear the figure after 9 lines
            if line_cnt % 9 == 0 and line_cnt != 0:
                fig.savefig(f'./output/{int(line_cnt / 9)}_plot.png')
                plt.close(fig)  # close the figure

                # Recreate the figure for the next 3 subplots
                fig, axs = plt.subplots(3, 1, figsize=(10, 20), sharex=True)

        # Save remaining subplots
        if line_cnt % 9 != 0:
            fig.savefig(f'./output/metric_based/{int(line_cnt / 9) + 1}_plot.png')
            plt.close(fig)

    return results


base_directory = './output/metric_based/a beautiful painting of a peaceful lake in th'
result = extract_numbers_from_dirs(base_directory)
for dir_number, file_numbers in result.items():
    print(f"Directory {dir_number}: {file_numbers}")

print()
