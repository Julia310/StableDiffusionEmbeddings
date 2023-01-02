import torch
from torch.distributions.normal import Normal
from stable_diffusion import StableDiffusion
import random
import pandas as pd
import requests
import csv
import os
from aesthetic_predictor.simple_inference import AestheticPredictor


def sample_noise(embedding):
    std = torch.std(embedding)
    mean = torch.mean(embedding)
    shape = embedding.shape
    sampler = Normal(mean, std)
    sample = sampler.sample(shape)

    return sample


def generate_random_integers():
    # Generate two random integers
    num1 = random.randint(0, 49)
    num2 = random.randint(0, 49)

    # Make sure the two numbers are not equal
    while num1 == num2:
        num2 = random.randint(0, 49)

    return [num1, num2]


def retrieve_prompts():
    url = 'https://lexica.art/api/v1/search?q=""'

    s = requests.Session()
    r = s.get(url)

    df = pd.json_normalize(r.json()['images'])

    return df['prompt']


def generate_prompt_pairs():
    prompt_pairs = list()

    while len(prompt_pairs) < 100:
        pair = generate_random_integers()
        if pair not in prompt_pairs:
            prompt_pairs.append(pair)
    return prompt_pairs


def make_dir(directory, seed = None):
    if not os.path.exists(f'./output/{directory}'):
        os.mkdir(f'./output/{directory}/')
    if not seed is None:
        if not os.path.exists(f'./output/{directory}/{seed}'):
            os.mkdir(f'./output/{directory}/{seed}')


def get_random_seeds(num_seeds):
    seeds = list()
    for i in range(num_seeds):
        seeds.append(random.randint(1000, 1000000))
    return seeds


def write_to_csv(csv_rows, filename, file_path, seed = None):
    if not seed is None:
        file_path = f'{file_path}{seed}/' + filename
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        for row in csv_rows:
            writer.writerow(row)
            csvfile.flush()


class Perturbations:

    def __init__(self):
        self.ldm = StableDiffusion()
        self.prompts = retrieve_prompts()
        """self.prompts = [
            'flat primitive drawing, front view, full face, aqueduct with 4 arches ',
            'Steve Buscemi as Willy Wonka',
            'mandelbrot 3 d volume fractal mandala ceramic chakra digital color stylized an ancient white bone and emerald gemstone relic, intricate engraving concept substance patern texture natural color scheme, global illumination ray tracing hdr fanart arstation by sung choi and eric pfeiffer and gabriel garza and casper konefal',
            'portrait of a splendid sad north korean woman from scary stories to tell in the dark in ibiza, spain with pearlescent skin and cyber yellow hair stronghold in the style of chalks by andre kohn, trending on 5 0 0 px : : 5, hdr, 8 k resolution, ray traced, screen space ambient occlusion : : 2, true blue color scheme',
            'beautiful mannequin sculpted out of amethyst by billelis + lit with 3 d geometric neon + facing a doorway opening with neon pink geometric fractal light + flowering hosta plants!!!, moon + city of los angeles in background!! dramatic, rule of thirds, award winning, 4 k, trending on artstation, photorealistic, volumetric lighting, octane render',
            'dystopian cyberpunk mcdonald dictatorship',
            'an object'
        ]"""
        self.prompt_pairs = generate_prompt_pairs()
        #self.prompt_pairs = [[0,1], [2, 3], [6, 0], [4, 5]]
        self.aesthetic_predictor = AestheticPredictor()

    def perturbation(self, prompt1, prompt2, seed):
        csv_row = [prompt1, prompt2]
        print('========================================')
        print(prompt1)
        print(prompt2)
        for i in range(11):
            noise = round(0.1 * i, 1)
            prompt = f'{prompt1[0:20]}_{prompt2[0:20]}_{noise}'
            emb_list = self.ldm.get_embedding([prompt1, prompt2])
            _, prediction, pil_image = self.aesthetic_prediction(emb_list, noise, prompt, seed)
            pil_image.save(f'./output/perturbations//{prompt}.jpg')
            csv_row.append(prediction)
        return csv_row

    def aesthetic_prediction(self, emb_list, noise, prompt, seed):
        emb = self.ldm.combine_embeddings(emb_list[0], emb_list[1], noise)
        pil_image = self.ldm.embedding_2_img(prompt, emb, seed=seed, save_int=False)
        aesthetic_predictor = self.aesthetic_predictor.predict(pil_image)
        return emb, aesthetic_predictor, pil_image

    def perturbate_between_prompts(self):
        seed = get_random_seeds(1)[0]
        make_dir('perturbations')
        filename = 'perturbations.csv'
        file_path = 'perturbations'
        csv_rows = [['prompt1', 'prompt2'] + list(map(lambda x: str(x), range(11)))]
        for i in range(len(self.prompt_pairs)):
            prompt1_idx = self.prompt_pairs[i][0]
            prompt2_idx = self.prompt_pairs[i][1]
            prompt1 = self.prompts[prompt1_idx]
            prompt2 = self.prompts[prompt2_idx]
            csv_rows.append(self.perturbation(prompt1, prompt2, seed))
        write_to_csv(csv_rows, filename, file_path)

    def random_perturbation(self):
        seeds = get_random_seeds(2)
        for seed in seeds:
            make_dir('random_perturbations', seed)
            csv_file = list()
            csv_file.append(['prompt'] + ["noise" + str(i) for i in range(50)])
            filename = f'{seed}.csv'
            for i in range(len(self.prompts)):
                csv_row = list()
                prompt = self.prompts[i]
                csv_row.append(prompt)
                print('========================================')
                print(prompt)
                emb = self.ldm.get_embedding([prompt])[0]
                pil_image = self.ldm.embedding_2_img('0_' + prompt, emb, seed=seed, save_int=False)
                pil_image.save(f'./output/random_perturbations/{seed}/{prompt[0:30]}_0.jpg')
                prediction = self.aesthetic_predictor.predict(pil_image)
                csv_row.append(prediction)
                for j in range(1, 50):
                    noise = sample_noise(emb)
                    emb, aesthetic_predictor, pil_image = self.aesthetic_prediction([emb, noise], 0.01, str(j) + '_' + prompt)
                    pil_image.save(f'./output/random_perturbations/{seed}/{prompt[0:30]}_{j}.jpg')
                    csv_row.append(aesthetic_predictor)
                csv_file.append(csv_row)
                try:
                    file_path = './output/random_perturbations/' + filename
                    write_to_csv(csv_file, filename, file_path, seed)
                except:
                    continue


def main():
    perturbations = Perturbations()
    perturbations.perturbate_between_prompts()
    #perturbations.random_perturbation()


if __name__ == "__main__":
    main()

    #[i / 10 for i in range(11)]
