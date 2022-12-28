import torch
from torch.distributions.normal import Normal
from stable_diffusion import StableDiffusion
import random
import pandas as pd
import requests
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


class Perturbations:

    def __init__(self):
        self.ldm = StableDiffusion()
        self.prompts = retrieve_prompts()
        self.prompt_pairs = generate_prompt_pairs()
        self.aesthetic_predictor = AestheticPredictor()

    def perturbation(self, prompt1, prompt2):
        emb_list = self.ldm.get_embedding([prompt1, prompt2])

        for i in range(1, 10):
            noise = 1 * i
            emb = self.ldm.combine_embeddings(emb_list[0], emb_list[1], noise)
            self.aesthetic_predictor.predict(emb)

    def perturbate_between_prompts(self):
        for i in range(len(self.prompt_pairs)):
            prompt1_idx = self.prompt_pairs[i][0]
            prompt2_idx = self.prompt_pairs[i][1]
            prompt1 = self.prompts[prompt1_idx]
            prompt2 = self.prompts[prompt2_idx]
            self.perturbation(prompt1, prompt2)
            if i == 3:
                break

    def random_perturbation(self):
        for i in range(len(self.prompts)):
            emb = self.ldm.get_embedding([self.prompts[i]])[0]
            self.aesthetic_predictor.predict(emb)
            for j in range(1, 10):
                noise = sample_noise(emb)
                emb = self.ldm.combine_embeddings(emb, noise, 0.1)
                self.aesthetic_predictor.predict(emb)
            print('')
            if i == 3:
                break
            #self.perturbation(self.prompts[i], )





def main():
    perturbations = Perturbations()
    perturbations.perturbate_between_prompts()
    #perturbations.random_perturbation()



if __name__ == "__main__":
    main()
