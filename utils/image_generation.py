import random
import requests
import pandas as pd
import torch
import re
from torch import randn
from utils.word_list import word_list


def create_prompts(num_prompts, prompt_len=1):
    prompt_list = list()
    for _ in range(num_prompts):
        prompt = ''
        for i in range(prompt_len):
            prompt += ' ' + word_list[random.randint(0, len(word_list) - 1 )]
        prompt_list.append(prompt)
    return prompt_list


def get_random_seeds(num_seeds):
    seeds = list()
    while len(seeds) < num_seeds:
        seed = random.randint(1000, 1000000)
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def retrieve_prompts(keyword=""):
    url = f'https://lexica.art/api/v1/search?q="{keyword}"'

    s = requests.Session()
    r = s.get(url)

    df = pd.json_normalize(r.json()['images'])

    return df['input']


def count_words(text):
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Split the text into words and count them
    words = text.split()
    return len(words)


def create_random_prompts(num_prompts, numeric=False, random_prompt_len=False):
    # Create a string of all possible characters
    all_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if numeric:
        all_characters = "0123456789"

    prompts = []
    for _ in range(num_prompts):
        torch.manual_seed(random.randint(1, 10000000))

        # Generate a random input length using the randint() function
        prompt_length = 1
        if random_prompt_len:
            prompt_length = random.randint(5, 11)
        prompt = ""
        for _ in range(prompt_length):
            # Generate a random word length for the input using the randint() function
            word_length = random.randint(3, 31)
            # Use the choices() function to randomly select `word_length` characters from `all_characters`
            characters = random.choices(all_characters, k=word_length)
            # Concatenate the selected characters to create a word
            word = "".join(characters)
            # Add the word to the input, separated by a space
            prompt += word + " "
        # Remove the extra space at the end of the input
        prompt = prompt[:-1]
        prompts.append(prompt)
    return prompts


def sample_noise(shape, num=1):
    sample_list = list()
    while len(sample_list) < num:
        sample = randn(size=shape, dtype=torch.float16)
        # sample = normal(mean=mean, std=std, size=shape)
        sample_list.append(sample)

    return sample_list
