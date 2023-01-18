import matplotlib.pyplot as plt
import random
import os
import csv
import pandas as pd
import requests


def create_boxplot(values, prompt_aesthetics=None, filename=None):
    # Create the figure and boxplot
    fig, ax = plt.subplots()
    bp = ax.boxplot(values)

    # Add the mean as an empty dot to the plot
    if prompt_aesthetics is not None:
        ax.scatter([1], [prompt_aesthetics], marker='o', color='blue', s=50)

    # Save the figure to disk if a filename is provided
    if filename is not None:
        print(f'save {filename}')
        plt.savefig(filename)

    return fig


def get_random_seeds(num_seeds):
    seeds = list()
    while len(seeds) < num_seeds:
        seed = random.randint(1000, 1000000)
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def write_to_csv(csv_rows, filename, file_path, seed=None):
    if not seed is None:
        file_path = f'{file_path}{seed}/' + filename
    else:
        file_path = f'{file_path}' + filename
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        for row in csv_rows:
            writer.writerow(row)
            csvfile.flush()


def retrieve_prompts():
    url = 'https://lexica.art/api/v1/search?q=""'

    s = requests.Session()
    r = s.get(url)

    df = pd.json_normalize(r.json()['images'])

    return df['prompt']


def make_dir(path, seed = None):
    if not os.path.exists(path):
        os.mkdir(path)
    if not seed is None:
        if not os.path.exists(f'{path}/{seed}'):
            os.mkdir(f'{path}/{seed}')


def create_random_prompts(num_prompts, numeric = False):
    # Create a string of all possible characters
    all_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if numeric:
        all_characters = "0123456789"

    prompts = []
    for _ in range(num_prompts):
        # Generate a random prompt length using the randint() function
        prompt_length = random.randint(5, 51)
        prompt_length = 1
        prompt = ""
        for _ in range(prompt_length):
            # Generate a random word length for the prompt using the randint() function
            word_length = random.randint(3, 31)
            # Use the choices() function to randomly select `word_length` characters from `all_characters`
            characters = random.choices(all_characters, k=word_length)
            # Concatenate the selected characters to create a word
            word = "".join(characters)
            # Add the word to the prompt, separated by a space
            prompt += word + " "
        # Remove the extra space at the end of the prompt
        prompt = prompt[:-1]
        prompts.append(prompt)
    return prompts