import matplotlib.pyplot as plt
import random
import os
import csv
import pandas as pd
import requests


def create_boxplot(values, filename=None):
    # Create the figure and boxplot
    fig, ax = plt.subplots()
    bp = ax.boxplot(values)

    # Save the figure to disk if a filename is provided
    if filename is not None:
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