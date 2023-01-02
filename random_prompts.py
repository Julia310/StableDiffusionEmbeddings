import random
from stable_diffusion import StableDiffusion
from aesthetic_predictor.simple_inference import AestheticPredictor
import os
import csv


def create_random_prompts(num_prompts):
    # Create a string of all possible characters
    all_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    prompts = []
    for _ in range(num_prompts):
        # Generate a random prompt length using the randint() function
        prompt_length = random.randint(5, 51)
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


def make_dir(seed = None):
    if not os.path.exists(f'./output/random'):
        os.mkdir(f'./output/random/')
    if not seed is None:
        if not os.path.exists(f'./output/random/{seed}'):
            os.mkdir(f'./output/random/{seed}')


def get_random_seeds(num_seeds):
    seeds = list()
    for i in range(num_seeds):
        seeds.append(random.randint(1000, 1000000))
    return seeds


def write_to_csv(csv_rows):
    file_path = './output/random/random_prompts.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
        for row in csv_rows:
            writer.writerow(row)
            csvfile.flush()


def main():
    aesthetic_predictor = AestheticPredictor()
    prompts = create_random_prompts(100)
    seeds = get_random_seeds(10)
    ldm = StableDiffusion()
    emb_list = ldm.get_embedding(prompts)
    make_dir()
    csv_file = list()
    csv_file.append(['prompt'] + seeds)
    for i in range(len(prompts)):
        csw_row = list()
        csw_row.append(prompts[i])
        for seed in seeds:
            make_dir(seed)
            emb = emb_list[i]
            pil_image = ldm.embedding_2_img(prompts[i], emb, seed=seed, save_int=False)
            pil_image.save(f'./output/random/{seed}/{prompts[i][0:30]}.jpg')

            csw_row.append(aesthetic_predictor.predict(pil_image))
        csv_file.append(csw_row)
        write_to_csv(csv_file)


if __name__ == "__main__":
    main()
