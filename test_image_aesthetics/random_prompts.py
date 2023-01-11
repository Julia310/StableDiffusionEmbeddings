import random
from stable_diffusion import StableDiffusion
from aesthetic_predictor.simple_inference import AestheticPredictor
import os
from utils import get_random_seeds, write_to_csv, make_dir

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


def main():
    aesthetic_predictor = AestheticPredictor()
    prompts = create_random_prompts(100)
    seeds = get_random_seeds(10)
    ldm = StableDiffusion()
    emb_list = ldm.get_embedding(prompts)
    make_dir(f'../output/random')
    csv_file_images = list()
    csv_file_prompts = list()
    csv_file_images.append(['prompt'] + seeds)
    csv_file_prompts.append(['prompt, aesthetic_score'])
    for i in range(len(prompts)):
        csw_row_images = list()
        csw_row_images.append(prompts[i])
        csw_row_prompts = [prompts[i], aesthetic_predictor.text_predict(prompts[i])]
        emb = emb_list[i]
        for seed in seeds:
            make_dir(f'../output/random', seed)
            pil_image = ldm.embedding_2_img(prompts[i], emb, seed=seed, save_int=False)
            #pil_image.save(f'./output/random/{seed}/{prompts[i][0:30]}.jpg')

            csw_row_images.append(aesthetic_predictor.img_predict(pil_image))
        csv_file_images.append(csw_row_images)
        csv_file_prompts.append(csw_row_prompts)
        write_to_csv(csv_file_images, 'random_prompts_images.csv', '../output/random/')
        write_to_csv(csv_file_prompts, 'random_prompts.csv', '../output/random/')


if __name__ == "__main__":
    main()
