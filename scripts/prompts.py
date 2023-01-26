import random
from stable_diffusion import StableDiffusion
from aesthetic_predictor.simple_inference import AestheticPredictor
import os
from utils import create_random_prompts, create_boxplot, retrieve_prompts, get_random_seeds




def test_prompts():
    aesthetic_predictor = AestheticPredictor()
    rand_prompts = create_random_prompts(50)
    prompts = list()
    for prompt in rand_prompts:
        try:
            retrieved = retrieve_prompts(prompt).tolist()
            prompts = prompts + retrieved
        except Exception as e:
            print(str(e))
    prompts = list(set(prompts))
    print(f'prompt list length: {len(prompts)}')

    aesthetic_pred_list = list()
    for i in range(len(prompts)):
        print(prompts[i])
        try:
            aesthetic_pred_list.append(aesthetic_predictor.text_predict(prompts[i]))
        except Exception as e:
            print(str(e))
    create_boxplot(aesthetic_pred_list, filename="prompts_boxplot.png")


def main():
    test_prompts()


if __name__ == "__main__":
    main()
