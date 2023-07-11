import os
import torch
from utils.create_graphics import plot_scores
from aesthetic_predictor.gradient_ascent import Gradient_Ascent
from aesthetic_predictor.simple_inference import AestheticPredictor

prompt1 = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
         'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'


prompt2 = "ugly meme, funniest thing ever"
prompt3 = "a dad angry at missing his flight from prague to nyc, the dad is drunk "


def visualize_aesthetic_scores(folder_path, prompt):
    # Get all file names in the folder
    file_names = os.listdir(folder_path)
    tuple_list = [(s.split("_")[0], s.split("_")[-1].replace('.jpg', '')) for s in file_names]
    tuple_list = [(int(x), float(y)) for x, y in tuple_list]
    tuple_list.sort(key=lambda x: x[0])
    scores_list = [y for x, y in tuple_list]
    plot_scores(scores_list, save_dir=f'./output/Adam_{prompt[0:20]}_2.png')


def get_aesthetic_scores(prompt, num_iterations = 150):
    scores = list()
    ga = Gradient_Ascent()
    aesthetic_predictor = AestheticPredictor()

    embedding = aesthetic_predictor.get_features(prompt, image_input=False, text_input=True, normalize=False)
    normalized_embedding = aesthetic_predictor.get_features(torch.clone(embedding), text_input=False, image_input=False)
    scores.append(aesthetic_predictor.mlp(normalized_embedding).squeeze().item())
    for i in range(num_iterations):
        embedding = ga.get_perturbated_features(torch.clone(embedding), text_input=False, image_input=False)
        scores.append(aesthetic_predictor.mlp(embedding).squeeze().item())
    return scores


if __name__ == "__main__":
    #folder_path = './output/adamOnLion/prompt1_2'
    #visualize_aesthetic_scores(folder_path, prompt1)
    scores = get_aesthetic_scores(prompt3)
    plot_scores(scores, save_dir=f'./output/{prompt1[0:20]}_mlp_only.png')


