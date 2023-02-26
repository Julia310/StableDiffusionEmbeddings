from utils.file_utils import make_dir
from utils.image_generation import sample_noise, get_random_seeds
from utils.create_graphics import create_boxplot
from ldm.stable_diffusion import StableDiffusion
from aesthetic_predictor.simple_inference import AestheticPredictor


if __name__ == "__main__":
    aesthetic_predictor = AestheticPredictor()
    ldm = StableDiffusion()
    emb = ldm.get_embedding(['embedding'])[0]
    seeds = get_random_seeds(200)
    rand_list = sample_noise(1, 0, emb.shape, num = 5)
    for i in range(len(rand_list)):
        predictions = list()
        make_dir('./output/rand_emb/', i)
        rand_emb = rand_list[i]
        rand_emb = rand_emb.to(device='cuda')
        for seed in seeds:
            pil_image = ldm.embedding_2_img('', rand_emb, seed=seed, save_img=False)
            pil_image.save(f'./output/rand_emb/{i}/{seed}_{i}.jpg')

            predictions.append(aesthetic_predictor.predict_aesthetic_score(pil_image))
        create_boxplot(values=predictions, filename=f'./output/rand_emb/{i}_rand_emb_boxplot.png')