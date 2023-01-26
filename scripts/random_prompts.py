from ldm.stable_diffusion import StableDiffusion
from aesthetic_predictor.simple_inference import AestheticPredictor
from utils import get_random_seeds, write_to_csv, make_dir, create_random_prompts, create_boxplot


def random_prompts():
    aesthetic_predictor = AestheticPredictor()
    prompts = create_random_prompts(100)
    seeds = get_random_seeds(10)
    ldm = StableDiffusion()
    emb_list = ldm.get_embedding(prompts)
    make_dir(f'../output/random')
    csv_file_images = list()
    csv_file_prompts = list()
    csv_file_images.append(['input'] + seeds)
    csv_file_prompts.append(['input, aesthetic_score'])
    for i in range(len(prompts)):
        csw_row_images = list()
        csw_row_images.append(prompts[i])
        csw_row_prompts = [prompts[i], aesthetic_predictor.predict_aesthetic_score(prompts[i], image_input=False)]
        emb = emb_list[i]
        for seed in seeds:
            make_dir(f'../output/random', seed)
            pil_image = ldm.embedding_2_img(prompts[i], emb, seed=seed, save_int=False)
            #predict_aesthetic_score.save(f'./output/random/{seed}/{prompts[i][0:30]}.jpg')

            csw_row_images.append(aesthetic_predictor.predict_aesthetic_score(pil_image))
        csv_file_images.append(csw_row_images)
        csv_file_prompts.append(csw_row_prompts)
        write_to_csv(csv_file_images, 'random_prompts_images.csv', '../output/random/')
        write_to_csv(csv_file_prompts, 'random_prompts.csv', '../output/random/')


def numeric_random_prompts():
    aesthetic_predictor = AestheticPredictor()
    prompts = create_random_prompts(1000, numeric=True)

    aesthetic_pred_list = list()
    for i in range(len(prompts)):
        print(prompts[i])
        aesthetic_pred_list.append(aesthetic_predictor.predict_aesthetic_score(prompts[i]))
    create_boxplot(aesthetic_pred_list, filename="numeric_prompts_boxplot.png")


def main():
    numeric_random_prompts()



if __name__ == "__main__":
    main()
