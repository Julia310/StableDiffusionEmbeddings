from ldm.stable_diffusion import StableDiffusion
from aesthetic_predictor.gradient_ascent import Gradient_Ascent
import torch
from aesthetic_predictor.simple_inference import AestheticPredictor

prompt = "realistic portrait of a beautiful fox in a fairy wood, 8k, ultra realistic, atmosphere, glow, detailed, " \
         "intricate, full of colour, trending on artstation, masterpiece"

prompt = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
               'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'

prompt = "ugly meme, funniest thing ever"


class ImageImprovement:
    def __init__(self, prompt):
        self.aesthetic_predictor = AestheticPredictor()
        self.ldm = StableDiffusion()
        self.gradient_ascent = Gradient_Ascent()
        self.prompt = prompt
        self.uncondition = self.ldm.text_enc([""], 77)
        self.seed = 3027230121

    def improve_features(self, condition_dim):
        tensor = condition_dim.clone()
        im_features = self.gradient_ascent.get_gradient(tensor, text_input=False, image_input=False)
        normalization = self.gradient_ascent.get_feature_normalization(tensor)
        im_features = torch.from_numpy(im_features.cpu().detach().numpy() * normalization).cuda()
        return im_features

    def improve_condition_dimensions(self, condition):
        condition2 = torch.empty(condition.shape, dtype=torch.float16).cuda()
        condition2[:, 0] = condition[:, 0].clone()
        for i in range(1, condition.shape[1]):
        #for i in range(44, 45):
            condition2[:, i] = self.improve_features(condition[:, i])
        return condition2

    def condition_to_image(self, condition, file_name):
        embedding = torch.cat([self.uncondition, condition])
        pil_image = self.ldm.embedding_2_img(prompt, embedding, seed=self.seed, save_img=False)
        pil_image.save(f'./output/{file_name}')
        self.aesthetic_predictor.predict_aesthetic_score(input=pil_image, image_input=True)

    def improve_image(self, num_images):
        condition = self.ldm.text_enc([prompt])
        print(condition.shape[1])
        self.condition_to_image(condition, f'0_{self.prompt[0:30]}.jpg')
        for i in range(1, num_images + 1):
            condition_new = self.improve_condition_dimensions(condition)
            self.condition_to_image(condition_new, f'{i}_{self.prompt[0:30]}.jpg')
            condition = condition_new.clone()


def main():
    img_improvement = ImageImprovement(prompt)
    img_improvement.improve_image(10)


if __name__ == "__main__":
    main()
