import torch
from transformers import CLIPTextModel, CLIPTokenizer
from aesthetic_predictor.simple_inference import AestheticPredictor
import numpy as np
from numpy.linalg import norm



prompt = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
               'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'

prompt2 = 'full body image of a norwegian forest cat of white and ginger fur, by dan mumford, yusuke murata and ' \
               'makoto shinkai, 8k, cel shaded, unreal engine, featured on artstation, pixiv'

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14",
                                             torch_dtype=torch.float16).to("cuda")


def get_clip_embedding(prompts, maxlen = None):

    if maxlen is None: maxlen = tokenizer.model_max_length
    inp = tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
    return text_encoder(inp.input_ids.to("cuda"))[0].half()


def compare_tensors(tensor1, tensor2, comp = 1):
    tens_b = tensor2
    equal = False
    size = tensor1.shape[1]
    for i in range(size):
        tens_a = tensor1[:, i]

        if comp == 1:
            equal = tens_a.eq(tens_b)

        elif comp == 2:
            equal = []
            for j in range(tensor1.shape[2]):
                a = float(tens_a[0][j])
                b = float(tens_b[0][j])
                if 0.8 < a / b < 1.2:
                    equal.append(True)
                else:
                    return False

        else:
            embedding_similarities = list()
            tens_b = torch.flatten(tensor2.cpu()).to("cpu").detach().numpy()
            for j in range(tensor1.shape[1]):
                tens_a = tensor1[:, j]
                tens_a = torch.flatten(tens_a).to("cpu").detach().numpy()
                # embedding_similarities.append(float(cos(tens_a, tens_b)))
                embedding_similarities.append(np.dot(tens_a, tens_b) / (norm(tens_a, ord=2) * norm(tens_b, ord=2)))
            return max(embedding_similarities)
    return torch.all(equal)


def compare_clip_embeddings(comp = 3):
    aesthetic_predictor = AestheticPredictor()
    emb_1 = get_clip_embedding([prompt])
    emb_2 = aesthetic_predictor.get_features(input=prompt, image_input=False, normalize=False)
    print(compare_tensors(emb_1, emb_2, comp=comp))
    print('')


def compare_ldm_conditions(comp = 1):
    emb_1 = get_clip_embedding([prompt])
    emb_2 = get_clip_embedding([prompt2])
    equal_dims = list()
    similar_values = list()
    for i in range(emb_1.shape[1]):
        tens_a = emb_1[:, i]
        tens_b = emb_2[:, i]
        #print(tens_a.eq(tens_b))
        if comp == 1:
            if torch.all(tens_a.eq(tens_b)):
                equal_dims.append(i)
                print('equal for index: ' + str(i))
        elif comp == 2:
            tens_a = torch.flatten(tens_a).to("cpu").detach().numpy()
            tens_b = torch.flatten(tens_b).to("cpu").detach().numpy()
            equal_dims.append(np.dot(tens_a, tens_b) / (norm(tens_a, ord=2) * norm(tens_b, ord=2)))
            #return embedding_similarities
        else:
            for j in range(emb_1.shape[2]):
                if i == 0:
                    break
                a = float(tens_a[0, j])
                b = float(tens_b[0, j])
                if 0.8 < a / b < 1.2:
                    similar_values.append([i, j])
                if j == emb_1.shape[2] - 1:
                    equal_dims.append([i, len(similar_values)])
                    similar_values = list()

    return equal_dims


def test_embedding(comp = 1):
    emb_1 = get_clip_embedding([prompt])
    equal_dims = list()
    for i in range(emb_1.shape[1] - 1):
        similar_values = list()
        tens_a = emb_1[:, i]
        tens_b = emb_1[:, i + 1]
        for j in range(emb_1.shape[2]):
            if i == 0:
                break
            a = float(tens_a[0, j])
            b = float(tens_b[0, j])
            if comp == 1:
                if a==b:
                    similar_values.append([i, j])
            if comp == 2:
                if 0.8 < a / b < 1.2:
                    similar_values.append([i, j])

        equal_dims.append([i, len(similar_values)])
        print(similar_values)
    return equal_dims


if __name__ == '__main__':
    #compare_clip_embeddings(comp=3)
    test_embedding()