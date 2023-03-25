from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
from utils.image_generation import get_random_seeds

seed = 61582
#seed_list = [892996, 896266, 831445]
seed_list = [370813, 417016, 683395]
#seed_list = [370813, 205620, 221557]
dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)

prompt = 'Single Color Ball'
#target_prompt = 'Black Single Color Ball'


class GradientDescent(torch.nn.Module):
    def __init__(self, condition, target_condition):
        super().__init__()
        self.condition1 = torch.nn.Parameter(condition)
        self.condition1.requires_grad = True
        self.condition2 = torch.nn.Parameter(condition)
        self.condition2.requires_grad = True
        self.condition3 = torch.nn.Parameter(condition)
        self.condition3.requires_grad = True
        self.uncondition = ldm.text_enc([""], condition.shape[1])
        self.target_condition = target_condition

    def get_text_embedding(self, condition):
        return torch.cat([self.uncondition, condition])

    def embedding_to_pil(self, num):
        pil_img = ldm.embedding_2_img('', self.get_text_embedding(self.condition1), save_img=False, dim=dim,
                                      seed=seed_list[0], keep_init_latents=False)
        pil_img.save(f'output/{num}_{seed_list[0]}_{prompt[0:25]}.jpg')
        pil_img = ldm.embedding_2_img('', self.get_text_embedding(self.condition2), save_img=False, dim=dim,
                                      seed=seed_list[1], keep_init_latents=False)
        pil_img.save(f'output/{num}_{seed_list[1]}_{prompt[0:25]}.jpg')
        pil_img = ldm.embedding_2_img('', self.get_text_embedding(self.condition3), save_img=False, dim=dim,
                                      seed=seed_list[2], keep_init_latents=False)
        pil_img.save(f'output/{num}_{seed_list[2]}_{prompt[0:25]}.jpg')

    def forward(self):
        condition1 = self.condition1.flatten(start_dim=1, end_dim=-1)
        cosine_similarity1 = torch.nn.functional.cosine_similarity(self.target_condition.to(torch.float64),
                                                                   condition1.to(torch.float64), dim=-1)
        condition2 = self.condition1.flatten(start_dim=1, end_dim=-1)
        cosine_similarity2 = torch.nn.functional.cosine_similarity(self.target_condition.to(torch.float64),
                                                                   condition2.to(torch.float64), dim=-1)
        condition3 = self.condition1.flatten(start_dim=1, end_dim=-1)
        cosine_similarity3 = torch.nn.functional.cosine_similarity(self.target_condition.to(torch.float64),
                                                                   condition3.to(torch.float64), dim=-1)
        return (cosine_similarity1 + cosine_similarity2 + cosine_similarity3) * 100

    def get_optimizer(self, eta, optim='SGD'):
        if optim == 'SGD':
            return torch.optim.SGD(
                self.parameters(),
                lr=eta,
                momentum=0.95,
                nesterov=True
            )
        elif optim == 'AdamTorch':
            return torch.optim.Adam(
                self.parameters(),
                lr=eta
                # eps=0.00000001
            )
        else:
            return AdamOnLion(
                params=self.parameters(),
                lr=eta,
                eps=0.001,
            )


if __name__ == '__main__':
    """max_score = 0
    max_emb = None
    gd = GradientDescent(ldm.text_enc([prompt]), ldm.text_enc([target_prompt]).flatten(start_dim=1, end_dim=-1))

    num_images = 200

    optimizer = gd.get_optimizer(eta=0.01, optim='AdamOnLion')

    for i in range(num_images):
        optimizer.zero_grad()
        score = gd.forward()
        print(score)
        loss = -score
        loss.backward(retain_graph=True)
        optimizer.step()
        gd.latents_to_pil(i)"""

    seed_list = get_random_seeds(30)
    for seed in seed_list:
        pil_img = ldm.embedding_2_img('', ldm.get_embedding([prompt])[0], save_img=False, dim=dim, seed=seed,
                                        keep_init_latents=False)
        pil_img.save(f'output/{prompt[0:25]}_{seed}.jpg')