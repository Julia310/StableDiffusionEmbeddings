from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
from utils.image_generation import get_random_seeds

seed = 61582
seed_list = [417016, 683395, 724839]
dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)

prompt = 'Single Color Ball'
target_prompt = 'Black Single Color Ball'


class GradientDescent(torch.nn.Module):
    def __init__(self, condition, target_latents):
        super().__init__()
        self.condition1 = torch.nn.Parameter(condition)
        self.condition1.requires_grad = True
        #self.condition2 = torch.nn.Parameter(condition)
        #self.condition2.requires_grad = True
        self.uncondition = ldm.text_enc([""], condition.shape[1])
        self.target_latents = target_latents
        self.latents_list = [None, None]

    def get_text_embedding(self, condition):
        return torch.cat([self.uncondition, condition])

    def latents_to_pil(self, num):
        pil_img = ldm.latents_to_image(self.latents_list[0])[0]
        pil_img.save(f'output/{num}_{seed_list[0]}_{prompt[0:25]}.jpg')
        #pil_img = ldm.latents_to_image(self.latents_list[1])[0]
        #pil_img.save(f'output/{num}_{seed_list[1]}_{prompt[0:25]}.jpg')

    def forward(self):

        latents = ldm.embedding_2_img('',
                                      self.get_text_embedding(self.condition1),
                                      seed=seed_list[0],
                                      save_img=False,
                                      dim=dim,
                                      return_pil=False,
                                      return_latents=True,
                                      keep_init_latents=False)
        self.latents_list[0] = torch.clone(latents)
        latents = latents.flatten(start_dim=1,
                                  end_dim=-1)
        cosine_similarity = torch.nn.functional.cosine_similarity(self.target_latents.to(torch.float64),
                                                                  latents.to(torch.float64),
                                                                  dim=-1)

        #latents = ldm.embedding_2_img('',
        #                              self.get_text_embedding(self.condition2),
        #                              seed=seed_list[1],
        #                              save_img=False,
        #                              dim=dim,
        #                              return_pil=False,
        #                              return_latents=True,
        #                              keep_init_latents=False)
        #self.latents_list[1] = torch.clone(latents)
        #latents = latents.flatten(start_dim=1, end_dim=-1)
        #cosine_similarity = cosine_similarity+\
        #                    torch.nn.functional.cosine_similarity(self.target_latents.to(torch.float64),
        #                                                          latents.to(torch.float64),
        #                                                          dim=-1)


        return cosine_similarity * 100

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
    max_score = 0
    max_emb = None

    with torch.no_grad():
        target_latents = ldm.embedding_2_img('',
                                             ldm.get_embedding([target_prompt])[0],
                                             save_img=False,
                                             dim=dim,
                                             seed=seed_list[2],
                                             return_pil=False,
                                             return_latents=True,
                                             keep_init_latents=False)

    gd = GradientDescent(ldm.text_enc([prompt]),
                         target_latents.flatten(start_dim=1, end_dim=-1))


    num_images = 200

    optimizer = gd.get_optimizer(eta=0.01,
                                 optim='AdamOnLion')

    for i in range(num_images):
        optimizer.zero_grad()
        score = gd.forward()
        print(score)
        gd.latents_to_pil(i)
        loss = -score
        loss.backward(retain_graph=True)
        optimizer.step()
    gd.latents_to_pil(num_images)
