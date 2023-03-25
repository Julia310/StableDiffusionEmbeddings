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
    def __init__(self, condition, target_image):
        super().__init__()
        self.condition1 = torch.nn.Parameter(condition)
        self.condition1.requires_grad = True
        #self.condition2 = torch.nn.Parameter(condition)
        #self.condition2.requires_grad = True
        self.uncondition = ldm.text_enc([""], condition.shape[1])
        self.target_image = target_image
        self.latents = None

    def get_text_embedding(self, condition):
        return torch.cat([self.uncondition, condition])

    def forward(self):

        self.latents = ldm.embedding_2_img('',
                                           self.get_text_embedding(self.condition1),
                                           seed=seed_list[0],
                                           save_img=False,
                                           dim=dim,
                                           return_pil=False,
                                           return_latents=True,
                                           keep_init_latents=False)

        image = ldm.latents_to_image(self.latents,
                                     return_pil=False)
        image = image.flatten(start_dim=1,
                              end_dim=-1)
        cosine_similarity = torch.nn.functional.cosine_similarity(self.target_image.to(torch.float64),
                                                                  image.to(torch.float64),
                                                                  dim=-1)



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
    max_cond = None
    update_optimizer = True
    update_optimizer2 = True
    eta = 0.01
    target_score = 73.76

    with torch.no_grad():
        latents = ldm.embedding_2_img('',
                                      ldm.get_embedding([target_prompt])[0],
                                      seed=seed_list[2],
                                      save_img=False,
                                      dim=dim,
                                      return_pil=False,
                                      return_latents=True,
                                      keep_init_latents=False)
        target_image = ldm.latents_to_image(latents,
                                            return_pil=False)


    gd = GradientDescent(ldm.text_enc([prompt]),
                         target_image.flatten(start_dim=1, end_dim=-1))


    num_images = 200

    optimizer = gd.get_optimizer(eta=eta,
                                 optim='AdamOnLion')

    for i in range(num_images):
        optimizer.zero_grad()
        score = gd.forward()
        print(score)
        #if score > max_score:
        #    max_score = score.item()
        #    max_cond = torch.clone(gd.condition1)
        if score > 73.76 and update_optimizer:
            optimizer = gd.get_optimizer(eta=0.001,
                                         optim='AdamOnLion')
            optimizer.zero_grad()
            update_optimizer = False

        if score > 73.78 and update_optimizer2:
            optimizer = gd.get_optimizer(eta=0.0001,
                                         optim='AdamOnLion')
            optimizer.zero_grad()
            update_optimizer2 = False

        pil_img = ldm.latents_to_image(gd.latents)[0]
        pil_img.save(f'output/{i}_{seed_list[0]}_{prompt[0:25]}_{round(score.item(), 4)}.jpg')
        loss = -score
        loss.backward(retain_graph=True)
        optimizer.step()

        #if (i + 1) % 35 == 0:
        #    print('update condition')
        #    gd.condition1 = torch.nn.Parameter(torch.clone(max_cond))
        #    gd.condition1.requires_grad = True
    gd.latents_to_pil(num_images)


