from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch

seed = 61582
seed2 = 615845
seed3 = 82641
dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)

prompt = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
         'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'


def get_shifted_embedding(text_embedding, default_std, default_mean):
    shifted_text_embedding = text_embedding / (torch.std(text_embedding)) * default_std
    shifted_text_embedding = shifted_text_embedding - torch.mean(shifted_text_embedding) + default_mean
    return shifted_text_embedding


def get_latents_angle(latents_target, latents):
    latents_target = latents_target.to(torch.float64)
    latents = latents.to(torch.float64)
    latents_target_norm = latents_target/torch.norm(latents_target, dim=1, keepdim=True)
    latents_norm = latents/torch.norm(latents, dim=1, keepdim=True)
    dot = (latents_target_norm*latents_norm).sum(1)
    omega = torch.acos(dot)
    return omega


class GradientDescent(torch.nn.Module):
    def __init__(self, text_embedding, target_latents, comb_init_latents):
        super().__init__()
        self.text_embedding = torch.nn.Parameter(text_embedding)
        self.text_embedding.requires_grad = True
        self.latents = None
        self.target_latents = target_latents
        ldm.initial_latents = comb_init_latents
        with torch.no_grad():
            self.default_mean = torch.mean(text_embedding)
            self.default_std = torch.std(text_embedding)

    def forward(self):
        # print(f'min: {torch.min(shifted_text_embedding)}, max: {torch.max(torch.max(shifted_text_embedding))}')
        latents = ldm.embedding_2_img('', self.text_embedding, save_img=False, dim=dim, return_pil=False,
                                      return_latents=True, keep_init_latents=True)

        self.latents = latents

        latents = latents.flatten(start_dim=1, end_dim=-1)

        omega = get_latents_angle(self.target_latents, latents)
        #cosine_similarity = torch.nn.functional.cosine_similarity(self.target_latents.to(torch.float64), latents.to(torch.float64), dim=-1)
        #print(f'min: {torch.min(omega)}, max: {torch.max(torch.max(omega))}')
        #omega_mean = torch.mean(omega) * 100
        #distance = torch.dist(self.target_latents.to(torch.float64), latents.to(torch.float64))
        #print(distance)
        #return cosine_similarity * 100
        return omega

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

    min_score = 100
    min_emb = None

    embedding = ldm.get_embedding([prompt])[0]

    with torch.no_grad():
        target_latents = ldm.embedding_2_img('', embedding, save_img=False, dim=dim, seed=seed, return_pil=False,
                                             return_latents=True, keep_init_latents=False)

    target_init_latents = torch.clone(ldm.initial_latents)
    ldm.embedding_2_img('', embedding, dim=dim, seed=seed2, return_latents=True, keep_init_latents=False)

    #combined_init_latents = ldm.combine_embeddings(target_init_latents, ldm.initial_latents, 0.2)
    combined_init_latents = ldm.slerp(target_init_latents, ldm.initial_latents, 0.1)

    target_latents = target_latents.flatten(start_dim=1, end_dim=-1)

    gd = GradientDescent(embedding, target_latents, combined_init_latents)

    eta = 0.01
    num_images = 600

    optimizer = gd.get_optimizer(eta, 'AdamOnLion')

    for i in range(num_images):
        optimizer.zero_grad()
        score = gd.forward()
        loss = score
        loss.backward(retain_graph=True)
        optimizer.step()


        # print(f'std: {torch.std(gd.text_embedding)}, mean: {torch.mean(gd.text_embedding)}')
        #pil_img = ldm.embedding_2_img('', gd.text_embedding,
        #                              return_pil=True, save_img=False,
        #                              return_latents=False, keep_init_latents=True)
        pil_img = ldm.latents_to_image(gd.latents)[0]
        pil_img.save(f'output/{i}_{prompt[0:25]}_{score.item()}.jpg')

        # if (i + 1) % 75 == 0:
        #    gd.text_embedding = torch.nn.Parameter(get_shifted_embedding(gd.text_embedding, gd.default_std, gd.default_mean))
        #    gd.text_embedding.requires_grad = True
