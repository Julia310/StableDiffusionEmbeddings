from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
from utils.create_graphics import plot_scores

seed = 61582
seed2 = 615845
seed3 = 82641
dim = 512


#seed = 724839

#seed = 683395
#seed2 = 417016

device = 'cuda'

ldm = StableDiffusion(device=device)

prompt = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
         'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'

#prompt = 'Single Color Ball'


def get_shifted_embedding(text_embedding, default_std, default_mean):
    shifted_text_embedding = text_embedding / (torch.std(text_embedding)) * default_std
    shifted_text_embedding = shifted_text_embedding - torch.mean(shifted_text_embedding) + default_mean
    return shifted_text_embedding


class GradientDescent(torch.nn.Module):
    def __init__(self, condition, target_latents, comb_init_latents):
        super().__init__()
        self.condition = condition
        self.initial_latents = comb_init_latents
        self.uncondition = ldm.text_enc([""], self.condition.shape[1])
        self.latents = None
        self.target_latents = target_latents
        self.default_std = None
        self.default_mean = None

    def set_torch_parameter(self, condition=False):
        if condition:
            self.condition = torch.nn.Parameter(self.condition)
            self.condition.requires_grad = True
        else:
            self.initial_latents = torch.nn.Parameter(self.initial_latents)
            self.initial_latents.requires_grad = True

    def get_text_embedding(self):
        if self.default_mean is None:
            self.default_mean = torch.mean(self.condition)
            self.default_std = torch.std(self.condition)
        get_shifted_embedding(self.condition, self.default_std, self.default_mean)
        return torch.cat([self.uncondition, self.condition])

    def forward(self):
        ldm.initial_latents = self.initial_latents
        latents = ldm.embedding_2_img('', self.get_text_embedding(), save_img=False, dim=dim, return_pil=False,
                                      return_latents=True, keep_init_latents=True)

        self.latents = latents

        cosine_similarity = torch.nn.functional.cosine_similarity(
            self.target_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64).to(torch.float64),
            latents.flatten(start_dim=1, end_dim=-1).to(torch.float64), dim=-1)

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
        target_latents = ldm.embedding_2_img('', ldm.get_embedding([prompt])[0], save_img=False, dim=dim, seed=seed, return_pil=False,
                                             return_latents=True, keep_init_latents=False)

        target_init_latents = torch.clone(ldm.initial_latents)
        ldm.embedding_2_img('', ldm.get_embedding([prompt])[0], dim=dim, seed=seed2, return_latents=True, keep_init_latents=False)
        latents = torch.clone(ldm.initial_latents)

    #combined_init_latents = ldm.combine_embeddings(target_init_latents, latents, 0.01)
    combined_init_latents = latents

    print(round(100 * torch.nn.functional.cosine_similarity(
                    target_init_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
                    latents.flatten(start_dim=1, end_dim=-1).to(torch.float64)
                ).item(), 4))

    gd_condition = GradientDescent(ldm.text_enc([prompt]), target_latents, combined_init_latents)
    gd_condition.set_torch_parameter(condition=True)
    gd_init_latents = GradientDescent(ldm.text_enc([prompt]), target_latents, combined_init_latents)
    gd_init_latents.set_torch_parameter()



    optimizer_condition = gd_condition.get_optimizer(0.01, 'AdamOnLion')
    optimizer_init_latents = gd_init_latents.get_optimizer(0.001, 'AdamOnLion')

    init_latents_dist_list = list()
    scores_list = list()

    initial_score = 0

    for i in range(100):
        if i >= 0:# or score < initial_score + 0.5:
            gd_condition.initial_latents = torch.clone(gd_init_latents.initial_latents)
            optimizer_condition.zero_grad()
            score = gd_condition.forward()
            if initial_score == 0: initial_score = score.item()
            loss = -score
            loss.backward(retain_graph=True)
            optimizer_condition.step()
            pil_img = ldm.latents_to_image(gd_condition.latents)[0]
        else:
            gd_init_latents.condition = torch.clone(gd_condition.condition)
            optimizer_init_latents.zero_grad()
            score = gd_init_latents.forward()
            print(score)
            scores_list.append(round(score.item(), 4))
            init_latents_dist_list.append(
                round(100 * torch.nn.functional.cosine_similarity(
                    target_init_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
                    gd_init_latents.initial_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64)
                ).item(), 4)
            )

            loss = score
            loss.backward(retain_graph=True)
            optimizer_init_latents.step()
            #pil_img = ldm.latents_to_image(gd_init_latents.latents)[0]

        pil_img.save(f'output/{i}_{prompt[0:25]}_{round(score.item(), 3)}.jpg')
    print(scores_list)
    print(init_latents_dist_list)

    plot_scores(scores_list, r'output/similarity_scores.jpg', x_label='Iterations', y_label='Cosine-Similarity')
    plot_scores(init_latents_dist_list, r'output/init_latent_distances.jpg', x_label='Iterations', y_label='Cosine-Similarity')