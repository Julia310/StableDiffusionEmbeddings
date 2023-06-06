from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
from utils.create_graphics import plot_scores
import matplotlib.pyplot as plt
import os

seed = 61582
seed2 = 615845
seed3 = 82641
dim = 512
seed = 683395
seed2 = 417016

#seed = 724839

#seed = 683395
#seed2 = 417016

device = 'cuda'

ldm = StableDiffusion(device=device)

#prompt = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
#         'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'

prompt = 'Single Color Ball'



def get_shifted_embedding(text_embedding, default_std, default_mean):
    shifted_text_embedding = text_embedding / (torch.std(text_embedding)) * default_std
    shifted_text_embedding = shifted_text_embedding - torch.mean(shifted_text_embedding) + default_mean
    return shifted_text_embedding

def create_next_directory(directory):
    numerical_directories = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and item.isdigit():
            numerical_directories.append(int(item))

    if numerical_directories:
        next_number = max(numerical_directories) + 1
    else:
        next_number = 1

    new_directory_name = str(next_number)
    new_directory_path = os.path.join(directory, new_directory_name)
    os.mkdir(new_directory_path)

    return next_number


class GradientDescent(torch.nn.Module):
    def __init__(self, condition, target_latents, comb_init_latents):
        super().__init__()
        self.condition_row = condition[:, 0, :]
        self.condition = condition[:, 1:, :]
        self.initial_latents = comb_init_latents
        self.uncondition = ldm.text_enc([""], condition.shape[1])[:, 1:, :]
        self.latents = None
        self.target_latents = target_latents
        self.alpha = torch.tensor(0.5)
        self.default_cond_std = torch.std(condition[:, 1:, :])
        self.default_cond_mean = torch.mean(condition[:, 1:, :])
        self.default_uncond_std = torch.std(self.uncondition)
        self.default_uncond_mean = torch.mean(self.uncondition)


    def get_text_embedding(self):
        cond = torch.cat((self.condition_row.unsqueeze(dim=1), self.condition), dim=1)
        uncond = torch.cat((self.condition_row.unsqueeze(dim=1), self.uncondition), dim=1)
        return torch.cat([uncond, cond])

    def forward(self, i, region_indices, metric):
        ldm.initial_latents = self.initial_latents
        latents = ldm.embedding_2_img('', self.get_text_embedding(), save_img=False, dim=dim, return_pil=False,
                                      return_latents=True, keep_init_latents=True)

        if region_indices == "every 4":
            target_latents = self.target_latents[:, :, ::4, ::4]
            latents = latents[:, :, ::4, ::4]
        elif region_indices == "every 4 alternating":
            target_latents = self.target_latents[:, :, i % 4::4, i % 4::4]
            latents = latents[:, :, i % 4::4, i % 4::4]
        elif region_indices == "center":
            target_latents = self.target_latents[:, :, 30:34, 30:34]
            latents = latents[:, :, 30:34, 30:34]
        else:
            target_latents = self.target_latents

        if metric == 'Euclidean Distance':
            score = torch.dist(
                target_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
                latents.flatten(start_dim=1, end_dim=-1).to(torch.float64))
        else:
            score = 10000 * torch.nn.functional.cosine_similarity(
                target_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
                latents.flatten(start_dim=1, end_dim=-1).to(torch.float64), dim=-1)

        return score

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
    with torch.no_grad():
        target_latents = ldm.embedding_2_img('', ldm.get_embedding([prompt])[0], save_img=False, dim=dim, seed=seed, return_pil=False,
                                             return_latents=True, keep_init_latents=False)

        target_init_latents = torch.clone(ldm.initial_latents)
        ldm.embedding_2_img('', ldm.get_embedding([prompt])[0], dim=dim, seed=seed2, return_latents=True, keep_init_latents=False)
        combined_init_latents = ldm.combine_embeddings(target_init_latents, ldm.initial_latents, 0.05)

    init_lat_mean = torch.mean(combined_init_latents).item()
    init_lat_std = torch.std(combined_init_latents).item()


    for score_metric in ['Euclidean Distance', 'Cosine Similarity']:
        for region in ['complete', 'every 4', 'every 4 alternating', 'center']:
            for mod in [2, 10]:
                #for learning_rates in [(0.01, 0.001), (0.1, 0.001), (0.01, 0.0001), (0.1, 0.0001), (1, 0.0001), (10, 0.0001)]:
                #for learning_rates in [(10, 0.001), (1, 0.001)]:
                for learning_rates in [(100, 0.001), (10, 0.001), (1, 0.001), (0.01, 0.001), (0.1, 0.001),
                                       (0.01, 0.0001),
                                       (0.1, 0.0001), (1, 0.0001), (10, 0.0001), (100, 0.0001)]:
                    #if score_metric == 'Euclidean Distance': continue
                    #if region in ['complete', 'every 4', 'center']: continue


                    lr_latents = learning_rates[1]
                    lr_cond = learning_rates[0]


                    init_latents_dist_list = list()
                    scores_list = list()
                    cnt = 0
                    dir_num = create_next_directory(f'output/embedding')

                    print('========================================')
                    print(f' {dir_num}.')
                    print(f'cond_lr: {lr_cond}, lr_latents: {lr_latents}')
                    print(f'region: {region}')
                    print(score_metric)
                    print(f'embedding update for {mod - 1} iterations')

                    max_cond = None
                    max_uncond = None
                    max_score = -100000
                    if score_metric == 'Euclidean Distance': max_score = 100000

                    gd_condition = GradientDescent(ldm.text_enc([prompt]), target_latents, combined_init_latents)
                    gd_condition.condition = torch.nn.Parameter(gd_condition.condition)
                    gd_condition.condition.requires_grad = True
                    gd_condition.uncondition = torch.nn.Parameter(gd_condition.uncondition)
                    gd_condition.uncondition.requires_grad = True
                    gd_init_latents = GradientDescent(ldm.text_enc([prompt]), target_latents, combined_init_latents)
                    gd_init_latents.initial_latents = torch.nn.Parameter(combined_init_latents)
                    gd_init_latents.initial_latents.requires_grad = True
                    optimizer_condition = gd_condition.get_optimizer(lr_cond, 'AdamOnLion')
                    optimizer_init_latents = gd_init_latents.get_optimizer(lr_latents, 'AdamOnLion')

                    for i in range(mod * 35):
                        cnt += 1
                        cnt = cnt % mod
                        if (i+1) % mod != 0: #i >= 0:
                            gd_condition.initial_latents = torch.clone(gd_init_latents.initial_latents)
                            optimizer_condition.zero_grad()
                            score = gd_condition.forward(int((i-cnt + 1)/2), region, score_metric)

                            print(int((i-cnt + 1)/2))

                            print(f'score: {score.item()}, max_score: {max_score}')

                            if score_metric == 'Euclidean Distance':
                                loss = score
                                if score.item() < max_score:
                                    max_score = score.item()
                                    max_cond = torch.clone(gd_condition.condition)
                                    max_uncond = torch.clone(gd_condition.uncondition)
                            elif score_metric == 'Cosine Similarity':
                                loss = -score
                                if score.item() > max_score:
                                    max_score = score.item()
                                    max_cond = torch.clone(gd_condition.condition)
                                    max_uncond = torch.clone(gd_condition.uncondition)
                            loss.backward(retain_graph=True)
                            optimizer_condition.step()
                        else:
                            gd_condition.condition = torch.nn.Parameter(get_shifted_embedding(max_cond, gd_condition.default_cond_std, gd_condition.default_cond_mean))
                            gd_condition.condition.requires_grad = True
                            gd_condition.uncondition = torch.nn.Parameter(get_shifted_embedding(max_uncond, gd_condition.default_uncond_std, gd_condition.default_uncond_mean))
                            gd_condition.uncondition.requires_grad = True
                            optimizer_condition = gd_condition.get_optimizer(lr_cond, 'AdamOnLion')
                            gd_init_latents.condition = torch.clone(gd_condition.condition)
                            gd_init_latents.uncondition = torch.clone(gd_condition.uncondition)


                            optimizer_init_latents.zero_grad()
                            print("i-cnt-1")
                            print(int((i - mod + 1) / 2))
                            score = gd_init_latents.forward(int((i - mod + 1) / 2), region, score_metric)
                            print(score)

                            #pil_img = ldm.latents_to_image(gd_init_latents.latents)[0]
                            pil_img = ldm.embedding_2_img('', gd_init_latents.get_text_embedding(), save_img=False, dim=dim, return_pil=True,
                                                          return_latents=False)
                            pil_img.save(f'output/embedding/{dir_num}/{i}_B_{prompt[0:25]}_{round(score.item(), 3)}.jpg')

                            scores_list.append(round(score.item(), 4))
                            if score_metric == 'Euclidean Distance':
                                init_latents_dist_list.append(
                                    round(torch.dist(
                                        target_init_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
                                        gd_init_latents.initial_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64)
                                    ).item(), 4)
                                )

                                loss = -score
                            elif score_metric == 'Cosine Similarity':
                                init_latents_dist_list.append(
                                    round(10000 * torch.nn.functional.cosine_similarity(
                                        target_init_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
                                        gd_init_latents.initial_latents.flatten(start_dim=1, end_dim=-1).to(
                                            torch.float64)
                                    ).item(), 4)
                                )

                                loss = score
                            loss.backward(retain_graph=True)
                            optimizer_init_latents.step()


                            print('shift latents')
                            gd_init_latents.initial_latents  = torch.nn.Parameter(get_shifted_embedding(gd_init_latents.initial_latents, init_lat_std, init_lat_mean))
                            optimizer_init_latents = gd_init_latents.get_optimizer(lr_latents, 'AdamOnLion')
                            #optimizer_init_latents.params = gd_init_latents.parameters()


                            print(f'max_score: {max_score}')

                            max_cond = None
                            max_uncond = None
                            max_score = -100000
                            if score_metric == 'Euclidean Distance': max_score = 100000


                    print(scores_list)
                    print(init_latents_dist_list)

                    plot_scores(init_latents_dist_list, f'output/embedding/{dir_num}/init_latent_distances.jpg', x_label='Iterations', y_label=score_metric)
                    plot_scores(scores_list, f'output/embedding/{dir_num}/similarity_scores.jpg', x_label='Iterations', y_label=score_metric)
                    plt.clf()


