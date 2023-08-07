from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
from utils.create_graphics import plot_scores
import matplotlib.pyplot as plt
import os

seed = 417016
seed2 = 683395
seeds = [
    [417016, 683395, 370813],
    [222261, 23916, 635868],
    [752801, 543920, 354007],
    [466388, 662243, 871288],
    [935806, 329084, 466388]
]


seeds = [
    [417016, 683395],
    [222261, 370813],
    [23916, 635868],
    [752801, 543920],
    [466388, 354007],
    [662243, 871288],
    [935806, 329084],
    [205620, 466388]
]

target_seed = 510675
#target_seed = 683395
dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)




# prompt = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
#         'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'

prompt = 'Single Color Ball'

prompt2 = 'Black Single Color Ball'

#interpolation only a single batch at a time
#interpolation1 all batches


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


def compute_dist_metric(metric, target_latents, latents):
    if metric == 'Euclidean Distance':
        score = torch.dist(
            target_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
            latents.flatten(start_dim=1, end_dim=-1).to(torch.float64))
    else:
        score = 10000 * torch.nn.functional.cosine_similarity(
            target_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
            latents.flatten(start_dim=1, end_dim=-1).to(torch.float64), dim=-1)
    return score


def get_shifted_embedding(text_embedding, default_std, default_mean):
    shifted_text_embedding = text_embedding / (torch.std(text_embedding)) * default_std
    shifted_text_embedding = shifted_text_embedding - torch.mean(shifted_text_embedding) + default_mean
    return shifted_text_embedding


class GradientDescent(torch.nn.Module):
    def __init__(self, condition, target_latents, target_init_latents, val):
        super().__init__()
        self.condition_row = condition[:, 0, :]
        self.condition = torch.nn.Parameter(condition[:, -1, :])
        self.target_init_latents = target_init_latents
        self.uncondition = ldm.text_enc([""], condition.shape[1])
        self.latents = None
        self.target_latents = target_latents
        self.default_std = torch.std(condition[:, 1:, :])
        self.default_mean = torch.mean(condition[:, 1:, :])
        self.val = val


    def get_text_embedding(self):
        condition = self.condition.repeat(1, 76, 1)
        cond = torch.cat((self.condition_row.unsqueeze(dim=1), condition), dim=1)
        return torch.cat([self.uncondition, cond])

    def forward(self, i, region_indices, metric, seed_batch):
        print(seed_batch)
        score = 0
        for j in range(len(seed_batch)):
            ldm.embedding_2_img('', ldm.get_embedding([prompt])[0], dim=dim, seed=seed_batch[j], return_latents=True,
                                keep_init_latents=False)

            ldm.initial_latents = ldm.slerp(self.target_init_latents, ldm.initial_latents, self.val)



            latents = ldm.embedding_2_img('', self.get_text_embedding(), save_img=False, dim=dim,
                                          return_pil=False,
                                          return_latents=True, keep_init_latents=True)


            if region_indices == "every 4":
                target_latents = self.target_latents[:, :, ::4, ::4]
                latents = latents[:, :, ::4, ::4]
            elif region_indices == "every 4 alternating":
                target_latents = self.target_latents[:, :, i % 4::4, i % 4::4]
                latents = latents[:, :, i % 4::4, i % 4::4]
            else:
                target_latents = self.target_latents

            score = compute_dist_metric(metric, target_latents, latents) + score

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
        target_latents = ldm.embedding_2_img('', ldm.get_embedding([prompt])[0], save_img=False, dim=dim,
                                             seed=target_seed, return_pil=False,
                                             return_latents=True, keep_init_latents=False)

        target_init_latents = torch.clone(ldm.initial_latents)



    for eta in [0.01, 0.1, 1., 10., 100.]:
        os.mkdir(f'./output/interpolation1/{eta}')
        for score_metric in ['Cosine Similarity', 'Euclidean Distance']:
            for region in ['complete', 'every 4', 'every 4 alternating']:
                for mod in [2, 10]:
                    val = 0.01

                    max_cond = None
                    max_score = -100000
                    if score_metric == 'Euclidean Distance': max_score = 100000
                    dir_num = create_next_directory(f'output/interpolation1/{eta}')

                    print('========================================')
                    print(f' {dir_num}.')
                    print(f'region: {region}')
                    print(score_metric)
                    print(f'embedding update for {mod - 1} iterations')




                    gd = GradientDescent(
                        ldm.text_enc([prompt]),
                        target_latents,
                        target_init_latents,
                        val
                    )

                    optimizer = gd.get_optimizer(eta, 'AdamOnLion')

                    cnt = 0
                    batch_cnt = 0

                    for i in range(mod * 100):
                        cnt += 1
                        cnt = cnt % mod
                        if (i + 1) % mod != 0:
                            seed_batch = seeds [batch_cnt]
                            batch_cnt += 1
                            if batch_cnt == len(seeds): batch_cnt = 0
                            optimizer.zero_grad()

                            for batch in seeds:
                                score = gd.forward(int((i - cnt + 1) / 2), region, score_metric, batch)
                            #score = gd.forward(int((i - cnt + 1) / 2), region, score_metric, seeds[batch_cnt])
                            print(int((i - cnt + 1) / 2))
                            if score_metric == 'Euclidean Distance':
                                loss = score
                                if score.item() < max_score:
                                    max_score = score.item()
                                    max_cond = torch.clone(gd.condition)
                            elif score_metric == 'Cosine Similarity':
                                loss = -score
                                if score.item() > max_score:
                                    max_score = score.item()
                                    max_cond = torch.clone(gd.condition)
                            loss.backward(retain_graph=True)
                            optimizer.step()
                            #interpolation_value.append(gd.alpha.item())
                        else:
                            gd.condition = torch.nn.Parameter(
                                get_shifted_embedding(max_cond, gd.default_std, gd.default_mean))
                            gd.condition.requires_grad = True
                            optimizer_condition = gd.get_optimizer(eta, 'AdamOnLion')
                            val = val + 0.0099
                            print('update initial latents')
                            print(val)
                            gd.val = val

                            pil_img = ldm.embedding_2_img('', gd.get_text_embedding(), save_img=False,
                                                                                        dim=dim, return_pil=True,
                                                                                        return_latents=False,
                                                          keep_init_latents=False,
                                                          seed=seed)

                            pil_img.save(f'output/interpolation1/{eta}/{dir_num}/A_{i}_{prompt[0:25]}_{round(score.item(), 3)}_{round(val, 2)}.jpg')
                            del pil_img

                            max_cond = None
                            max_score = -100000
                            if score_metric == 'Euclidean Distance': max_score = 100000

                    #plot_scores(interpolation_value, f'output/interpolation/{eta}/{dir_num}/interpolation_values.jpg',
                    #            x_label='Iterations',
                    #            y_label='alpha')
                    #plt.clf()