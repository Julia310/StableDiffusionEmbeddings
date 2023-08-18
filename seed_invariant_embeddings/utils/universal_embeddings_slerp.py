from ldm.stable_diffusion import StableDiffusion
from optimizer.adam_on_lion import AdamOnLion
import torch
from utils.create_graphics import plot_scores
import matplotlib.pyplot as plt
import os

seed = 417016


#target_seed = 510675
target_seed = 683395
dim = 512

device = 'cuda'

ldm = StableDiffusion(device=device)




# prompt = 'a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, ' \
#         'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful'

prompt = 'Single Color Ball'

prompt2 = 'Blue Single Color Ball'


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


def compute_dist_metric( target_latents, latents):
    score = 10000 * torch.nn.functional.cosine_similarity(
        target_latents.flatten(start_dim=1, end_dim=-1).to(torch.float64),
        latents.flatten(start_dim=1, end_dim=-1).to(torch.float64), dim=-1)
    return score

class GradientDescent(torch.nn.Module):
    def __init__(self, condition, target_condition, target_latents, target_init_latents, val):
        super().__init__()
        self.condition_row = condition[:, 0, :]
        self.condition = condition[:, 1:, :]
        self.target_init_latents = target_init_latents
        self.target_condition = target_condition[:, 1:, :]
        self.uncondition = ldm.text_enc([""], condition.shape[1])
        self.latents = None
        self.target_latents = target_latents
        self.alpha = torch.nn.Parameter(torch.tensor(-5.))
        self.default_std = torch.std(condition[:, 1:, :])
        self.default_mean = torch.mean(condition[:, 1:, :])
        self.val = val

    def get_text_embedding(self):
        condition = ldm.slerp(self.condition, self.target_condition, torch.sigmoid(self.alpha))
        cond = torch.cat((self.condition_row.unsqueeze(dim=1), condition), dim=1)
        return torch.cat([self.uncondition, cond])

    def forward(self):
        score = 0
        ldm.embedding_2_img(ldm.get_embedding([prompt])[0], dim=dim, seed=seed, return_latents=True,
                            keep_init_latents=False)

        ldm.initial_latents = ldm.slerp(self.target_init_latents, ldm.initial_latents, self.val)

        latents = ldm.embedding_2_img(self.get_text_embedding(), dim=dim,
                                      return_pil=False,
                                      return_latents=True, keep_init_latents=True)

        score = compute_dist_metric(self.target_latents, latents) + score

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
        latents_list = list()
        target_latents = ldm.embedding_2_img(ldm.get_embedding([prompt])[0], dim=dim,
                                             seed=target_seed, return_pil=False,
                                             return_latents=True, keep_init_latents=False)

        target_init_latents = torch.clone(ldm.initial_latents)



    for eta in [0.1, 0.01]:
        os.makedirs(f'./output/interpolation/{eta}')
        val = 0.01

        gd = GradientDescent(
            ldm.text_enc([prompt]),
            ldm.text_enc([prompt2]),
            target_latents,
            target_init_latents,
            val
        )

        optimizer = gd.get_optimizer(eta, 'AdamOnLion')

        interpolation_value = [-5.]
        values = []
        cnt = 0
        batch_cnt = 0

        for i in range(100):
            optimizer.zero_grad()
            score = gd.forward()
            loss = -score
            loss.backward(retain_graph=True)
            optimizer.step()
            interpolation_value.append(gd.alpha.item())
            values.append([val, gd.alpha.item()])

            val = val + 0.0099
            print('update initial latents')
            print(val)
            gd.val = val

#

        plot_scores(interpolation_value, f'output/interpolation/{eta}/interpolation_values.jpg',
                    x_label='Iterations',
                    y_label='alpha')
        plt.clf()
        print(values)