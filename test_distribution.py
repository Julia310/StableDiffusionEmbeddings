from scipy.stats import anderson, kstest, logistic
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from utils import make_dir, retrieve_prompts
from stable_diffusion import StableDiffusion
import torch
import numpy as np
import os
from fitter import Fitter, get_distributions, get_common_distributions


def test_normality_anderson(data):
    result = anderson(data)
    sig_level, crit_val = result.significance_level[2], result.critical_values[2]
    if result.statistic < crit_val:
        print(f'Probability Gaussian : {crit_val} critical value at {sig_level} level of significance')
    else:
        print(f'Probability not Gaussian : {crit_val} critical value at {sig_level} level of significance')


def test_normality_kstest(data):
    statistic, pvalue = kstest(data, 'norm')
    print('statistic=%3f, p=%3f\n' %(statistic, pvalue))
    if pvalue > 0.05:
        print('Gaussian')
    else:
        print('not Gaussian')


def plot_histogram(data, path, prompt):
    sns.histplot(data, kde=True)
    make_dir(path)
    plt.savefig(f'{path}/{prompt[0:30]}.png')


def fit_distribution(data):
    f = Fitter(data,
               distributions=['laplace', 'logistic', 'genhyperbolic'] + get_common_distributions(), timeout=60)
    f.fit()
    f.df_errors.sort_values('sumsquare_error')
    print(f.summary())
    print(f.get_best(method='sumsquare_error'))


def plot_distribution(data, dist_name):
    dist = getattr(scipy.stats, dist_name)

    # Fit a distribution to the data
    params = dist.fit(data)

    # Plot and save the PDF
    x = np.linspace(np.min(data), np.max(data))
    p = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
    plt.plot(x, p,)


def save_distr_plot(prompt, path, dist_name):
    make_dir(f'{path}')
    plt.savefig(f'{path}/{prompt[0:30].replace(" ", "_")}_{dist_name}.png')
    plt.close()



def test_distributions(data, prompt, dist_names):
    # Distributions to check
    #dist_names = ['laplace', 'logistic', 'norm', 'gamma', 'hypsecant', 'genhyperbolic']

    for dist_name in dist_names:
        # Plot the histogram
        plt.hist(data, bins=100, density=True)
        plot_distribution(data, dist_name)

        title = 'Distribution: ' + dist_name
        plt.title(title)
        save_distr_plot(prompt, './output/distr_test', dist_name)


def test_normality(data, prompt, test = 'anderson', plot_distr = False):
    print(prompt)
    if test == 'anderson':
        test_normality_anderson(data)
    elif test == 'kstest':
        test_normality_kstest(data)
    print('')
    if plot_distr:
        plot_distribution(data, prompt)


if __name__ == "__main__":
    prompts = retrieve_prompts()
    ldm = StableDiffusion()
    for prompt in prompts:
        print('=======================================')
        print(prompt)
        emb = ldm.get_embedding(prompts)
        emb_flat = torch.flatten(emb)
        #test_normality(emb_flat.cpu().detach().numpy(), prompt, test = 'kstest')
        #test_distributions(emb_flat.cpu().detach().numpy(), prompt, ['norm', 'logistic', 'genhyperbolic'])
        fit_distribution(emb_flat.cpu().detach().numpy())

        print('')

