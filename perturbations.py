import torch
from torch.distributions.normal import Normal


def sample_noise(embedding):
    std = torch.std(embedding)
    mean = torch.mean(embedding)
    shape = embedding.shape
    sampler = Normal(mean, std)
    sample = sampler.sample(shape)

    return sample
