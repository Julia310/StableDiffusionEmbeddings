from stable_diffusion import StableDiffusion
from perturbations import sample_noise

def main():
    ldm = StableDiffusion()
    prompt = [
        "illustration of raven watching the planet underneath, d & d, rule of thirds, fantasy, intricate, elegant, "
        "highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, "
        "art by dragolisco "
    ]

    emb = ldm.get_embedding(prompt)[0]
    noise = sample_noise(emb)


if __name__ == "__main__":
    main()
