from scipy.stats import anderson, kstest
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
from utils import make_dir, retrieve_prompts, create_random_prompts
from ldm.stable_diffusion import StableDiffusion
import torch
import numpy as np
from fitter import Fitter, get_common_distributions

prompts = ["goddess made of ice long hair like a waterfall, full body, horizontal symmetry!, elegant, intricate, highly detailed, fractal background, digital painting, artstation, concept art, wallpaper, smooth, sharp focus, illustration, epic light, art by kay nielsen and zeen chin and wadim kashin and sangyeob park, terada katsuya ",
"a dad angry at missing his flight from prague to nyc, the dad is drunk ",
"mandelbrot 3 d volume fractal mandala ceramic chakra digital color stylized an ancient white bone and emerald gemstone relic, intricate engraving concept substance patern texture natural color scheme, global illumination ray tracing hdr fanart arstation by sung choi and eric pfeiffer and gabriel garza and casper konefal ",
"spongebob's house, 16 bit,",
"oil painting portrait of brock pierce, american flag on background, cowboy style. ",
"Oh no",
"The last thing a human sees before death",
"Steve Buscemi as Willy Wonka",
"very old and very detailed great canvas with wonderful gradient from warm to cold tones, multilayer last supper johannes itten and hiroshi nagai colors,, pattern of escher style 3 6 0 panorama with hieronymus bosch style bubbles, contrast of light and shadows, unfinished,, digital 4 k, super resolution ",
"Monkey Pointing a Gun at a Computer Meme",
"bright psychedelic portrait of tom waits baking pizza, diffuse lighting, fantasy, intricate, elegant, highly detailed, lifelike, photorealistic, digital painting, artstation, illustration, concept art, smooth, sharp focus, art by John Collier and Albert Aublet and Krenz Cushart and Artem Demura and Alphonse Mucha",
"elon troll face ",
"vladimur putin on a magic the gathering card ",
"a photo of an banana ",
"captain obvious",
" warmly lit close up studio portrait of young angry! teenage Cosmo Kramer angrily singing, impasto oil painting thick brushstrokes by Cy Twombly and Anselm Kiefer , trending on artstation dramatic lighting Expressionism",
" illustration by mel ramos artstation hyper realistic 4 k poster ",
"A black hole with event horizon in the center with space around it, high detail, Junji Ito",
"todd solondz, high quality high detail graphic novel of todd solondz, clear sharp face of todd solondz, night, by lucian freud and gregory crewdson and francis bacon, ",
"A photo of vladimir putin the barbarian sitting on his throne, award winning photography, sigma 85mm Lens F/1.4, perfect faces",
"Beans, what the fuck",
"an object ",
"oil painting portrait of brock pierce, american flag on background, cowboy style. ",
"portrait of darth vader in the oval office. secret service. intricate abstract. intricate artwork. by tooth wu, wlop, beeple, dan mumford. octane render, trending on artstation, greg rutkowski very coherent symmetrical artwork. cinematic, hyper realism, high detail, octane render, 8 k, iridescent accents ",
"the funniest thing I've ever seen!",
"portrait of a splendid sad north korean woman from scary stories to tell in the dark in ibiza, spain with pearlescent skin and cyber yellow hair stronghold in the style of chalks by andre kohn, trending on 5 0 0 px : : 5, hdr, 8 k resolution, ray traced, screen space ambient occlusion : : 2, true blue color scheme ",
"flat primitive drawing, front view, full face, aqueduct with 4 arches ",
"oil painting portrait of brock pierce, american flag on background, cowboy style. ",
"A photo of vladimir putin the barbarian sitting on his throne, award winning photography, sigma 85mm Lens F/1.4, perfect faces",
"ascii art of george costanza frustrated by a bad donut ",
"illusory motion optical illusion ",
"elon troll face ",
"an average human being ",
"A human made of pizza, digital art",
"the shape silhouette of a fluffy cat, stacked plot of radio emissions from a pulsar, abstracted light refractions and stripy interference, making up a cat isolated on black, highly detailed high resolution, silk screen t-shirt design in the style of FELIPE PANTONE 4K",
"a fibonacci sequence, cascading. retro minimalist art by jean giraud. ",
"willy wonka giving a tour inside his dogfood factory, horse guillotine, dancing oompa loompas covered in blood ",
"beautiful mannequin sculpted out of amethyst by billelis + lit with 3 d geometric neon + facing a doorway opening with neon pink geometric fractal light + flowering hosta plants!!!, moon + city of los angeles in background!! dramatic, rule of thirds, award winning, 4 k, trending on artstation, photorealistic, volumetric lighting, octane render ",
"oil painting portrait of brock pierce, american flag on background, cowboy style. ",
"cursed images ",
"Shrek, horror, cursed images, spooky",
"intricate maze linework highly detailed optical illusion escher",
"dystopian cyberpunk mcdonald dictatorship",
"😏 Shrek",
"elon's musk ",
"facial portrait of putin is a monster, movie poster in the style of ( ( ( ( drew struzan ) ) ) ) ",
"president of belorussia, alexander lukashenko in style of sailor moon, anime, perfect faces, fine details",
"beautiful studio photograph of colorful postmodern portrait sculpture of adam sandler smiling, beautiful symmetrical face accurate face detailed face realistic proportions, made of watercolor - painted plaster on a pedestal by ron mueck and matthew barney and greg rutkowski, hysterical realism intense cinematic lighting shocking detail 8 k ",
"a portrait of an overwhelmed young man in a painting from stalenhag, 4 k, 8 k, hdr, artstation, concept art ",
"ben shapiro destroys the leftists with facts an logic "]

def test_normality_anderson(data, distribution = 'norm'):
    print('======================')
    result = anderson(data, dist=distribution)
    #result = result.significance_level, result.critical_values
    for i in range(len(result.critical_values)):
        sig_level, crit_val = result.significance_level[i], result.critical_values[i]
        if result.statistic < crit_val:
            print(f'Probability Gaussian : {crit_val} critical value at {sig_level} level of significance')
        else:
            print(f'Probability not Gaussian : {crit_val} critical value at {sig_level} level of significance')


def ks_test(data, distribution = 'norm'):
    dist = getattr(scipy.stats, distribution)

    # Fit a distribution to the data
    params = dist.fit(data)
    statistic, pvalue = kstest(data, distribution, params)
    print('statistic=%3f, p=%3f\n' %(statistic, pvalue))
    if pvalue > 0.05:
        print(f'{distribution}')
    else:
        print(f'not {distribution}')


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


def create_distribution_plots(data, prompt, dist_names, bins):
    # Distributions to check
    #dist_names = ['laplace', 'logistic', 'norm', 'gamma', 'hypsecant', 'genhyperbolic']

    for dist_name in dist_names:
        print(dist_name)
        # Plot the histogram
        plt.hist(data, bins=bins, density=True)
        plot_distribution(data, dist_name)

        title = 'Distribution: ' + dist_name
        plt.title(title)
        save_distr_plot(prompt, './output/distr_test', dist_name)


def test_distribution(data, prompt, distribution = 'norm', test = 'kstest', plot_distr = False):
    print(prompt)
    if test == 'anderson':
        test_normality_anderson(data)
    elif test == 'kstest':
        ks_test(data, distribution)
    print('')
    if plot_distr:
        plot_distribution(data, prompt)


def test_complete_embeddings():
    ldm = StableDiffusion()
    for prompt in prompts:
        print('=======================================')
        print(prompt)
        emb = ldm.get_embedding(prompt)[0]
        emb_flat = torch.flatten(emb)
        emb_flat = emb_flat.cpu().detach().numpy()
        create_distribution_plots(emb_flat, prompt, ['norm'], bins=40)
        # test_distribution(emb_fla, input, test = 'kstest')
        # test_distribution(emb_flat, input, 'genhyperbolic')
        # test_distributions(emb_flat, input, ['norm', 'logistic', 'genhyperbolic'])
        # fit_distribution(emb_flat)
        break
        print('')


def test_embedding_dimensions(prompts = None):
    if prompts is None:
        prompts = retrieve_prompts()
    ldm = StableDiffusion()
    for j in range(len(prompts)):
        prompt = prompts[j]
        emb_condition = ldm.get_embedding([prompt])[0][1]
        i = 0
        for emb in emb_condition:
            print(emb.shape)
            i += 1
            emb = emb.cpu().detach().numpy()
            create_distribution_plots(emb, f'{i}_{prompt}', ['norm'], bins=50)
        if j == 3:
            break

def q_q_plot(prompts = None):
    if prompts is None:
        prompts = retrieve_prompts()
    ldm = StableDiffusion()
    for j in range(len(prompts)):
        prompt = prompts[j]
        emb_condition = ldm.get_embedding([prompt])[0][1]
        i = 0
        if j == 1:
            break
        for emb in emb_condition:
            print(emb.shape)
            i += 1
            emb = emb.cpu().detach().numpy()
            #probplot(emb, dist="norm", plot=pylab)
            #emb = np.random.normal(loc=20, scale=5, size=emb.shape)
            ks_test(emb, distribution='genhyperbolic')
            #print(shapiro(emb))
            #sm.qqplot(emb, line='45')
            #pylab.savefig(f'{i}_{input}')



if __name__ == "__main__":
    #test_complete_embeddings()
    #test_embedding_dimensions(prompts)
    rand_prompts = create_random_prompts(5, numeric=True)
    #test_embedding_dimensions(rand_prompts)
    q_q_plot(prompts)

