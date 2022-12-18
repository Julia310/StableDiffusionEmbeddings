from stable_diffusion import SrtableDiffusion


def main():
    noise = 0.5
    prompts = ['full body image of a norwegian forest cat of white and ginger fur, by dan mumford, yusuke murata and '
               'makoto shinkai, 8k, cel shaded, unreal engine, featured on artstation, pixiv',
               'fluffy dog']

    prompts = [
        'futuristic spiderman in style of wolf ,highly detailed, 4k, HDR, award-winning, artstation, octane render',
        'happy pepe, artstation, marvel, anfas, futuristic, cyberpunk hyper detailed, transformers ']

    prompts = [
        'khajit tabaxi catfolk humanoid with maine coon features with an eye patch on the left eye and black fur '
        'holding two shortswords cloaked in shadow and wearing hooded leather armor toned muscle, dungeons and dragons, '
        'pure white background, fantasy, tarot card style, half body portrait, high detail, hyper realistic ',
        'profile of a tan maine coon bold natural colors masterpiece trending on artstation photograph ']

    prompts = [
        'walter white is sherlock holmes, high quality illustration, trending on artstation, octane render, 4 k, pixar rendering, ',
        'cinematic still of will smith in Blade (2001), XF IQ4, f/1.4, ISO 200, 1/160s, 8K, RAW, dramatic lighting, '
        'heisymmetrical balance, in-frame, highly accurate facial features']

    prompts = ['a beautiful painting of a peaceful lake in the Land of the Dreams, full of grass, sunset, red horizon, '
               'starry-night!!!!!!!!!!!!!!!!!!!!,  Greg Rutkowski, Moebius, Mohrbacher, peaceful, colorful',
               "epic landscape with a lake, golden hour, misty ground, rocky ground, distant mountains, hazy, foggy, "
               "atmospheric perspective, altostratus clouds,, cinematic, 1000mm lens, anamorphic lens flare, photographic, "
               "octane render, cinematography by roger deakins, in the style of ansel adams, low details,"]

    prompts = [
        'a painting of a tree with pink flowers, a detailed matte painting by shin yun - bok, cgsociety, photorealism, '
        'matte painting, detailed painting, matte drawing ',
        "Beautiful hyperrealistic detailed matte painting of a landscape of a landscape of wheat fields on which "
        "stands a large wooden toilet with a heart cut out of lost vibes in the foreground and a dark forest in the "
        "background during autumn, afternoon, winter, by andreas rocha and john howe, and Martin Johnson Heade, "
        "featured on artstation, featured on behance, golden ratio, ultrawide angle, f32, well composed, cohesive"]

    prompts = ["beautiful mountain landscape, lake, snow, oil painting 8 k hd ",
               "a beautiful and highly detailed matte painting of the epic mountains of avalon, intricate details, epic scale, "
               "insanely complex, 8 k, sharp focus, hyperrealism, very realistic, by caspar friedrich, albert bierstadt, james gurney, brian froud, "]

    # Converting textual prompts to embedding

    ldm = SrtableDiffusion()
    emb_list = ldm.get_embedding(prompts)
    emb = ldm.combine_embeddings(emb_list[0], emb_list[1], noise)
    emb_list.append(emb)
    prompts.append(f'{prompts[0][0:20]}_{prompts[1][0:20]}_{noise}')

    for i in range(len(prompts)):
        ldm.embedding_2_img(prompts[i], emb_list[i])


if __name__ == "__main__":
    main()
