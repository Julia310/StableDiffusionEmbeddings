from ldm.stable_diffusion import StableDiffusion
from aesthetic_predictor.simple_inference import AestheticPredictor
from utils.file_utils import make_dir
from utils.create_graphics import create_boxplot
from utils.image_generation import get_random_seeds, create_random_prompts

prompts = ["Cute small humanoid bat sitting in a movie theater eating popcorn watching a movie ,unreal engine, cozy " \
         "indoor lighting, artstation, detailed, digital painting,cinematic,character design by mark ryden and pixar " \
         "and hayao miyazaki, unreal 5, daz, hyperrealistic, octane render",
         "goddess made of ice long hair like a waterfall, full body, horizontal symmetry!, elegant, intricate, highly detailed, fractal background, digital painting, artstation, concept art, wallpaper, smooth, sharp focus, illustration, epic light, art by kay nielsen and zeen chin and wadim kashin and sangyeob park, terada katsuya ",
         "a dad angry at missing his flight from prague to nyc, the dad is drunk ",
         "mandelbrot 3 d volume fractal mandala ceramic chakra digital color stylized an ancient white bone and emerald gemstone relic, intricate engraving concept substance patern texture natural color scheme, global illumination ray tracing hdr fanart arstation by sung choi and eric pfeiffer and gabriel garza and casper konefal ",
         "spongebob's house, 16 bit,",
         "oil painting portrait of brock pierce, american flag on background, cowboy style. ",
         "Oh no"]

prompts = create_random_prompts(num_prompts=5, numeric=True)

#input = "the funniest thing i've ever seen! "


def main():
    for prompt in prompts:
        make_dir(f'./output/test_random_seeds', seed=prompt)
        print(prompt)
        predictions = list()
        seeds = get_random_seeds(1000)
        ldm = StableDiffusion()
        aesthetic_predictor = AestheticPredictor()
        prompt_prediction = aesthetic_predictor.predict_aesthetic_score(prompt,  image_input=False)
        emb = ldm.get_embedding(prompts=[prompt])[0]

        for seed in seeds:

            pil_image = ldm.embedding_2_img('', emb, seed=seed, save_img=False)
            pil_image.save(f'./output/test_random_seeds/{prompt}/{seed}_{prompt[0:30]}.jpg')

            predictions.append(aesthetic_predictor.predict_aesthetic_score(pil_image))

        #predictions = [2.7579, 3.0262, 3.1552, 2.6313, 1.9623, 1.9561, 3.1155, 2.9933, 3.5519, 2.0996, 2.6772, 2.3749, 2.8698, 3.1294, 3.0717, 3.6689, 3.7291, 3.5370, 2.5392, 3.0721, 2.8832, 2.6850, 2.6214, 2.7346, 3.8023, 2.5648, 3.0840, 1.8821, 2.3215, 2.1453, 3.0764, 2.2827, 2.3014, 2.5039, 2.7697, 2.8723, 2.7706, 3.1646, 3.3184, 1.8050, 3.2939, 2.1338, 2.7333, 3.1716, 2.6060, 2.6157, 3.6159, 3.1380, 3.0540, 3.6607, 3.6036, 3.2287, 2.4645, 3.6949, 2.8153, 2.3470, 2.3317, 2.0081, 2.9648, 2.0548, 3.1215, 3.3046, 2.3954, 2.7253, 3.4986, 2.6768, 2.9258, 3.7847, 3.4793, 2.6175, 2.0190, 1.9342, 2.9116, 3.6741, 2.9763, 3.0836, 4.0210, 1.7083, 2.3960, 3.5096, 1.9407, 2.8768, 2.9361, 2.2768, 3.0657, 2.6002, 3.2901, 2.9420, 3.1907, 2.3172, 2.3367, 2.5102, 2.6241, 2.6622, 2.8866, 3.2390, 2.9549, 2.9021, 2.7264, 2.7014, 2.7176, 3.0738, 3.8977, 3.9006, 2.0003, 2.2085, 2.7453, 2.2050, 3.0393, 3.5184, 2.2346, 2.1165, 1.9690, 3.2777, 2.9416, 2.9037, 3.1233, 2.8840, 2.3336, 2.9203, 3.1291, 3.0340, 3.5773, 2.2242, 3.0351, 2.0342, 2.8082, 2.7988, 2.4956, 2.8071, 2.2536, 2.0366, 3.6515, 3.5432, 2.6107, 3.7434, 3.4001, 2.9940, 2.9796, 3.1948, 3.0700, 2.8653, 2.5284, 3.4099, 3.6783, 2.6989, 2.6439, 3.3717, 2.2155, 3.2902, 1.9492, 2.1854, 4.6568, 3.0136, 2.7138, 2.5838, 3.3163, 2.5999, 3.1624, 2.5757, 2.5670, 1.9600, 2.4132, 2.5260, 2.7154, 2.3739, 2.6028, 2.6074, 2.6606, 2.3849, 2.2802, 2.9495, 2.7437, 2.1217, 2.3980, 2.4608, 2.1067, 2.8733, 2.2119, 3.2108, 3.0968, 2.4068, 3.0048, 3.3656, 3.3899, 2.7428, 3.4499, 2.3222, 2.0508, 3.2572, 2.8067, 2.8209, 3.0035, 3.9765, 3.2875, 2.3131, 1.8893, 3.1060, 5.1460, 2.6969, 1.5213, 2.0634, 2.9547, 2.5758, 1.5943, 3.3500, 2.8950, 1.9924, 2.5176, 2.1909, 1.9422, 3.0844, 3.2129, 2.9367, 2.0504, 2.9284, 2.8812, 2.4619, 2.8719, 1.8647, 3.8293, 2.6593, 3.8303, 2.6788, 3.2969, 2.5296, 3.2077, 2.6282, 3.8196, 2.9936, 2.7029, 3.6807, 2.4278, 2.6080, 2.8720, 2.5645, 2.4601, 3.3359, 3.1926, 2.4915, 3.2764, 4.0878, 2.4435, 2.7010, 2.7721, 3.7447, 2.4284, 3.2826, 3.0884, 2.5832, 2.1562, 2.0795, 2.4122, 3.2940, 3.2172, 3.4615, 2.2722, 2.3855, 2.7077, 2.0385, 2.1017, 2.9567, 2.0133, 3.2842, 2.5076, 2.3097, 2.9646, 2.9636, 2.1678, 2.9721, 3.7336, 2.7577, 3.6931, 2.9210, 2.8819, 4.6698, 3.0241, 2.6720, 2.6236, 3.2539, 2.0009, 3.6004, 2.7127, 2.2493, 2.9699, 2.8033, 4.1615, 2.6413, 3.0459, 2.9896, 2.9200, 2.0604, 2.7739, 3.8385, 2.9101, 2.8435, 2.7595, 2.2878, 2.9797, 3.0503, 2.7121, 2.6053, 3.5479, 2.6239, 2.5233, 2.0386, 2.9609, 2.5998, 3.2728, 3.5218, 3.5165, 3.1116, 2.3101, 3.7083, 2.9865, 3.4704, 2.4423, 4.4535, 2.5253, 2.1975, 3.1548, 2.1762, 4.2186, 2.8292, 2.9168, 3.1888, 2.2119, 3.3416, 3.3430, 2.5504, 3.4626, 2.2367, 4.0104, 2.6543, 3.9966, 3.3257, 2.9727, 3.5121, 2.4202, 2.9380, 3.2920, 2.8847, 2.9959, 2.7158, 2.7020, 3.2495, 2.4370, 3.2662, 2.4135, 1.9506, 1.8348, 2.4053, 3.3758, 3.1046, 3.0507, 3.4278, 3.1976, 2.4873, 2.1220, 3.9420, 3.1975, 2.5620, 4.9023, 3.2550, 2.4538, 2.5729, 3.2368, 2.8314, 2.8659, 3.5448, 2.0543, 2.2194, 2.6999, 3.3271, 2.9845, 2.1320, 2.9868, 3.1230, 3.7898, 2.9083, 2.6073, 3.4213, 3.1325, 1.6790, 3.1183, 2.0231, 3.4107, 2.8122, 3.1843, 3.3937, 2.9679, 2.8650, 2.1276, 2.2062, 3.1168, 3.4896, 3.0591, 2.4878, 3.4432, 2.6356, 3.6048, 2.9894, 2.7842, 3.0065, 2.2606, 2.9561, 2.7638, 2.9166, 2.8898, 2.6085, 2.2624, 3.7834, 2.9643, 3.9527, 4.6434, 2.4376, 3.9324, 2.6847, 2.9164, 2.3755, 2.6146, 2.4195, 3.1448, 3.6877, 2.9301, 2.6721, 3.2990, 3.0577, 4.0321, 2.2728, 2.7216, 2.9005, 2.7313, 3.2549, 2.3585, 2.7086, 3.4387, 3.5531, 2.7963, 2.9762, 2.7681, 3.0050, 2.4943, 3.0497, 3.3247, 3.5100, 2.5785, 3.9697, 2.9744, 3.9246, 2.4185, 3.3769, 2.5715, 4.4710, 2.9040, 3.1856, 3.5155, 2.3003, 2.4122, 2.7816, 3.4287, 2.7303, 4.2577, 3.0754, 4.4129, 3.0106, 4.3880, 3.3771, 2.1782, 2.3442, 2.0184, 3.3948, 2.2877, 3.0337, 2.8091, 2.2914, 4.0957, 2.7737, 3.3927, 2.6631, 1.9769, 3.3914, 2.9529, 3.2774, 3.0239, 2.5834, 2.5356, 2.8609, 3.2437, 3.8536, 2.1826, 2.6696, 3.0460, 2.5887, 2.9627, 2.8685, 3.0597, 3.0520, 2.4542, 2.6985, 3.1226, 3.2447, 3.9165, 2.4869, 3.4395, 2.0215, 2.5600, 2.9731, 2.0716, 2.9478, 2.0379, 3.3095, 2.9918, 3.1312, 1.9478, 2.5536, 2.6506, 2.7126, 3.4129, 2.8062, 2.9764, 2.0872, 2.7841, 3.1969, 2.8296, 2.4589, 2.4669, 3.0978, 3.0488, 3.2009, 2.0430, 3.7970, 2.6992, 3.5287, 3.4597, 3.8410, 2.2019, 2.8907, 2.8503, 1.8378, 2.5895, 2.2860, 3.1774, 3.0618, 2.9881, 2.0243, 2.6158, 3.3346, 2.4786, 2.9024, 2.7568, 2.3478, 1.7237, 2.7504, 3.4279, 3.5053, 2.0675, 3.3004, 2.8461, 2.2390, 2.3879, 2.7528, 4.0064, 3.1228, 3.1950, 3.3327, 2.6922, 2.7992, 2.3795, 2.6349, 2.0244, 2.2921, 3.4796, 3.0454, 3.2474, 2.5382, 2.7850, 2.0989, 3.2404, 2.4910, 2.6191, 2.4288, 2.2933, 2.8812, 3.2011, 2.7485, 3.9556, 3.4459, 2.2298, 2.4664, 3.2197, 2.4485, 2.3830, 3.2719, 3.2255, 1.8289, 3.1866, 2.4125, 2.7682, 3.8511, 3.3559, 3.6546, 2.1843, 3.0561, 2.4319, 1.8351, 1.9734, 1.9087, 2.4293, 3.5600, 1.9756, 1.6537, 3.3128, 2.9686, 3.8974, 2.9521, 2.4567, 3.0088, 2.0501, 4.5725, 3.7301, 2.9177, 2.8012, 2.7154, 2.9391, 3.1394, 3.5029, 1.7538, 2.4454, 2.4606, 3.4812, 2.6001, 3.3452, 2.5061, 2.6102, 2.6595, 2.8169, 2.7658, 3.2252, 3.5397, 3.1035, 1.9031, 2.5651, 3.3948, 2.7891, 2.4539, 3.8289, 2.6829, 3.1151, 1.7908, 2.3616, 2.4888, 2.6758, 3.9179, 2.7769, 2.5867, 3.1426, 3.3537, 2.2088, 2.4617, 2.7081, 3.5381, 2.6243, 3.8106, 2.1573, 2.6023, 1.8858, 3.0282, 2.5049, 4.3925, 2.4983, 2.7065, 1.6631, 2.9220, 2.3255, 2.4826, 2.3401, 3.0093, 2.8433, 2.8971, 2.0420, 3.6939, 3.1619, 3.7810, 2.1489, 2.8972, 3.4949, 3.0421, 2.8526, 3.9260, 2.7746, 3.5411, 2.5597, 3.1495, 3.5224, 2.8327, 3.6069, 3.3403, 3.4675, 2.8967, 3.8675, 2.4301, 2.2342, 2.8384, 2.6547, 2.1121, 2.3775, 2.8648, 2.2161, 2.4822, 3.7013, 2.6551, 3.5288, 2.3585, 2.1062, 3.4004, 3.5763, 2.3187, 3.0762, 2.6775, 3.0841, 3.3370, 3.4360, 2.8272, 3.0362, 3.1315, 2.9041, 3.6783, 2.1864, 3.4303, 2.8353, 2.3201, 3.3977, 2.6446, 3.0285, 4.1342, 3.1659, 3.0027, 3.6317, 3.0832, 2.4555, 2.2032, 4.0132, 3.5284, 2.1133, 3.9192, 3.2647, 3.2463, 2.6962, 2.6892, 3.0683, 3.1859, 2.9798, 2.3257, 2.8627, 2.9560, 1.9823, 3.0074, 4.5201, 2.4904, 2.8052, 2.2309, 2.2404, 3.2818, 2.5513, 2.9619, 4.5530, 2.6371, 3.6893, 2.6270, 2.3594, 2.4878, 2.7213, 3.3384, 2.7644, 2.3735, 2.2372, 2.0137, 2.8124, 2.4682, 2.6746, 2.8879, 2.8271, 2.8208, 2.5514, 3.8721, 2.2720, 1.8109, 2.9037, 3.1558, 2.1343, 2.7856, 3.8934, 2.4118, 2.7505, 2.0770, 2.3301, 4.4664, 2.7284, 2.3062, 2.4777, 2.5148, 2.4192, 3.1254, 3.0845, 2.1164, 2.8718, 1.7074, 3.2012, 2.2906, 3.4081, 3.2128, 3.3788, 3.0734, 2.4062, 3.1021, 3.6890, 3.0080, 3.2003, 2.7798, 3.0957, 2.2778, 3.4782, 3.7657, 2.4688, 3.7254, 2.6983, 1.7938, 3.6039, 2.7634, 2.4663, 1.8379, 2.5273, 3.0339, 3.3691, 3.4489, 2.1960, 2.5677, 2.2724, 3.6153, 2.6256, 3.7052, 3.0220, 1.8399, 3.2226, 3.4779, 2.9203, 2.5578, 1.8869, 3.7279, 2.4947, 2.4910, 2.3954, 3.1436, 2.0012, 2.7534, 4.2488, 3.1475, 1.6177, 2.5622, 2.6824, 3.6498, 2.7130, 1.6903, 2.5847, 2.1042, 3.0015, 2.2300, 2.6378, 2.2978, 3.0801, 3.3312, 1.9483, 2.1915, 2.8222, 2.8466, 2.2586, 2.5791, 3.3420, 3.2758, 3.8371, 1.9104, 2.8298, 2.1490, 2.3498, 4.2563, 3.4925, 2.6884, 3.1576, 2.2097, 3.1960, 2.2744, 2.2839, 2.1000, 2.1568, 3.3221, 3.0855, 3.2759, 2.7298, 2.2918, 2.3433, 2.5561, 2.4387, 3.5951, 2.5296, 2.8141, 2.5495, 3.2444, 3.5651, 3.1566, 3.0205, 3.8998, 3.1721, 3.1643, 2.9547, 2.4348, 1.8974, 2.8430, 2.4813, 2.8601, 1.9402, 2.4824, 3.5405, 3.4701, 3.5680, 3.0563, 3.1869, 2.5698, 3.1469, 3.6047, 3.3745, 2.1803, 2.6064, 3.3454, 2.1353, 2.5879, 2.4304, 2.6855, 3.0365, 3.5929, 2.4686, 3.0055, 3.1972, 4.4444, 3.0834, 3.0010, 2.0371, 3.1416, 3.7176, 2.9135, 2.7579, 2.7449, 3.6906, 3.5330, 3.3493, 3.2312, 2.9039, 2.2252, 3.3660, 3.5720, 2.0176, 3.6341, 2.6048, 3.4173, 2.7617, 2.3851, 2.2580, 3.3714, 3.2538, 3.2436, 3.3095, 2.2564, 3.5104, 3.4151, 3.1243, 3.2381, 2.2956, 1.9864, 2.3197, 2.0819, 2.5092, 3.0623, 2.0791, 2.9426, 3.3970, 2.5398, 2.9621, 2.7496, 2.2691, 2.1387, 2.5057, 2.5642, 2.3819, 3.5551, 3.7628, 2.2128, 4.5131, 3.2261, 2.6412, 2.8372, 2.7573, 1.8621, 3.1105, 3.1922, 4.1229, 3.1429, 2.8885, 3.7930, 3.0424, 2.2075, 3.4188, 3.6321, 3.2994, 3.1127, 2.5021, 2.3387, 2.7928, 2.4767, 3.0550, 1.8612, 2.4944, 3.5555, 3.8146, 2.7956, 3.1999, 3.0856, 1.8274, 2.6650, 2.3381, 3.2589, 2.8313, 3.3729, 2.2134, 3.8045, 2.8299, 2.4615, 2.0800, 4.5569, 2.7151, 3.1943, 2.2671, 3.1336, 1.6482, 3.6995, 3.7054, 3.8238, 3.6507, 3.2719, 2.9736, 2.2441, 3.4423, 3.0113, 3.0981, 2.8040, 2.5436, 3.3704, 3.0764, 2.6593, 1.7822, 3.6203, 3.9693, 3.3686, 3.3014, 3.3707, 2.4841, 3.1733, 2.5550, 3.3531, 3.0690, 3.3918, 3.2928, 3.1856, 3.0079, 3.1860, 3.3062, 4.0056, 2.6974, 3.2967, 2.4452, 2.4329, 3.5951, 2.0902, 2.7383, 3.3608, 3.6574, 2.3460, 3.0799, 2.8839, 3.2842, 2.3913, 3.6705, 2.4782, 3.8595, 3.2580, 2.6025, 3.4869, 2.7895, 1.9675, 2.3577, 2.6052, 3.2933, 3.8130, 2.3707, 2.9388, 2.3357, 2.0659, 3.1664, 3.4805, 3.2874, 2.5141, 2.7074, 3.8440, 3.1387, 3.1862, 2.5733, 3.2315, 2.6875, 4.4362, 3.6799, 2.6259, 2.6181, 3.4602, 2.8921, 4.1953, 2.9197, 2.2947, 3.9856, 2.8062, 2.8500, 2.4769, 3.5269, 3.4064, 3.3576, 2.9377, 3.3746, 2.8834, 4.0322, 3.0480, 2.7820, 4.0163, 3.0941, 3.7077, 2.9323, 2.1620, 4.1868, 3.4390, 2.5191, 2.6714, 3.9228, 5.0064, 2.3832, 2.2773, 3.2318, 2.3376, 3.1516, 2.1466, 3.3954, 2.7402, 3.7917, 2.5691, 2.0819, 3.1218, 3.1891, 2.6455, 3.1537, 2.7590, 3.3509, 2.5025, 4.7900, 3.0486, 1.8227, 2.8802, 2.0018, 2.9322, 2.4293, 2.9513, 2.9559, 2.1098, 3.3343, 2.5346, 2.2803, 3.1586, 3.4203, 2.9490, 2.4731, 3.2889, 2.4621, 2.1003, 3.3784, 2.4331, 3.7243, 2.2906, 2.1440, 3.1588, 2.4023, 3.1091, 3.5330, 3.1445, 2.8006, 3.3575, 3.4530, 3.5282, 3.7270, 2.8701, 2.4010, 2.9704, 2.6843, 3.5694, 2.7760, 2.7149, 3.6093, 3.6219, 2.6731, 2.4929, 2.7865, 2.6414, 3.1317, 3.6736, 2.8229, 2.5901, 2.4970, 2.2773, 3.1982, 2.2347, 2.3340, 2.6803, 2.6328, 2.2966, 2.1672, 2.6968, 2.0393, 3.0153, 2.7716, 2.8629, 3.3946, 3.2348, 3.8896, 2.9536, 2.9652, 2.9264, 2.5151, 2.5102, 2.9612, 2.2430, 2.8030, 2.8134, 3.1568, 3.0925, 2.8560, 2.7992, 3.5046, 2.5501, 2.8425, 2.8323, 3.3704, 3.3358, 2.6351, 2.3480, 1.5719, 3.2685, 2.0490, 4.6032, 3.8024, 3.3341, 1.9500, 2.9653, 2.5972, 3.1800, 3.0383, 2.8256, 2.3660, 3.7613, 3.0386, 3.0112, 2.8390, 3.0419, 2.6421, 2.9080, 3.2152, 2.5016, 3.0188, 2.7963, 2.6093, 2.0271, 2.9140, 2.4821, 2.9344, 2.7376, 3.4639, 4.0225, 2.5822, 3.2194, 2.6223, 2.6171, 2.7254, 3.6255, 2.0555, 3.2290, 3.4333, 2.0549, 3.7444, 2.6176, 2.8634, 2.8559, 2.2504, 2.8904, 4.0440, 2.8700, 2.8488, 3.4048, 2.5280, 3.5532, 2.5402, 3.7132, 4.3107, 1.9901, 4.4374, 2.3338, 3.2333, 3.6112, 3.1994, 3.2026, 2.5790, 2.9640, 3.3234, 3.3016, 3.3286, 2.9749, 3.1420, 2.5568, 2.4155, 2.5923, 2.2556, 3.3511, 3.2957, 3.6216, 3.2658, 2.4546, 3.9118, 2.5143, 2.7573, 2.5193, 2.9610, 3.7553, 2.8928, 2.5196, 3.1085, 3.1474, 3.5703, 1.7869, 2.2039, 2.7654, 2.9342, 2.2947, 3.0739, 3.6330, 2.6656, 2.3159, 2.8932, 3.0517, 2.3860, 2.3936, 2.3398, 2.8346, 2.1105, 3.3666, 3.1747, 3.5873, 2.4891, 3.4592, 2.5061, 2.7141, 3.3112, 2.5721, 2.4277, 3.0924, 3.4332, 3.1472, 3.7132, 2.1045, 2.1837, 3.7117, 2.9502, 2.6465, 1.7894, 2.0627, 3.6792, 3.4082, 2.8182, 2.9235, 3.3944, 2.3999, 3.2696, 2.6214, 3.0002, 2.3344, 1.7578, 3.0873, 2.9820, 3.7015, 3.5135, 2.7838, 2.4623, 2.5338, 2.7570, 3.1392, 1.2938, 2.4871, 3.1283, 1.7044, 2.9139, 3.2055, 2.6691, 3.4272, 2.2594, 2.9228, 2.3473, 3.1045, 2.6876, 1.7942, 3.3343, 2.3796, 3.6413, 2.5155, 3.0789, 2.8411, 1.8808, 2.3476, 3.1161, 3.2990, 2.9050, 3.3146, 3.1465, 2.2628, 2.2099, 2.8043, 2.5417, 2.8797, 2.3357, 2.4727, 2.5434, 1.6485, 3.2095, 3.0825, 2.7516, 3.4290, 2.7699, 3.1176, 2.4808, 2.7503, 2.3797, 3.2928, 2.5294, 2.9351, 3.2410, 2.6100, 2.6595, 3.4928, 2.8310, 2.8849, 3.5189, 3.0785, 3.4260, 2.3882, 3.1813, 3.5799, 2.0209, 4.6452, 2.9055, 2.3752, 2.6327, 4.0700, 2.3596, 3.6134, 3.0796, 4.2198, 2.6143, 2.8075, 3.4892, 2.6821, 3.0371, 2.2432, 3.0343, 2.6973, 2.0375, 3.4950, 3.0537, 1.9244, 3.2260, 2.6967, 2.8092, 3.0040, 3.6189, 2.8144, 3.4013, 1.9345, 3.2387, 3.4531, 3.5891, 3.0287]

        create_boxplot(predictions, prompt_prediction, filename=f"{prompt[0:30]}_boxplot.png")


if __name__ == "__main__":
    main()
