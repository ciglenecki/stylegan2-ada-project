import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
sys.path.append('../nvidia-stylegan2-ada')
from dnnlib import *
from torch_utils import *
sys.path.append('../src')
from interpolate_utils import *

# if you reposition to nvidia/stylegan2-ada
# eg.
# python ffhq-align.py
# python ../nvidia-stylegan2-ada/projector.py --outdir=../data/outputs/img1 --target=../data/processed/align-gen1.jpg --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
# python ../nvidia-stylegan2-ada/projector.py --outdir=../data/outputs/img2 --target=../data/processed/align-gen2.jpg --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
# python ../nvidia-stylegan2-ada/projector.py --outdir=../data/outputs/img3 --target=../data/processed/align-kostanjcar.jpg --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# ---------------------------------------

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_latent_list', nargs='+', default=['../data/outputs/img1/projected_w_303.npz', '../data/outputs/img2/projected_w_303.npz', '../data/outputs/img3/projected_w_303.npz'])
    parser.add_argument('--weights', nargs='+', default=[0.33, 0.33, 0.34])
    parser.add_argument('--noise_mode', type=str, default='const', help='Noise mode') #'const', 'random', 'none'
    parser.add_argument('--network_pkl', type=str, default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl', help='The URL to the network .pkl file')
    args = parser.parse_args()
    return args
    
def main():
    
    args = parse()

    ws = [np.load(latent_i)['w'] for latent_i in args.img_latent_list]
    w_weighted_avg = torch.from_numpy(np.average(ws, weights=args.weights, axis=0)).cuda()

    G = load_network(args.network_pkl)

    mixed_img = calculate_image(G, w_weighted_avg, args.noise_mode)
    _, ax = setup_figure()
    ax.imshow(mixed_img)
    plt.show()

main()