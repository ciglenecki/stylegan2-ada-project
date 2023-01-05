import torch
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import sys
sys.path.append('../nvidia-stylegan2-ada')
from dnnlib import *
from legacy import load_network_pkl
from torch_utils import *
sys.path.append('../src')

from tqdm import tqdm
import matplotlib as mpl
from interpolate_utils import *

# if you reposition to src/
# eg.
# python ffhq-align.py
# python ../nvidia-stylegan2-ada/projector.py --outdir=../data/outputs/img1 --target=../data/processed/align-gen1.jpg --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
# python ../nvidia-stylegan2-ada/projector.py --outdir=../data/outputs/img2 --target=../data/processed/align-gen2.jpg --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

# ---------------------------------------


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1_latent', type=str, default='../data/outputs/img1/projected_w_303.npz' , help='The path to the first face image')
    parser.add_argument('--img2_latent', type=str, default='../data/outputs/img2/projected_w_303.npz', help='The path to the second face image')
    parser.add_argument('--noise_mode', type=str, default='const', help='Noise mode') #'const', 'random', 'none'
    parser.add_argument('--slider_step', type=float, default=0.01, help='Size of the step for the slider')
    parser.add_argument('--network_pkl', type=str, default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl', help='The URL to the network .pkl file')
    args = parser.parse_args()
    return args
    
def main():

    args = parse()

    w1 = torch.from_numpy(np.load(args.img1_latent)['w']).cuda()
    w2 = torch.from_numpy(np.load(args.img2_latent)['w']).cuda()

    G = load_network(args.network_pkl)

    all_imgs = calculate_all_images(G, args.slider_step, args.noise_mode, w1, w2)

    fig, ax = setup_figure()

    im = ax.imshow(all_imgs[0])
    image2_slider = Slider(plt.axes([0.15, 0.05, 0.75, 0.03]), 'Image 1 %', 0, 1.0, valinit=0, valstep=args.slider_step)

    def update(val):
        img2_percentage = image2_slider.val
        step = int(img2_percentage * int(1 / args.slider_step))
        new_img = all_imgs[step]
        im.set_array(new_img)
        fig.canvas.draw_idle()

    image2_slider.on_changed(update)
    plt.show()

main()