import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

# ---------------------------------------

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_latent', type=str, default='../data/outputs/img1/projected_w_303.npz' , help='The path to the face image')
    parser.add_argument('--feature_num', type=int, default=0, help='Number between 0 and 511, both ends included')
    parser.add_argument('--noise_mode', type=str, default='const', help='Noise mode') #'const', 'random', 'none'
    parser.add_argument('--slider_step', type=float, default=0.01, help='Size of the step for the slider')
    parser.add_argument('--network_pkl', type=str, default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl', help='The URL to the network .pkl file')
    args = parser.parse_args()
    return args

def main():
    args = parse()

    w = torch.from_numpy(np.load(args.img_latent)['w']).cuda()

    G = load_network(args.network_pkl)

    all_imgs = calculate_all_images2(
        G, 
        args.slider_step, 
        args.noise_mode, 
        w,
        args.feature_num, 
    )

    fig, ax = setup_figure()

    im = ax.imshow(all_imgs[0])
    image2_slider = Slider(plt.axes([0.15, 0.05, 0.75, 0.03]), 'w value', 0, 1.0, valinit=0, valstep=args.slider_step)

    def update(val):
        img2_percentage = image2_slider.val
        step = int(img2_percentage * int(1 / args.slider_step))
        new_img = all_imgs[step]
        im.set_array(new_img)
        fig.canvas.draw_idle()

    image2_slider.on_changed(update)
    plt.show()

main()