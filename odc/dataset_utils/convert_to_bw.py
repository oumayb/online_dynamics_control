import os
import moviepy.editor as mp

from tqdm import tqdm

import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Resizing')
parser.add_argument('--source_dir', default=None, type=str, help='Source directory of videos to convert')
parser.add_argument('--target_dir', default=None, type=str, help='Target directory of converted videos')
parser.add_argument('--out_channels', default=1, type=int, help='Wether to return 1 or 3 channels img')
args = parser.parse_args()

SOURCE_DIR = args.source_dir
TARGET_DIR = args.target_dir


def blackwhite(clip, RGB = None, preserve_luminosity=True, out_channels=1):
    """ Desaturates the picture, makes it black and white.
    Parameter RGB allows to set weights for the different color
    channels.
    If RBG is 'CRT_phosphor' a special set of values is used.
    preserve_luminosity maintains the sum of RGB to 1."""
    if RGB is None:
        RGB = [1,1,1]

    if RGB == 'CRT_phosphor':
        RGB = [0.2125, 0.7154, 0.0721]

    R,G,B = 1.0*np.array(RGB)/ (sum(RGB) if preserve_luminosity else 1)

    def fl(im):
        im = (R*im[:,:,0] + G*im[:,:,1] + B*im[:,:,2])
        if out_channels == 3:
            return np.dstack(3*[im]).astype('uint8')
        elif out_channels == 1:
            print('ou_channels should be set to 3')
            raise NotImplementedError
            #return im.astype('uint8')
            #return np.dstack([im]).astype('uint8')

    return clip.fl_image(fl)

if not os.path.isdir(TARGET_DIR):
    os.mkdir(TARGET_DIR)

for video_name in tqdm(os.listdir(SOURCE_DIR)):
    if 'mp4' in video_name:
        video_path = os.path.join(SOURCE_DIR, video_name)
        clip = mp.VideoFileClip(video_path)
        bw_clip = blackwhite(clip, out_channels=args.out_channels)
        target_path = os.path.join(TARGET_DIR, video_name)
        bw_clip.write_videofile(target_path, audio=False)
