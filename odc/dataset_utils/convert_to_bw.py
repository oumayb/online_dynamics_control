import os
import moviepy.editor as mp
from mltools.utils.videoutils import blackwhite

from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Resizing')
parser.add_argument('--source_dir', default=None, type=str, help='Source directory of videos to convert')
parser.add_argument('--target_dir', default=None, type=str, help='Target directory of converted videos')
parser.add_argument('--out_channels', default=1, type=int, help='Wether to return 1 or 3 channels img')
args = parser.parse_args()

SOURCE_DIR = args.source_dir
TARGET_DIR = args.target_dir

if not os.path.isdir(TARGET_DIR):
    os.mkdir(TARGET_DIR)

for video_name in tqdm(os.listdir(SOURCE_DIR)):
    if 'mp4' in video_name:
        video_path = os.path.join(SOURCE_DIR, video_name)
        clip = mp.VideoFileClip(video_path)
        bw_clip = blackwhite(clip, out_channels=args.out_channels)
        target_path = os.path.join(TARGET_DIR, video_name)
        bw_clip.write_videofile(target_path, audio=False)
