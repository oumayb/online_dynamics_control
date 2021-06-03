import os
from tqdm import tqdm
import moviepy.editor as mp

import json

import argparse
parser = argparse.ArgumentParser(description='Resizing')
parser.add_argument('--source_dir', default=None, type=str, help='Source directory of videos to resize')
parser.add_argument('--target_dir', default=None, type=str, help='Target directory of resized videos')
parser.add_argument('--size', type=int, help='size to resize smallest dim to')

args = parser.parse_args()

SOURCE_DIR = args.source_dir
TARGET_DIR = args.target_dir

if not os.path.isdir(TARGET_DIR):
    os.mkdir(TARGET_DIR)
os.mkdir(os.path.join(TARGET_DIR, 'train'))
os.mkdir(os.path.join(TARGET_DIR, 'test'))
if 'eval' in os.listdir(SOURCE_DIR):
    os.mkdir(os.path.join(TARGET_DIR, 'eval'))

## Resize train videos
train_path = os.path.join(SOURCE_DIR, 'train')
target_train_path = os.path.join(TARGET_DIR, 'train')
for video_name in tqdm(os.listdir(train_path)):
    if 'mp4' in video_name:
        video_path = os.path.join(train_path, video_name)
        #print(video_path)
        clip = mp.VideoFileClip(video_path)
        clip_resized = clip.resize(newsize=(args.size, args.size))
        target_path = os.path.join(target_train_path, video_name)
        clip_resized.write_videofile(target_path, audio=False)

## Resize test videos
test_path = os.path.join(SOURCE_DIR, 'test')
target_test_path = os.path.join(TARGET_DIR, 'test')
for video_name in tqdm(os.listdir(test_path)):
    if 'mp4' in video_name:
        video_path = os.path.join(test_path, video_name)
        clip = mp.VideoFileClip(video_path)
        clip_resized = clip.resize(newsize=(args.size, args.size))
        target_path = os.path.join(target_test_path, video_name)
        clip_resized.write_videofile(target_path, audio=False)

if 'eval' in os.listdir(SOURCE_DIR):
    ## Resize eval videos
    eval_path = os.path.join(SOURCE_DIR, 'eval')
    target_eval_path = os.path.join(TARGET_DIR, 'eval')
    for video_name in tqdm(os.listdir(eval_path)):
        if 'mp4' in video_name:
            video_path = os.path.join(eval_path, video_name)
            clip = mp.VideoFileClip(video_path)
            clip_resized = clip.resize(newsize=(args.size, args.size))
            target_path = os.path.join(target_eval_path, video_name)
            clip_resized.write_videofile(target_path, audio=False)
