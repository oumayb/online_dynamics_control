import torch
import os
import json
import numpy as np

from torchvision.io import read_video

import random
torch.manual_seed(0)
random.seed(0)


def get_video_names_pendulum(videos_path):
    """
    Params
    ------
    videos_path: path to folder that contains all the videos

    Returns
    -------
    videos_names: list of all available video names, sorted
    """
    video_names = []
    for video_name in os.listdir(videos_path):
        #if '.DS' not in video_name and "json" not in video_name:
        if 'mp4' in video_name:
            video_names.append(video_name)
    video_ids = [int(el.split('.mp4')[0]) for el in video_names]
    sorted_ids = sorted(video_ids)
    sorted_video_names = []
    for id in sorted_ids:
        sorted_video_names.append('{}.mp4'.format(id))
    return sorted_video_names


get_video_names = get_video_names_pendulum


class PendulumDataset(torch.utils.data.Dataset):
    """
    Pendulum video dataset class. Labels are omega0
    """

    def __init__(self, root_path, split, overfit, sample_duration=0, video_fmt='tensor', dataset_fraction=1, control=0, normalize=0, dtype='double'):
        """
        Params
        sample_duration: set to 0 if no temporal crop.
        dataset_fraction: float between 0 and 1
            whether to use the whole dataset or just a fraction of it
        control: whether it's a dataset with control or not
        ------
        root_path: path to your Pendulum folder
        """
        self.root_path = root_path
        self.videos_path = os.path.join(root_path, split)
        self.video_fmt = video_fmt
        self.sample_duration = sample_duration
        self.control = control
        self.dtype= dtype

        if self.video_fmt == 'npy':
            data_path = os.path.join(self.videos_path, split + '.npy')
            videos_sparse = np.load(data_path, allow_pickle=True)
            videos = np.zeros((videos_sparse.shape[0], videos_sparse[0].shape[0], videos_sparse[0].shape[1]))
            for i, spvideo in enumerate(videos_sparse):
                videos[i] = spvideo.todense()
            self.videos = torch.tensor(videos)

        elif self.video_fmt == 'video':
            try:
                self.labels_path = os.path.join(self.videos_path, "labels.json")
                with open(self.labels_path) as f:
                    self.labels = json.load(f)
            except:
                self.labels = None

            if overfit:
                self.videos_path = os.path.join(root_path, "train")
                self.labels_path = os.path.join(self.videos_path, "labels.json")
                try:
                    with open(self.labels_path) as f:
                        self.labels = json.load(f)
                except:
                    self.labels = None
                self.videos_names = [get_video_names(videos_path=self.videos_path)[overfit]]
            else:
                videos_names = get_video_names(videos_path=self.videos_path)
                dataset_size = int(dataset_fraction * len(videos_names))
                self.videos_names = videos_names[:dataset_size]
        self.videos_ids = []
        for name in self.videos_names:
            self.videos_ids.append(name.split('.mp4')[0])

        if self.control:
            u_dict_path = os.path.join(self.videos_path, 'u.json')
            with open(u_dict_path, 'r') as f:
                self.u = json.load(f)
            n_frames = len(self.u['1'].keys())
            d = len(self.u['1']['0'])
            self.U = torch.zeros((len(self.u), n_frames, d), dtype=torch.double)
            for key in self.u.keys():
                if key in self.videos_ids:
                    tensor = torch.zeros((n_frames, d), dtype=torch.double)
                    for key2 in self.u[key].keys():
                        tensor[int(key2), :] = torch.tensor(self.u[key][key2], dtype=torch.double)
                    self.U[int(key), :, :] = tensor

            if normalize:
                self.U_mean = self.U[:, 2:, :].mean()
                self.U_std = torch.std(self.U[:, 2:, :])
                self.U = (self.U - self.U_mean) / self.U_std

    def __getitem__(self, index):
        """
        Returns a dict with keys:
            - data: video clip tensor
            - label: tensor of size 1, omega0
            - name: video name
        """
        batch = dict()

        if self.video_fmt == 'npy':
            clip = self.videos[index].view((100, 1, 32, 32))
            batch['label'] = 0
            batch['name'] = 0

        elif self.video_fmt == 'video':
            if len(self.videos_names) == 1:
                video_name = self.videos_names[0]
            else:
                video_name = self.videos_names[index]
            video_path = os.path.join(self.videos_path, video_name)
            if self.video_fmt == 'tensor':
                clip = torch.load(video_path)
            elif self.video_fmt == 'video':
                clip, _, _ = read_video(video_path, pts_unit='sec')
                clip = clip.transpose(1, 3)
            try:
                label = self.labels[str(video_name.split(".")[0])]
                batch['label'] = torch.tensor(label['omega0']).view((1))  # .float()
            except:
                batch['label'] = 0
            batch['name'] = video_name

        if self.sample_duration:
            start = 0
            if self.dtype == 'float':
                batch['data'] = clip[start:start + self.sample_duration, :, :, :].float()
            elif self.dtype == 'double':
                batch['data'] = clip[start:start + self.sample_duration, :, :, :].double()
        else:
            if self.dtype == 'float':
                batch['data'] = clip.float()
            elif self.dtype == 'double':
                batch['data'] = clip.double()

        if self.video_fmt == 'video' and 'bw' in self.root_path:
            seq_len, c, h, w = batch['data'].shape
            batch['data'] = batch['data'][:, 0, :, :].reshape((seq_len, 1, h, w))

        if batch['data'].max() > 1:
            batch['data'] /= 255

        if self.control:
            if self.dtype == 'float':
                batch['control_inputs'] = self.U[index].float()
            elif self.dtype == 'double':
                batch['control_inputs'] = self.U[index]
            if self.sample_duration:
                batch['control_inputs'] = batch['control_inputs'][:self.sample_duration, :]
        return batch

    def __len__(self):
        if self.video_fmt == 'video':
            return len(self.videos_names)
        elif self.video_fmt == 'npy':
            return self.videos.shape[0]
