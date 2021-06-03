import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from odc.train_utils import get_loss, get_autoencoder
import os

from tqdm import tqdm

import random
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from odc.models import DMDAE
from odc.pendulum_dataset import PendulumDataset
from odc.metrics import RFMetrics, DAEGradients, RecoMetrics
import torch.nn as nn
import time
import json

from odc.model_utils import build_shifted_matrices, get_time

from abc import abstractmethod


import numpy as np

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

torch.backends.cudnn.deterministic=True


class Trainer:
    def __init__(self, config, gpu):
        self.config = config

        self.train_dataset, self.val_dataset, self.train_sampler = None, None, None
        self.train_loader, self.val_loader, self.val_sampler = None, None, None
        self.eval_dataset, self.eval_loader, self.eval_sampler = None, None, None
        self.create_data()

        self.model = None
        self.build_model()
        if self.config["resume_path"]:
            self.load_model()

        self.save_dir = None
        self.set_save_dir()
        self.writer = None
        if (self.config["dist"] and gpu != 0):
            time.sleep(5)
        if (self.config["dist"] and gpu == 0) or (not self.config["dist"]):
            print(self.config)
            if self.config['save']:
                self.create_save_dir()
                with open(os.path.join(self.save_dir, "config.json"), "w") as f:
                    json.dump(self.config, f)

                self.create_writer()

        if self.config['save_weights']:
            self.save_weights(gpu)

        self.optimiser = None
        self.create_optimiser()

        self.scheduler = None
        self.create_scheduler()

        self.loss_fn = None
        self.create_loss()

        self.train_metrics = None
        self.val_metrics = None
        self.eval_metrics = None

        self.best_losses_dict = None
        self.create_best_losses_dict()

    @abstractmethod
    def create_data(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def forward_model(self, batch):
        pass

    @abstractmethod
    def create_optimiser(self):
        pass

    @abstractmethod
    def create_scheduler(self):
        pass

    @abstractmethod
    def create_loss(self):
        pass

    @abstractmethod
    def forward_loss(self, batch, output):
        pass

    @abstractmethod
    def create_metrics(self, name):
        pass

    @abstractmethod
    def create_eval_metrics(self, name, config=None):
        pass

    @abstractmethod
    def create_gradients(self, name):
        pass

    @abstractmethod
    def return_gradients(self):
        pass

    @abstractmethod
    def create_writer(self):
        pass

    @abstractmethod
    def save_scalars(self):
        pass

    @abstractmethod
    def create_save_dir(self):
        pass

    @abstractmethod
    def set_save_dir(self):
        pass

    def create_losses_dict(self, metrics):
        """
        return dict where key='loss'. It is the total loss (that is optimized for)
        When overriding this method, one can add the other losses to track, e.g. 'res_loss', 'reco_loss', ..
        """
        losses = dict()
        losses['loss'] = metrics.avg_loss  # Total loss (that we optimize for)
        return losses

    def create_best_losses_dict(self):
        """
        Returns
        --------
        None

        Creates dict with lowest values of losses we're interested in
        When overriding this method, one can add the other losses to track, e.g. 'res_loss', 'reco_loss', ..
        """
        self.best_losses_dict = dict()
        self.best_losses_dict["loss"] = 1e10

    def update_best_losses_dict(self, losses):
        """
        Returns
        -------
        None

        Updates self.best_losses_dict with lowest values of the tracked losses.
        """
        for key in self.best_losses_dict.keys():
            if self.best_losses_dict[key] > losses[key]:
                self.best_losses_dict[key] = losses[key]

    def save_weights(self, gpu):
        weights_path = os.path.join(self.save_dir, 'weights_{}'.format(gpu))
        os.mkdir(weights_path)
        for name, p in self.model.named_parameters():
            np.save(os.path.join(weights_path, '{}.npy'.format(name)), p.detach().cpu().numpy())

    def load_model(self):
        pretrained_dict = torch.load(self.config['resume_path'], map_location='cpu')
        pretrained_dict_copy = dict()
        for key in pretrained_dict.keys():
            if self.config['dist']:
                pretrained_dict_copy[key[7:]] = pretrained_dict[key]
            else:
                pretrained_dict_copy[key[7:]] = pretrained_dict[key]
            #    #pretrained_dict_copy[key] = pretrained_dict[key]
        try:
            self.model.load_state_dict(pretrained_dict_copy)
        except:
            self.model.load_state_dict(pretrained_dict)
        print("Loaded weights")

    def print_metrics(self, metrics, scope, gpu):
        if scope == 'train':
            print("Train loss is {} at GPU {}".format(metrics.avg_loss, gpu))
        elif scope == 'val':
            print("Val loss is {} at GPU {}".format(metrics.avg_loss, gpu))

    def train_epoch(self, epoch, gpu):
        metrics = self.create_metrics(name="Train metrics")
        gradients = self.create_gradients(name="Gradients")
        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            batch['data'] = batch['data'].cuda(gpu)
            if 'control_inputs' in batch.keys():
                batch['control_inputs'] = batch['control_inputs'].cuda(gpu)
            batch['label'] = batch['label'].cuda(gpu)
            output = self.forward_model(batch)
            self.optimiser.zero_grad()
            batch['label'] = batch['label']
            losses = self.forward_loss(batch, output, epoch=epoch)
            loss = losses['loss']
            loss.backward()
            self.optimiser.step()
            self.scheduler.step()
            metrics.update(loss=losses, pred=output, n=batch['data'].size(0), targets=batch['label'], batch=batch)
            gradients = None
            if self.config['log'] == 'iter':
                if (self.config["dist"] and gpu == 0) or (not self.config["dist"]):
                    if self.config["write_videos"]:
                        bs = batch['data'].shape[0]
                        n_write_videos = np.minimum(bs, self.config[
                            "write_videos"])  # Videos are taken from the last batch, which might not be of size batch_size
                        idx = torch.randint(bs, (n_write_videos, 1), generator=None).view(
                            (n_write_videos))  # Randomly select ids to show
                        sub_batch = dict()
                        sub_batch['data'] = batch['data'][idx].detach().cpu().numpy()
                        sub_batch['name'] = [batch['name'][idc] for idc in idx]
                        sub_batch['label'] = batch['label'][idx].detach().cpu().numpy()
                        if type(output) == dict:
                            if 'recon' in output.keys():
                                sub_output = dict()
                                sub_output['recon'] = output['recon'][idx].detach().cpu().numpy()
                                self.add_scalars(scope='train', metrics=metrics, gradients=gradients, epoch=batch_idx,
                                                 batch=sub_batch, output=sub_output)
                        else:
                            self.add_scalars(scope='train', metrics=metrics, gradients=gradients, epoch=batch_idx,
                                             batch=sub_batch)
                    else:
                        self.add_scalars(scope='train', metrics=metrics, gradients=gradients, epoch=batch_idx,
                                         batch=None)
        if self.config['log'] == "epoch":
            if (self.config["dist"] and gpu == 0) or (not self.config["dist"]):
                if self.config["write_videos"]:
                    bs = batch['data'].shape[0]
                    n_write_videos = np.minimum(bs, self.config[
                        "write_videos"])  # Videos are taken from the last batch, which might not be of size batch_size
                    idx = torch.randint(bs, (n_write_videos, 1), generator=None).view(
                        (n_write_videos))  # Randomly select ids to show
                    sub_batch = dict()
                    sub_batch['data'] = batch['data'][idx].detach().cpu().numpy()
                    sub_batch['name'] = [batch['name'][idc] for idc in idx]
                    sub_batch['label'] = batch['label'][idx].detach().cpu().numpy()
                    if type(output) == dict:
                        if 'recon' in output.keys():
                            sub_output = dict()
                            sub_output['recon'] = output['recon'][idx].detach().cpu().numpy()
                            self.add_scalars(scope='train', metrics=metrics, gradients=gradients, epoch=epoch,
                                             batch=sub_batch, output=sub_output)
                    else:
                        self.add_scalars(scope='train', metrics=metrics, gradients=gradients, epoch=epoch,
                                         batch=sub_batch)
                else:
                    self.add_scalars(scope='train', metrics=metrics, gradients=gradients, epoch=epoch, batch=None)
        if (self.config["dist"] and gpu == 0) or (not self.config["dist"]):
            self.writer.flush()
        self.print_metrics(metrics, gpu=gpu, scope='train')
        return metrics.avg_loss

    def test_epoch(self, epoch, gpu):
        metrics = self.create_metrics(name="Val metrics")
        for batch_idx, batch in enumerate(tqdm(self.val_loader)):
            batch['data'] = batch['data'].cuda(gpu)
            if 'control_inputs' in batch.keys():
                batch['control_inputs'] = batch['control_inputs'].cuda(gpu)
            batch['label'] = batch['label'].cuda(gpu)
            batch['label'] = batch['label']
            output = self.forward_model(batch)
            self.optimiser.zero_grad()
            losses = self.forward_loss(batch, output, epoch=epoch)
            metrics.update(loss=losses, pred=output, n=batch['data'].size(0), targets=batch['label'], batch=batch)
            if self.config['log'] == 'iter':
                if (self.config["dist"] and gpu == 0) or (not self.config["dist"]):
                    if self.config["write_videos"]:
                        bs = batch['data'].shape[0]
                        n_write_videos = np.minimum(bs, self.config[
                            "write_videos"])  # Videos are taken from the last batch, which might not be of size batch_size
                        idx = torch.randint(bs, (n_write_videos, 1), generator=None).view(
                            (n_write_videos))  # Randomly select ids to show
                        sub_batch = dict()
                        sub_batch['data'] = batch['data'][idx].detach().cpu().numpy()
                        sub_batch['name'] = [batch['name'][idc] for idc in idx]
                        sub_batch['label'] = batch['label'][idx].detach().cpu().numpy()
                        if type(output) == dict:
                            if 'recon' in output.keys():
                                sub_output = dict()
                                sub_output['recon'] = output['recon'][idx].detach().cpu().numpy()
                                self.add_scalars(scope='val', metrics=metrics, epoch=batch_idx, batch=sub_batch,
                                                 output=sub_output)
                        else:
                            self.add_scalars(scope='val', metrics=metrics, epoch=batch_idx, batch=sub_batch)
                    else:
                        self.add_scalars(scope='val', metrics=metrics, epoch=batch_idx, batch=None)

        if self.config['log'] == 'epoch':
            if (self.config["dist"] and gpu == 0) or (not self.config["dist"]):
                if self.config["write_videos"]:
                    bs = batch['data'].shape[0]
                    n_write_videos = np.minimum(bs, self.config[
                        "write_videos"])  # Videos are taken from the last batch, which might not be of size batch_size
                    idx = torch.randint(bs, (n_write_videos, 1), generator=None).view(
                        (n_write_videos))  # Randomly select ids to show
                    sub_batch = dict()
                    sub_batch['data'] = batch['data'][idx].detach().cpu().numpy()
                    sub_batch['name'] = [batch['name'][idc] for idc in idx]
                    sub_batch['label'] = batch['label'][idx].detach().cpu().numpy()
                    if type(output) == dict:
                        if 'recon' in output.keys():
                            sub_output = dict()
                            sub_output['recon'] = output['recon'][idx].detach().cpu().numpy()
                            self.add_scalars(scope='val', metrics=metrics, epoch=epoch, batch=sub_batch,
                                             output=sub_output)
                    else:
                        self.add_scalars(scope='val', metrics=metrics, epoch=epoch, batch=sub_batch)
                else:
                    self.add_scalars(scope='val', metrics=metrics, epoch=epoch, batch=None)
        if (self.config["dist"] and gpu == 0) or (not self.config["dist"]):
            self.writer.flush()
        self.print_metrics(metrics, gpu=gpu, scope='val')
        losses = self.create_losses_dict(metrics=metrics)
        return metrics.avg_loss, losses

    def train(self, gpu, ngpus_per_node=None, args=None):
        global best_loss
        best_loss = 1e10
        if self.config["dist"]:
            args["gpu"] = gpu
            args["rank"] = args["rank"] * ngpus_per_node + gpu
            torch.cuda.set_device(gpu)
            dist.init_process_group("gloo", rank=args["rank"], world_size=args["world_size"])
            self.model.cuda(gpu)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[gpu])
            batch_size = int(self.config["batch_size_train"] / ngpus_per_node)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,
                                                                                 num_replicas=args["world_size"],
                                                                                 rank=args["rank"])
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset,
                                                                               num_replicas=args["world_size"],
                                                                               rank=args["rank"])
        else:
            self.model.cuda(gpu)
            batch_size = self.config["batch_size_train"]

        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=batch_size,
                                                        shuffle=(self.train_sampler is None),
                                                        num_workers=self.config["num_workers"], pin_memory=True,
                                                        sampler=self.train_sampler)

        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=batch_size,
                                                      shuffle=(self.train_sampler is None),
                                                      num_workers=self.config["num_workers"], pin_memory=True,
                                                      sampler=self.val_sampler)
        self.create_optimiser()
        self.create_scheduler()
        for epoch in tqdm(range(self.config["n_epochs"])):
            self.model.train()
            train_loss = self.train_epoch(epoch, gpu)
            with torch.no_grad():
                self.model.eval()
                val_loss, losses = self.test_epoch(epoch, gpu)
                if (self.config["dist"] and gpu == 0) or (not self.config["dist"]):
                    self.update_best_losses_dict(losses)
                    self.save(self.best_losses_dict)

    def save(self, losses):
        """
        Returns
        -------
        None

        Saves models with lowest loss values
        When overriding this method, add sth like:
            torch.save(self.model.state_dict(),
                    os.path.join(self.save_dir,
                                            "model_reco_loss_{}.pth".format(self.best_losses_dict["reco_loss"])))
        """
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir, "model_loss_{}.pth".format(self.best_losses_dict["loss"])))

    def evaluate(self, gpu, ngpus_per_node=None, args=None):
        if self.config["dist"]:
            args["gpu"] = gpu
            args["rank"] = args["rank"] * ngpus_per_node + gpu
            torch.cuda.set_device(gpu)
            dist.init_process_group("gloo", rank=args["rank"], world_size=args["world_size"])
            self.model.cuda(gpu)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[gpu])
            batch_size = int(self.config["batch_size_train"] / ngpus_per_node)
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset,
                                                                               num_replicas=args["world_size"],
                                                                               rank=args["rank"])
            if 'eval' in self.config.keys():
                if self.config['eval']:
                    self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.eval_dataset,
                                                                                        num_replicas=args["world_size"],
                                                                                        rank=args["rank"])
        elif self.config['dist'] == 0 and self.config['gpu']:
            self.model.cuda(gpu)
            batch_size = self.config["batch_size_train"]

        elif (self.config['dist'] == 0) and (self.config['gpu'] == 0):
            batch_size = self.config["batch_size_train"]
        self.val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=batch_size,
                                                      shuffle=(self.val_sampler is None),
                                                      num_workers=self.config["num_workers"], pin_memory=True,
                                                      sampler=self.val_sampler)
        if 'eval' in self.config.keys():
            if self.config['eval']:
                self.eval_loader = torch.utils.data.DataLoader(dataset=self.eval_dataset, batch_size=batch_size,
                                                               shuffle=(self.eval_sampler is None),
                                                               num_workers=self.config["num_workers"], pin_memory=True,
                                                               sampler=self.eval_sampler)
                self.val_loader = self.eval_loader

        metrics = self.create_eval_metrics(name="Evaluation metrics", config=self.config)
        self.model.eval()
        for batch_idx, batch in enumerate(tqdm(self.val_loader)):
            batch['data'] = batch['data'].cuda(gpu)
            if 'control_inputs' in batch.keys():
                batch['control_inputs'] = batch['control_inputs'].cuda(gpu)
            if self.config['gpu']:
                batch['data'] = batch['data'].cuda(gpu)
            output = self.forward_model(batch)
            metrics.update(pred=output, targets=batch['data'], n=batch['data'].size(0), batch=batch)
        metrics.print()
        metrics.save(path=self.config['exp_path'])


class DAETrainer(Trainer):
    def __init__(self, config, gpu):
        Trainer.__init__(self, config, gpu)

    def create_data(self):
        if self.config['dataset_type'] == 'pendulum':
            self.train_dataset = PendulumDataset(root_path=self.config["data_path"], split="train",
                                                 dataset_fraction=self.config['dataset_fraction'],
                                                 overfit=self.config["overfit"],
                                                 sample_duration=self.config["sample_duration"],
                                                 video_fmt=self.config['video_fmt'], control=self.config['ddmdc'],
                                                 normalize=self.config['normalize_control_inputs'],
                                                 dtype=self.config['dtype'])
            self.val_dataset = PendulumDataset(root_path=self.config["data_path"], split="test",
                                               dataset_fraction=self.config['dataset_fraction_test'],
                                               overfit=self.config["overfit"],
                                               sample_duration=self.config["sample_duration"],
                                               video_fmt=self.config['video_fmt'], control=self.config['ddmdc'],
                                               normalize=self.config['normalize_control_inputs'],
                                               dtype=self.config['dtype'])
            if 'eval' in self.config.keys():
                if self.config['eval']:
                    self.eval_dataset = PendulumDataset(root_path=self.config["data_path"], split="eval",
                                                        dataset_fraction=self.config['dataset_fraction_test'],
                                                        overfit=self.config["overfit"],
                                                        sample_duration=self.config["sample_duration"],
                                                        video_fmt=self.config['video_fmt'],
                                                        control=self.config['ddmdc'],
                                                        normalize=self.config['normalize_control_inputs'],
                                                        dtype=self.config['dtype'])

        elif self.config['dataset_type'] == 'flow':
            print('Flow dataset not implemented')
            raise NotImplementedError

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.config["batch_size_train"],
                                       shuffle=True, num_workers=self.config["num_workers"],
                                       pin_memory=self.config["pin_memory"])
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.config["batch_size_train"],
                                     shuffle=False, num_workers=self.config["num_workers"],
                                     pin_memory=self.config["pin_memory"])
        if 'eval' in self.config.keys():
            if self.config['eval']:
                self.eval_loader = DataLoader(dataset=self.eval_dataset, batch_size=self.config["batch_size_train"],
                                              shuffle=False, num_workers=self.config["num_workers"],
                                              pin_memory=self.config["pin_memory"])

    def set_save_dir(self):
        if self.config['dataset_type'] == 'pendulum':
            self.save_dir = os.path.join("runs_dae", self.config["exp_name"] + get_time(time.time()))
        elif self.config['dataset_type'] == 'flow':
            self.save_dir = os.path.join("runs_dae_flow", self.config["exp_name"] + get_time(time.time()))

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def create_writer(self):
        self.writer = SummaryWriter(log_dir=self.save_dir)

    def create_metrics(self, name):
        return RFMetrics(self, name)

    def create_eval_metrics(self, name, config):
        return RecoMetrics(self, name, config)

    def print_metrics(self, metrics, gpu, scope):
        if scope == 'train':
            print('Train loss at GPU {} is {} '.format(gpu, metrics.avg_loss))
            print('Train reco loss at GPU {} is {}'.format(gpu, metrics.avg_reco_loss))
        elif scope == 'val':
            print('Val loss at GPU {} is {} '.format(gpu, metrics.avg_loss))
            print('Val Reco loss at GPU {} is {}'.format(gpu, metrics.avg_reco_loss))

    def create_gradients(self, name):
        return DAEGradients(self, name, config=self.config)

    def add_scalars(self, scope, metrics, epoch, gradients=None, batch=None, output=None):
        if scope == 'train':
            self.writer.add_scalar("train/epoch_loss", metrics.avg_loss, epoch)
            self.writer.add_scalar("train/epoch_reco_loss", metrics.avg_reco_loss, epoch)
            self.writer.add_scalar("train/epoch_reco_first_loss", metrics.avg_reco_first_loss, epoch)
            self.writer.add_scalar("train/epoch_reco_last_loss", metrics.avg_reco_last_loss, epoch)
            self.writer.add_scalar("train/epoch_penalized_loss", metrics.avg_longterm_loss, epoch)

            if self.config['missing_states']:
                self.writer.add_scalar("train/epoch_reco_missing_loss", metrics.avg_reco_missing_loss, epoch)
            self.writer.add_scalar("train/epoch_features_cycle_loss", metrics.avg_features_cycle_loss, epoch)
            self.writer.add_scalar("learning_rate", self.optimiser.param_groups[0]['lr'], epoch)
            if self.config['model_name'] == 'DMDAE':
                self.writer.add_scalar("train/residual_loss", metrics.avg_res_loss, epoch)
            if batch is not None:
                if self.config['dataset_type'] == 'pendulum':
                    self.writer.add_video("train/gt", batch['data'], fps=20, global_step=epoch)
                    bs, m, c, h, w = output['recon'].shape
                    self.writer.add_video("train/recon_videos", output['recon'].reshape((bs, m, c, h, w)), fps=20,
                                          global_step=epoch)
                elif self.config['dataset_type'] == 'flow':
                    bs, m, c, h, w = output['recon'].shape
                    for channel in range(c):
                        self.writer.add_video("train/gt_{}".format(channel),
                                              batch['data'][:, :, channel, :, :].reshape((bs, m, 1, h, w)),
                                              fps=20, global_step=epoch)
                        self.writer.add_video("train/recon_videos_{}".format(channel),
                                              output['recon'].reshape((bs, m, c, h, w))[:, :, channel, :, :].reshape(
                                                  (bs, m, 1, h, w)),
                                              fps=20, global_step=epoch)

            if gradients is not None:
                for module_name in gradients.modules_gradients.keys():
                    for metric in gradients.modules_gradients[module_name].keys():
                        self.writer.add_scalar("train/{}_grad_{}".format(metric, module_name),
                                               torch.tensor(gradients.modules_gradients[module_name][
                                                                metric]).detach().cpu().numpy(), epoch)
        elif scope == 'val':
            self.writer.add_scalar("val/epoch_loss", metrics.avg_loss, epoch)
            self.writer.add_scalar("val/epoch_reco_loss", metrics.avg_reco_loss, epoch)
            self.writer.add_scalar("val/epoch_reco_first_loss", metrics.avg_reco_first_loss, epoch)
            self.writer.add_scalar("val/epoch_reco_last_loss", metrics.avg_reco_last_loss, epoch)
            self.writer.add_scalar("val/epoch_penalized_loss", metrics.avg_longterm_loss, epoch)
            if self.config['missing_states']:
                self.writer.add_scalar("val/epoch_reco_missing_loss", metrics.avg_reco_missing_loss, epoch)
            self.writer.add_scalar("val/epoch_features_cycle_loss", metrics.avg_features_cycle_loss, epoch)
            if self.config['model_name'] == 'DMDAE':
                self.writer.add_scalar("val/residual_loss", metrics.avg_res_loss, epoch)
            if batch is not None:
                if self.config['dataset_type'] == 'pendulum':
                    self.writer.add_video("val/gt", batch['data'], fps=20, global_step=epoch)
                    bs, m, c, h, w = output['recon'].shape
                    self.writer.add_video("val/recon_videos", output['recon'].reshape((bs, m, c, h, w)), fps=20,
                                          global_step=epoch)
                elif self.config['dataset_type'] == 'flow':
                    bs, m, c, h, w = output['recon'].shape
                    for channel in range(c):
                        self.writer.add_video("val/gt_{}".format(channel),
                                              batch['data'][:, :, channel, :, :].reshape((bs, m, 1, h, w)),
                                              fps=20, global_step=epoch)
                        self.writer.add_video("val/recon_videos_{}".format(channel),
                                              output['recon'].reshape((bs, m, c, h, w))[:, :, channel, :, :].reshape(
                                                  (bs, m, 1, h, w)),
                                              fps=20, global_step=epoch)

    def build_model(self):
        autoenc = get_autoencoder(name=self.config['autoencoder_name'], config=self.config)
        if self.config['model_name'] == 'DMDAE':
            self.model = DMDAE(autoencoder=autoenc, r=self.config['r'], n_points_A=self.config['n_frames_A'],
                               config=self.config)

    def forward_model(self, batch):
        if self.config['ddmdc']:

            out = self.model(batch)
        else:
            out = self.model(batch['data'])
        return out

    def return_gradients(self):
        if self.config["dist"] == 1:
            return self.model.module.return_gradients()
        else:
            return self.model.return_gradients()

    def create_optimiser(self):
        if self.config["optim"] == "sgd":
            self.optimiser = torch.optim.SGD(self.model.parameters(), lr=self.config["lr"],
                                             momentum=self.config["momentum"], dampening=self.config["dampening"],
                                             weight_decay=self.config["weight_decay"], nesterov=False)
        elif self.config["optim"] == "adam":
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"], betas=self.config["betas"],
                                              weight_decay=self.config["weight_decay"])

    def create_scheduler(self):
        step_size = int(self.config["decay_lr_every"] * self.train_dataset.__len__() / self.config["batch_size_train"])
        self.scheduler = StepLR(self.optimiser, step_size=step_size, gamma=self.config["lr_decay"])

    def create_loss(self):
        self.loss_fn = get_loss(name=self.config["loss"], config=self.config)

    def create_best_losses_dict(self):
        self.best_losses_dict = dict()
        self.best_losses_dict["loss"] = 1e10
        self.best_losses_dict["reco_loss"] = 1e10
        self.best_losses_dict["reco_first_loss"] = 1e10
        self.best_losses_dict["reco_last_loss"] = 1e10
        self.best_losses_dict["features_cycle_loss"] = 1e10
        self.best_losses_dict["residual_loss"] = 1e10

    def create_losses_dict(self, metrics):
        losses = dict()
        losses['loss'] = metrics.avg_loss  # Total loss (that we optimize for)
        losses['reco_loss'] = metrics.avg_reco_loss
        losses['reco_first_loss'] = metrics.avg_reco_first_loss
        losses['reco_last_loss'] = metrics.avg_reco_last_loss
        losses['features_cycle_loss'] = metrics.avg_features_cycle_loss
        losses['residual_loss'] = metrics.avg_res_loss
        return losses

    def save(self, losses):
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir, "model_loss_{}.pth".format(self.best_losses_dict["loss"])))
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir, "model_reco_loss_{}.pth".format(self.best_losses_dict["reco_loss"])))
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir,
                                "model_reco_first_loss_{}.pth".format(self.best_losses_dict["reco_first_loss"])))
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir,
                                "model_reco_last_loss_{}.pth".format(self.best_losses_dict["reco_last_loss"])))
        torch.save(self.model.state_dict(),
                   os.path.join(self.save_dir,
                                "model_residual_loss_{}.pth".format(self.best_losses_dict["residual_loss"])))

    def forward_loss(self, batch, output, epoch):
        """
        Params
        ------

        Returns
        -------
        losses: dict
            - loss: loss that is minimized (can be = to reco loss, ord reco+residual, ...)
            - reco: reconstruction loss
            - residual

        """
        mse = nn.MSELoss(reduction='mean')
        losses = dict()
        output_first = output.copy()
        output_last = output.copy()
        output_last['recon'] = output['recon'][:, self.config['n_frames_A']:, :, :, :]
        output_first['recon'] = output['recon'][:, :self.config['n_frames_A'], :, :, :]
        loss_reco_all = self.loss_fn(x=batch['data'], output=output)
        loss_reco_first = self.loss_fn(x=batch['data'][:, :self.config['n_frames_A'], :, :, :], output=output_first)
        if self.config['n_frames_A'] == self.config['sample_duration']:
            loss_reco_last = 0
        else:
            loss_reco_last = self.loss_fn(x=batch['data'][:, self.config['n_frames_A']:, :, :, :], output=output_last)
        bs, m, c, h, w = output['recon'].shape
        if self.config['missing_states']:
            n_missing_states = len(self.config['missing_states'])
            output_missing = dict()
            output_missing['recon'] = torch.zeros((bs, n_missing_states, c, h, w),
                                                  device=output['recon'].device, dtype=output['recon'].dtype)
            input_missing = dict()
            input_missing['data'] = torch.zeros((bs, n_missing_states, c, h, w),
                                                device=output['recon'].device, dtype=output['recon'].dtype)
            for i, k in enumerate(self.config['missing_states']):
                output_missing['recon'] = output['recon'][:, k, :, :, :]
                input_missing['data'] = batch['data'][:, k, :, :, :]
            loss_reco_missing = self.loss_fn(x=input_missing['data'], output=output_missing)
            input_first = dict()
            output_first = dict()
            input_first['data'] = torch.zeros(
                (bs, self.config['n_frames_A'] - len(self.config['missing_states']), c, h, w),
                dtype=output['recon'].dtype, device=output['recon'].device)
            output_first['recon'] = torch.zeros(
                (bs, self.config['n_frames_A'] - len(self.config['missing_states']), c, h, w),
                dtype=output['recon'].dtype, device=output['recon'].device)
            first_missing_idx = self.config['missing_states'][0]
            input_first['data'][:, :first_missing_idx, :, :, :] = batch['data'][:, :first_missing_idx, :, :, :]
            output_first['recon'][:, :first_missing_idx, :, :, :] = output['recon'][:, :first_missing_idx, :, :, :]
            for i in range(n_missing_states - 1):
                k = self.config['missing_states'][i]
                k_next = self.config['missing_states'][i + 1]
                output_first['recon'][:, k - i + 1:k_next - i, :, :, :] = output['recon'][:, k + 1:k_next, :, :, :]
                input_first['data'][:, k - i + 1:k_next - i, :, :, :] = batch['data'][:, k + 1:k_next, :, :, :]
            last_missing_idx = self.config['missing_states'][-1]
            input_first['data'][:, last_missing_idx - n_missing_states:, :, :, ] = batch['data'][:,
                                                                                   last_missing_idx + 1:self.config[
                                                                                                            'n_frames_A'] + 1,
                                                                                   :, :, :]
            output_first['recon'][:, last_missing_idx - n_missing_states:, :, :, ] = output['recon'][:,
                                                                                     last_missing_idx + 1:self.config[
                                                                                                              'n_frames_A'] + 1:,
                                                                                     :, :, :]
            loss_reco_first = self.loss_fn(x=input_first['data'], output=output_first)

        if self.config['loss_reduction'] == 'sum':
            if self.config['missing_states']:
                return NotImplemented
            else:
                loss_reco = self.config['alpha_reco_first'] * loss_reco_first + self.config[
                    'alpha_reco_last'] * loss_reco_last
        elif self.config['loss_reduction'] == 'mean':
            alpha = self.config['n_frames_A'] / self.config['sample_duration']
            if self.config['missing_states']:
                # Override alpha
                alpha = (self.config['n_frames_A'] - len(self.config['missing_states'])) / self.config[
                    'sample_duration']  # coeff of loss of first terms
                alpha_missing = len(self.config['missing_states']) / self.config['sample_duration']
                loss_reco = self.config['alpha_reco_first'] * alpha * loss_reco_first + self.config[
                    'alpha_reco_missing'] * alpha_missing * loss_reco_missing + (1 - alpha - alpha_missing) * \
                            self.config['alpha_reco_last'] * loss_reco_last
            else:
                loss_reco = self.config['alpha_reco_first'] * alpha * loss_reco_first + (1 - alpha) * self.config[
                    'alpha_reco_last'] * loss_reco_last
        losses['reco'] = loss_reco_all
        losses['reco_last'] = loss_reco_last
        losses['reco_first'] = loss_reco_first
        if self.config['missing_states']:
            losses['reco_missing'] = loss_reco_missing
        losses['features_cycle'] = 0
        if (self.config['delayed_dmd']) and not (self.config['ddmdc']):
            Y_h, Yh = build_shifted_matrices(Y=output['features'].transpose(1, 2), h=self.config['history'])
            losses['residual'] = torch.abs(Yh - output['A_matrix'] @ Y_h).max()
        elif self.config['ddmdc']:
            Y_h, Yh = build_shifted_matrices(Y=output['features'].transpose(1, 2), h=self.config['history'])
            Y_h_ = torch.zeros((bs, self.config['history'] * self.config['ae_out_dim'] + self.config['control_d'],
                                m - self.config['history']), device=output['features'].device,
                               dtype=output['features'].dtype)
            Y_h_[:, :-self.config['control_d'], :] = Y_h
            Y_h_[:, -self.config['control_d']:, :] = batch['control_inputs'].transpose(1, 2)[:, :,
                                                     self.config['history']:]
            losses['residual'] = torch.abs(Yh - output['A_matrix'] @ Y_h_).max()
        else:
            losses['residual'] = torch.norm(
                output['features'].transpose(1, 2)[:, :, 1:] - output['A_matrix'].bmm(
                    output['features'].transpose(1, 2)[:, :, :-1]))

        if self.config['longterm_pen']:
            loss_pen = mse(batch['data'][:, self.config['n_frames_A']:, :, :, :], output['penalized'])
        else:
            loss_pen = 0
        loss = loss_reco + self.config['alpha_res'] * losses[
            'residual'] + loss_pen
        losses['loss'] = loss
        return losses
