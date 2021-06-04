import torch
import torch.nn as nn
from odc.models import CNNAE, DMDAE
import numpy as np
import os



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool  = nn.AvgPool2d(3, 1)
        self.mu_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((SSIM_n / SSIM_d), 0, 1)


class RecoMetrics(object):
    def __init__(self, name, fmt=':f', config=None):
        self.name = name
        self.fmt = fmt
        self.config = config
        if 'metric_prefix' in self.config.keys():
            self.metric_prefix = self.config['metric_prefix']
        else:
            self.metric_prefix = ''
        self.reset()

    def reset(self):
        sd = self.config['sample_duration']
        self.count = 0
        self.mse = torch.zeros((sd))
        self.sum_mse = torch.zeros((sd))
        self.avg_mse = torch.zeros((sd))

        self.std = torch.zeros((sd))
        self.sum_std = torch.zeros((sd))
        self.avg_std = torch.zeros((sd))
        

        self.features_cycle = torch.zeros((sd))
        self.sum_features_cycle = torch.zeros((sd))
        self.avg_features_cycle = torch.zeros((sd))
   
        self.psnr = torch.zeros((sd))
        self.sum_psnr = torch.zeros((sd))
        self.avg_psnr = torch.zeros((sd))


        self.ssim = torch.zeros((sd))
        self.sum_ssim = torch.zeros((sd))
        self.avg_ssim = torch.zeros((sd))

        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM()
        
    def update(self, pred, targets, batch=None, n=1):
        self.count += n
        mse = torch.mean((pred['recon'] - targets)**2, dim=[-1, -2, -3]).detach().cpu()
        std = torch.std((pred['recon'] - targets)**2, dim=[-1, -2, -3]).detach().cpu()
        sd = self.config['sample_duration']
        bs = pred['recon'].shape[0]
        ssim_loss = torch.zeros((bs, sd))
        for i in range(sd):
            ssim_loss[:, i] = self.ssim_loss(pred['recon'][:, i, :, :, :], targets[:, i, :, :, :]).mean(-1).mean(-1).mean(-1).detach().cpu()

        psnr = (10 * np.log10(1 / mse)).sum(0)
        self.sum_psnr += psnr 
        self.avg_psnr = self.sum_psnr / self.count
        self.sum_mse += mse.sum(0)
        self.sum_std += std.sum(0)
        self.avg_mse = self.sum_mse / self.count
        self.avg_std = self.sum_std / self.count
        self.sum_ssim += ssim_loss.sum(0)
        self.avg_ssim = self.sum_ssim / self.count

    def print(self):
        pass

    def save(self, path):
        psnr_path = os.path.join(path, self.metric_prefix + 'psnr.npy')
        mse_path = os.path.join(path, self.metric_prefix+ 'mse.npy')
        std_path = os.path.join(path, self.metric_prefix + 'std.npy')
        ssim_path = os.path.join(path,self.metric_prefix + 'ssim.npy')
        np.save(psnr_path, self.avg_psnr)
        np.save(mse_path, self.avg_mse)
        np.save(std_path, self.avg_std)
        np.save(ssim_path, self.avg_ssim)


class RFMetrics(object):
    """
    Attributes:
    - avg_loss: loss that is minimized
    - avg_reco_loss: avg reconstruction loss 
    - avg_reco_first_loss: avg reconstruction loss of the first frames
    - avg_reco_last_loss: avg reconstruction loss of the last frames
    - avg_res_loss: avg residual loss
    


    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
   
    def reset(self):
        """
        reset is called at each epoch 
        """
        self.loss = 0
        self.avg_loss = 0
        self.sum_loss = 0
        self.reco_loss = 0
        self.reco_first_loss = 0
        self.reco_last_loss = 0
        self.reco_missing_loss = 0
        self.avg_reco_loss = 0
        self.avg_reco_first_loss = 0
        self.avg_reco_last_loss = 0
        self.avg_reco_missing_loss = 0
        self.sum_reco_loss = 0
        self.sum_reco_first_loss = 0
        self.sum_reco_last_loss = 0
        self.sum_reco_missing_loss = 0
        self.a_loss = 0
        self.avg_a_loss = 0
        self.sum_a_loss = 0
        self.res_loss = 0
        self.sum_res_loss = 0
        self.avg_res_loss = 0
        self.features_cycle_loss = 0
        self.sum_features_cycle_loss = 0
        self.avg_features_cycle_loss = 0

        self.longterm_loss = 0
        self.sum_longterm_loss = 0
        self.avg_longterm_loss = 0

        self.count = 0


    def update(self, pred, loss, n=1, targets=None, batch=None):
        """
        update is done at each iteration
        """
        self.sum_loss += loss['loss'] * n
        if 'reco' in loss.keys():
            self.sum_reco_loss += loss['reco'] * n
        if 'A_loss' in loss.keys():
            self.sum_a_loss += loss['A_loss'] * n
        if 'residual' in loss.keys():
            self.sum_res_loss += loss['residual'] * n
        if 'reco_first' in loss.keys():
            self.sum_reco_first_loss += loss['reco_first'] * n
        if 'reco_last' in loss.keys():
            self.sum_reco_last_loss += loss['reco_last'] * n
        if 'features_cycle' in loss.keys():
            self.sum_features_cycle_loss += loss['features_cycle'] * n
        if 'reco_missing' in loss.keys():
            self.sum_reco_missing_loss += loss['reco_missing'] * n
        if 'longterm_pen' in loss.keys():
            self.sum_longterm_loss += loss['longterm_loss'] * n
        self.count += n
        self.avg_loss = self.sum_loss / self.count
        self.avg_a_loss = self.sum_a_loss / self.count
        self.avg_reco_loss = self.sum_reco_loss / self.count
        self.avg_reco_last_loss = self.sum_reco_last_loss / self.count
        self.avg_reco_first_loss = self.sum_reco_first_loss / self.count
        self.avg_reco_missing_loss = self.sum_reco_missing_loss / self.count
        self.avg_res_loss = self.sum_res_loss / self.count
        self.avg_features_cycle_loss = self.sum_features_cycle_loss / self.count
        self.avg_longterm_loss = self.sum_longterm_loss / self.count


class DAEGradients(object):
    def __init__(self, name, fmt=':f', config=None):
        """
        self.module_gradients: dict()
            - keys: names of the modules whose gradients we're saving 
            - values: dict()
                - keys: 'min', 'max', 'avg'
                - values: values
        """
        self.name = name
        self.fmt = fmt
        config['ae_resume_path'] = None
        autoenc = CNNAE(config=config)
        model = DMDAE(autoencoder=autoenc, config=config, n_points_A=50)
        self.modules_gradients = dict()
        for name, _ in model.named_modules():
            self.modules_gradients[name] = dict()
        self.reset()

    def reset(self):
        for module_name in self.modules_gradients.keys():
            self.modules_gradients[module_name]['max'] = 1e10
            self.modules_gradients[module_name]['min'] = -1e10
            self.modules_gradients[module_name]['avg'] = 0
    
    def update(self, gradients, n=1):
        """
        gradients: dict()
            dict of gradients returned by gradient hook. Should be st:
            - keys: module_names
            - values: dict()
                - keys: 'min', 'max', 'avg'
                - values: values
        """
        for module_name in gradients.keys():
            min_nan = False
            for val in gradients[module_name]['min']:
                if torch.isnan(val):
                    self.modules_gradients[module_name]['min'] = float('nan')
                    min_nan = True
            if not min_nan:
                self.modules_gradients[module_name]['min'] = min(gradients[module_name]['min'])

            max_nan = False
            for val in gradients[module_name]['max']:
                if torch.isnan(val):
                    self.modules_gradients[module_name]['max'] = float('nan')
                    max_nan = True
            if not max_nan:
                self.modules_gradients[module_name]['max'] = max(gradients[module_name]['max'])
