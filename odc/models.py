"""
Contains CNNAE autoencoder, and DMDAE model
"""

import torch
import torch.nn as nn
from odc.model_utils import batch_eye, build_shifted_matrices, estimate_from_history_matrices_pytorch
from odc.modules import DelayedDMD, DelayedDMDc, DMD
from odc.train_utils import make_gradients_dict, my_register_backward_hook
from odc.autoencoders import AutoEncoder


class CNNAE(AutoEncoder):
    def __init__(self, config=None):
        super(CNNAE, self).__init__()
        self.config = config
        n_conv = len(self.config['ae_out_planes'])
        self.convolutions = []
        self.transposed_convolutions = []
        self.bns_down = []
        self.bns_up = []
        self.conv1 = nn.Conv2d(in_channels=self.config['n_channels'], out_channels=self.config['ae_out_planes'][0],
                               stride=self.config['strides'][0], padding=self.config['paddings'][0], kernel_size=3)
        bn1 = nn.BatchNorm2d(self.config['ae_out_planes'][0])
        self.bns_down.append(bn1)
        self.convolutions.append(self.conv1)
        for i, out_planes in enumerate(self.config['ae_out_planes'][1:]):
            conv = self.make_3x3_conv_layer(in_channels = self.config['ae_out_planes'][i],
                                            out_channels = self.config['ae_out_planes'][i + 1],
                                            stride=self.config['strides'][i+1],
                                            padding=self.config['paddings'][i+1])
            t_conv = nn.ConvTranspose2d(in_channels=self.config['ae_out_planes'][-(i +1)],
                                        out_channels=self.config['ae_out_planes'][-(i +2)],
                                        kernel_size=2, stride=2)
            bn = nn.BatchNorm2d(self.config['ae_out_planes'][i + 1])
            bn_up = nn.BatchNorm2d(self.config['ae_out_planes'][-(i +2)])
            self.convolutions.append(conv)
            self.transposed_convolutions.append(t_conv)
            self.bns_down.append(bn)
            self.bns_up.append(bn_up)
        t_conv4 = nn.ConvTranspose2d(in_channels=self.config['ae_out_planes'][0],
                                     out_channels=self.config['n_channels'],
                                     kernel_size=2, stride=2)
        self.transposed_convolutions.append(t_conv4)
        bn_last = nn.BatchNorm2d(self.config['n_channels'])
        self.bns_up.append(bn_last)
        if self.config['dtype'] == 'double':
            self.transposed_convolutions = [conv.double() for conv in self.transposed_convolutions]
            self.convolutions = [conv.double() for conv in self.convolutions]
            self.bns_up = [bn.double() for bn in self.bns_up]
            self.bns_down = [bn.double() for bn in self.bns_down]

        self.transposed_convolutions = nn.ModuleList(self.transposed_convolutions)
        self.convolutions = nn.ModuleList(self.convolutions)
        self.bns_up = nn.ModuleList(self.bns_up)
        self.bns_down = nn.ModuleList(self.bns_down)
        if self.config['mlp_in_ae']:
            self.linears_down = []
            self.linears_up = []
            layer_down = nn.Linear(self.config['conv_o_dim'], config['linear_neurons'][0])
            self.linears_down.append(layer_down)
            for i, neurons in enumerate(config['linear_neurons'][1:]):
                layer = nn.Linear(config['linear_neurons'][i], config['linear_neurons'][i + 1])
                layer_up = nn.Linear(config['linear_neurons'][- (i + 1)], config['linear_neurons'][-(i + 2)])
                self.linears_down.append(layer)
                self.linears_up.append(layer_up)
            layer_up = nn.Linear(config['linear_neurons'][0], self.config['conv_o_dim'])
            self.linears_up.append(layer_up)
            self.linears_up = nn.ModuleList(self.linears_up)
            self.linears_down = nn.ModuleList(self.linears_down)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        if self.config['ae_resume_path']:
            pretrained_dict = torch.load(self.config['ae_resume_path'], map_location='cpu')
            pretrained_dict_copy = dict()
            for key in pretrained_dict.keys():
                pretrained_dict_copy[key[12:]] = pretrained_dict[key]
                #pretrained_dict_copy[key[7:]] = pretrained_dict[key]
            self.load_state_dict(pretrained_dict_copy)

    def encode(self, x):
        c, h, w = x.shape[-3], x.shape[-2], x.shape[-1]
        out = self.convolutions[0](x.reshape((-1, c, h, w)))
        out = self.bns_down[0](out)
        out = self.relu(out)
        out = self.pool(out)
        for i, conv in enumerate(self.convolutions[1:-1]):
            out = conv(out)
            out = self.bns_down[i+1](out)
            out = self.relu(out)
            out = self.pool(out)
        out = self.convolutions[-1](out)
        out = self.bns_down[-1](out)
        out = self.pool(out)
        _, self.c_o, self.h_o, self.w_o = out.shape
        res = dict()
        out = out.view((-1, self.c_o * self.h_o * self.w_o))
        if self.config['mlp_in_ae']:
            for linear in self.linears_down[:-1]:
                out = linear(out)
                out = self.relu(out)
            out = self.linears_down[-1](out)
        res['latent'] = out
        return res

    def decode(self, x):
        if self.config['mlp_in_ae']:
            out = self.linears_up[0](x['latent'])
            for linear in self.linears_up[1:-1]:
                out = linear(out)
                out = self.relu(out)
            out = self.linears_up[-1](out)
        else:
            out = x['latent']
        out = self.transposed_convolutions[0](out.reshape((-1, self.c_o, self.h_o, self.w_o)))
        out = self.bns_up[0](out)
        out = self.relu(out)
        for i, t_conv in enumerate(self.transposed_convolutions[1:-1]):
            out = t_conv(out)
            out = self.bns_up[i+1](out)
            out = self.relu(out)
        out = self.transposed_convolutions[-1](out)
        out = self.bns_up[-1](out)
        out = self.sig(out)
        return out

    def forward(self, x):
        bs, *_ = x.shape
        out = self.encode(x)
        out = self.decode(out)
        output = dict()
        output['recon'] = out.reshape((x.shape))
        return output


    def make_3x3_conv_layer(self, in_channels, out_channels, stride, padding):
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding,
                         kernel_size=3)


class DMDAE(nn.Module):
    def __init__(self, autoencoder, n_points_A, encoder_mode=False, r=10, save_gradients=False, config=None):
        """
        Params
        ------
        autoencoder: AutoEncoder object, from pendulum/models/autoencoders.py
            Autoencoder that will be used to encode video frames in features, then decode features to video frames
        dt: float
        encoder_mode: bool
        r: int
        a_method: str
            can be 'svd', 'pinv', or 'still'
        """
        super(DMDAE, self).__init__()
        self.config = config
        self.encoder_mode = encoder_mode
        self.autoencoder = autoencoder
        self.n_reduced = self.config['ae_out_dim']
        self.r = self.config['r']
        self.a_method = self.config['a_method']
        if self.config['b_learnt']:
            self.B_matrix = nn.Linear(self.n_reduced, self.config['control_d'], bias=False)
            self.B_matrix.weight.data.fill_(0)
            if self.config['dtype'] == 'double':
                self.B_matrix = self.B_matrix.double()
        if self.config['a_method'] == 'learnt':
            self.A_matrix = nn.Linear(self.n_reduced, self.n_reduced * self.config['history'] + self.config['control_d'], bias=False)
        if self.config['delayed_dmd']:
            self.ddmd = DelayedDMD(a_method=self.a_method)
        if self.config['ddmdc']:
            if self.config['bn_control']:
                # self.bn_control = nn.BatchNorm1d(100)
                self.bn_control = nn.BatchNorm1d(1)
            if not self.config['b_learnt']:
                self.ddmdc = DelayedDMDc(a_method=self.a_method)
            else:
                # If B is learnt, we'ell use standard DDMD
                self.ddmd = DelayedDMD(a_method=self.a_method)

        else:
            self.dmd = DMD(a_method=self.a_method)
        self.n_points_A = n_points_A
        self.gradients = make_gradients_dict(self)
        if save_gradients:
            self.register_gradients()

    def decode(self, z):
        return self.autoencoder.decode(z)

    def forward(self, x):
        """
        Params
        ------
        x: tensor
            shape (bs, m, c, h, w)
            m: number of frames
        Returns
        -------
        output: dict with keys:
            - 'recon': tensor of shape (bs, m, c, h, w)
            encoded, advanced, decoded.
            - 'features': tensor of shape (bs, m, self.n_reduced)
            - 'features_cycle': tensor of shape (bs, m, self.n_reduced)
            - 'mu': (bs, 1)
            - 'logvar': (bs, 1)
            - 'fc1'
            - 'fc2'
            - 'P'
            - 'Q'
        """
        if self.config['ddmdc']:
            x, U = x['data'], x['control_inputs']
            _, _, d = U.shape
        bs, m, *_ = x.shape
        encoding = self.autoencoder.encode(x[:, :, ...])  # dict with one key: 'latent'. encoding['latent']: shape (bs, m, ae_out_dim)
        if 'skip' in encoding.keys():
            skip_connections = encoding['skip']
        else:
            skip_connections = None
        history_features = encoding['latent'].view((bs, m, self.n_reduced))[:, :self.n_points_A, :].transpose(1, 2)  # of shape (bs, self.n_reduced, self.n_points_A)
        if self.config['online_update']:
            a_dtype = 'double'
        else:
            a_dtype = self.config['dtype']
        if self.config['a_method'] != 'learnt':
            if (self.config['delayed_dmd']) and (not self.config['ddmdc']):
                A_matrix = self.ddmd(input=history_features,
                                               r=self.r, return_USV=False, a_method=self.a_method,
                                               a_return=self.config['a_return'],
                                               m_prox=self.config['m_prox'], lam_prox=self.config['lam_prox'],
                                               n_reduced=self.config['ae_out_dim'],
                                               missing_states=self.config['missing_states'],
                                               history=self.config['history'], a_dtype=a_dtype)

            elif self.config['ddmdc']:
                if self.config['bn_control']:
                    #U = self.bn_control(U)
                    U = self.bn_control(U.reshape((-1, 1))).reshape((bs, self.config['sample_duration'], self.config['control_d']))
                if not self.config['b_learnt']:
                    A_matrix = self.ddmdc(input=history_features, U=U.transpose(1, 2),
                                                   r=self.r, d=d, return_USV=False, a_method=self.a_method,
                                                   a_return=self.config['a_return'],
                                                   m_prox=self.config['m_prox'], lam_prox=self.config['lam_prox'],
                                                   n_reduced=self.config['ae_out_dim'],
                                                   missing_states=self.config['missing_states'],
                                                   history=self.config['history'], a_dtype=a_dtype)
                else:
                    B_matrix = self.B_matrix.weight.view((1, self.n_reduced, self.config['control_d'])).repeat(bs, 1, 1)
                    A_matrix = self.ddmd(input=history_features,
                                               r=self.r, return_USV=False, a_method=self.a_method,
                                               a_return=self.config['a_return'],
                                               m_prox=self.config['m_prox'], lam_prox=self.config['lam_prox'],
                                               n_reduced=self.config['ae_out_dim'],
                                               missing_states=self.config['missing_states'],
                                               history=self.config['history'], a_dtype=a_dtype, B=B_matrix,
                                                U=U.transpose(1, 2), d=self.config['control_d'])
            else:
                A_matrix = self.dmd(input=history_features,
                                r=self.r, return_USV=False, a_method=self.a_method, a_return=self.config['a_return'], 
                                m_prox=self.config['m_prox'], lam_prox=self.config['lam_prox'], 
                                n_reduced=self.n_reduced, missing_states=self.config['missing_states'])  # Build A matrix from the first n_points_A of features
        elif self.config['a_method'] == 'learnt':
            if (self.config['delayed_dmd']) and (not self.config['ddmdc']):
                A_matrix = self.A_matrix.weight.view((1, self.n_reduced, self.config['history'] * self.n_reduced)).repeat(bs, 1, 1)
            elif self.config['ddmdc']:
                A_matrix = self.A_matrix.weight.view(
                    (1, self.n_reduced, self.config['history'] * self.n_reduced + self.config['control_d'])).repeat(bs, 1, 1)

        if self.config['online_update']:
            if self.config['measured_states']:
                if not self.config['ddmdc']:
                    U = torch.zeros((bs, self.n_reduced, m))
                reco = self.online_reconstruct_k(
                    all_features=encoding['latent'].view((bs, m, self.n_reduced)).transpose(1, 2).double(),
                    A_aug_matrix=A_matrix, measured_states=self.config['measured_states'], U=U.transpose(1, 2).double())
            else:
                if self.config['ddmdc']:
                    measured_states = range(self.n_points_A - 2, m - self.config['history'])
                    reco = reco = self.online_reconstruct_k(
                    all_features=encoding['latent'].view((bs, m, self.n_reduced)).transpose(1, 2).double(),
                    A_aug_matrix=A_matrix, measured_states=measured_states, U=U.transpose(1, 2).double())
                else:
                    reco = self.online_reconstruct(
                        all_features=encoding['latent'].view((bs, m, self.n_reduced)).transpose(1, 2).double(),
                        A_aug_matrix=A_matrix)

        else:
            # If online_update == 0, reconstruct with corresponding method
            if (self.config['ddmdc']):
                reco = self.reconstruct_with_control(history_features=history_features, A=A_matrix[:, :, :-d],
                                                     B=A_matrix[:, :, -d:], U=U.transpose(1, 2), bs=bs, m=m)
            elif self.config['delayed_dmd']:
                reco = self.dreconstruct(history_features=history_features, A_aug_matrix=A_matrix, bs=bs, m=m,
                                         skip_connections=skip_connections)
            else:
                reco = self.reconstruct(history_features=history_features, A_matrix=A_matrix, bs=bs, m=m,
                                        skip_connections=skip_connections)
        output = dict()
        output['recon'] = reco['x'].view(x.shape)
        for key in encoding.keys():
            output[key] = encoding[key]
        output['features'] = reco['features']
        if self.config['delayed_dmd']:
            if (self.config['online_update']) and not (self.config['ddmdc']):
                if self.config['dtype'] == 'float':
                    output['A_matrix'] = reco['A_matrix'].float()
                elif self.config['dtype'] == 'double':
                    output['A_matrix'] = reco['A_matrix']
            elif self.config['ddmdc']:
                if self.config['dtype'] == 'float':
                    output['A_matrix'] = A_matrix.float()
                else:
                    output['A_matrix'] = A_matrix
            else:
                output['A_matrix'] = A_matrix
        else:
            output['A_matrix'] = A_matrix
        output['features_cycle'] = self.autoencoder.encode(output['recon'])['latent'].view((bs, m, self.n_reduced))
        if self.config['longterm_pen']:
            bs, m, c, h, w = x.shape
            output['penalized'] = reco['penalized'].reshape((bs, m - self.n_points_A, c, h, w))

        return output

    def reconstruct(self, history_features, A_matrix, bs, m, skip_connections=None):
        """
        Params
        ------
        history_features: tensor of shape (bs, n_reduced, n_points_A). encodings of the first n_points_A frames
        A_matrix: tensor of shape (bs, n_reduced, n_reduced)
        bs: int, batch size
        m: int, config['sample_duration']
        skip_connections: Not implemented yet - but should be a list
        Returns
        -------
        reco: dict.
            2 keys: 
                - features: reconstructed features (multiplied by A)
                - x: reconstructed of the input of forward
        """
        if self.config['reco_method'] == 'iter':
            all_features = torch.zeros((bs, self.n_reduced, m), device=A_matrix.device, dtype=A_matrix.dtype)
            all_features[:, :, :self.n_points_A] = history_features
            # missing frames
            if self.config['missing_states']:
                for k in self.config['missing_states']:
                    all_features[:, :, k] = torch.bmm(A_matrix, all_features[:, :, k-1].clone().view((bs, self.n_reduced, 1))).view((bs, self.n_reduced))
            # last frames
            for i in range(m - self.n_points_A):
                all_features[:, :, self.n_points_A + i] = (
                    torch.bmm(A_matrix, all_features[:, :, self.n_points_A + i - 1].clone().view((bs, self.n_reduced, 1))).view(
                        (bs, self.n_reduced)))
        elif self.config['reco_method'] == 'one_step':
            print("reco_method == one_step not verified yet")
            raise NotImplementedError

        # Prepare input to self.decode
        all_features_dict = dict()
        all_features_dict['latent'] = all_features.transpose(1, 2)  # of shape (bs, m, self.n_reduced)

        if skip_connections:
            print("Not implemented for skip connections")
            raise NotImplementedError
        x_reco = self.decode(all_features_dict)  # of shape (bs * m, c, h, w)
        reco = dict()
        reco['features'] = all_features_dict['latent']  # of shape (bs, m, self.n_reduced)
        reco['x'] = x_reco
        return reco

    def dreconstruct(self, history_features, A_aug_matrix, bs, m, skip_connections=None):
        """
        Params
        ------
        history_features: tensor of shape (bs, n_reduced, n_points_A). encodings of the first n_points_A frames
        A_matrix: tensor of shape (bs, n_reduced, n_reduced)
        bs: int, batch size
        m: int, config['sample_duration']
        skip_connections: Not implemented yet - but should be a list
        Returns
        -------
        reco: dict.
            2 keys:
                - features: reconstructed features (multiplied by A)
                - x: reconstructed of the input of forward
        """
        h = self.config['history']
        if self.config['reco_method'] == 'iter':
            all_features = torch.zeros((bs, self.n_reduced, m), device=history_features.device, dtype=history_features.dtype)
            all_features[:, :, :self.n_points_A] = history_features
            # missing frames
            if self.config['missing_states']:
                print("missing_states not implemented for delayed dmd")
                raise NotImplementedError
            # last frames
            for i in range(m - self.n_points_A):
                history_vector = torch.zeros((bs, h * self.n_reduced, 1), dtype=all_features.dtype, device=all_features.device)
                for t in range(h):
                    history_vector[:, t * self.n_reduced:(t + 1) * self.n_reduced, :] = all_features[:, :, self.n_points_A - h + i + t].view((bs, self.n_reduced, 1))
                all_features[:, :, self.n_points_A + i] = (
                    torch.bmm(A_aug_matrix, history_vector).view((bs, self.n_reduced)))

            # Prepare input to self.decode
            all_features_dict = dict()
            all_features_dict['latent'] = all_features.transpose(1, 2)  # of shape (bs, m, self.n_reduced)

        if skip_connections:
            print("Not implemented for skip connections")
            raise NotImplementedError

        x_reco = self.decode(all_features_dict)
        reco = dict()
        reco['features'] = all_features_dict['latent']  # of shape (bs, m, self.n_reduced)
        reco['x'] = x_reco
        return reco

    def reconstruct_with_control(self, history_features, U, A, B, bs, m):
        """
                Params
                ------
                history_features: tensor of shape (bs, n_reduced, n_points_A). encodings of the first n_points_A frames
                A_matrix: tensor of shape (bs, n_reduced, n_reduced)
                bs: int, batch size
                m: int, config['sample_duration']
                skip_connections: Not implemented yet - but should be a list
                Returns
                -------
                reco: dict.
                    2 keys:
                        - features: reconstructed features (multiplied by A)
                        - x: reconstructed of the input of forward
                """
        h = self.config['history']
        if self.config['reco_method'] == 'iter':
            all_features = torch.zeros((bs, self.n_reduced, m), device=history_features.device,
                                       dtype=history_features.dtype)
            all_features[:, :, :self.n_points_A] = history_features
            # missing frames
            if self.config['missing_states']:
                print("missing_states not implemented for delayed dmd")
                raise NotImplementedError
            # last frames
            for i in range(m - self.n_points_A):
                history_vector = torch.zeros((bs, h * self.n_reduced, 1), dtype=all_features.dtype,
                                             device=all_features.device)
                for t in range(h):
                    history_vector[:, t * self.n_reduced:(t + 1) * self.n_reduced, :] = all_features[:, :,
                                                                                        self.n_points_A - h + i + t].view(
                        (bs, self.n_reduced, 1))

                all_features[:, :, self.n_points_A + i] = (
                    torch.bmm(A, history_vector).view((bs, self.n_reduced))) + B.bmm(U[:, :, self.n_points_A + i].view((bs, self.config['control_d'], 1))).view((bs, self.n_reduced))
            # Prepare input to self.decode
            all_features_dict = dict()
            all_features_dict['latent'] = all_features.transpose(1, 2)  # of shape (bs, m, self.n_reduced)

        x_reco = self.decode(all_features_dict)
        reco = dict()
        reco['features'] = all_features_dict['latent']  # of shape (bs, m, self.n_reduced)
        reco['x'] = x_reco
        return reco

    def online_reconstruct(self, all_features, A_aug_matrix, skip_connections=None):
        h = self.config['history']

        # Compute A
        bs, n, m = all_features.shape
        reco_vectors = torch.zeros((bs, n, m), dtype=all_features.dtype, device=all_features.device)
        # Define Y_his
        Y_his = all_features[:, :, :self.n_points_A]  # Y_his has n_points_A columns, indexed from 0 to n_points_A - 1
        # Define Y_h and Yh from Y_his
        Y_h, Yh = build_shifted_matrices(Y=Y_his, h=h)
        # Verify that Y_h and Yh are correct
        # Between two row blocks of Y_h, the columns should be shifted.  OK
        reco_vectors[:, :, :self.n_points_A] = Y_his
        rho = 1 / self.config['lam_prox']
        # Compute P_b_inv for this Y_h and Yh
        P = Y_h @ Y_h.transpose(1, 2) + rho * batch_eye(n=h * n, bs=1, device=all_features.device)
        L = torch.cholesky(P)
        P_inv = torch.cholesky_solve(batch_eye(n=h * n, bs=1, device=all_features.device), L)

        # Compute A_prox with history
        A_prox = A_aug_matrix
        # Build history vector
        history_vector = torch.zeros((bs, n * h, 1), dtype=all_features.dtype, device=all_features.device)
        for r in range(h):
            history_vector[:, r * n:(r + 1) * n, :] = Y_his[:, :, - h + r].view((-1, n, 1))
        next_pred = A_prox @ history_vector

        reco_vectors[:, :, self.n_points_A] = next_pred.reshape((-1, n))
        if not self.config['online_update_horizon']:
            for i in range(self.n_points_A + 1, m):
                Y_his = all_features[:, :, :i]
                y1 = torch.zeros((bs, h * n, 1), dtype=all_features.dtype, device=all_features.device)
                for r in range(h):
                    y1[:, r * n:(r + 1) * n, :] = Y_his[:, :, i + r - h - 1].reshape((-1, n, 1))
                y2 = Y_his[:, :, i - 1].view((-1, n, 1))
                A_prox, P_inv = self.ddmd.update(A_start=A_prox, Y1=Y_h, Y2=Yh, y1=y1, y2=y2, P_inv_start=P_inv, h=h)
                Y_his = all_features[:, :, :i]
                # Build history vector
                history_vector = torch.zeros((bs, n * h, 1), dtype=all_features.dtype, device=all_features.device)
                for r in range(h):
                    history_vector[:, r * n:(r + 1) * n, :] = Y_his[:, :, - h + r].reshape((-1, n, 1))
                Y_h, Yh = build_shifted_matrices(Y=Y_his, h=h)
                reco_vectors[:, :, i] = (A_prox @ history_vector).reshape((-1, n))
        else:
            # Up until self.config['online_update_horizon'], update matrix and predict only next frame
            for i in range(self.n_points_A + 1, self.n_points_A + self.config['online_update_horizon']):
                Y_his = all_features[:, :, :i]
                y1 = torch.zeros((bs, h * n, 1), dtype=all_features.dtype, device=all_features.device)
                for r in range(h):
                    y1[:, r * n:(r + 1) * n, :] = Y_his[:, :, i + r - h - 1].reshape((-1, n, 1))
                y2 = Y_his[:, :, i - 1].view((-1, n, 1))
                A_prox, P_inv = self.ddmd.update(A_start=A_prox, Y1=Y_h, Y2=Yh, y1=y1, y2=y2, P_inv_start=P_inv, h=h)
                Y_his = all_features[:, :, :i]
                # Build history vector
                history_vector = torch.zeros((bs, n * h, 1), dtype=all_features.dtype, device=all_features.device)
                for r in range(h):
                    history_vector[:, r * n:(r + 1) * n, :] = Y_his[:, :, - h + r].reshape((-1, n, 1))
                Y_h, Yh = build_shifted_matrices(Y=Y_his, h=h)
                reco_vectors[:, :, i] = (A_prox @ history_vector).reshape((-1, n))
            # From self.config['online_update_horizon'], do recursive prediction without update, with the last obtained A
            for i in range(self.n_points_A + self.config['online_update_horizon'], m):
                history_vector = torch.zeros((bs, h * self.n_reduced, 1), dtype=all_features.dtype,
                                             device=all_features.device)
                for t in range(h):
                    history_vector[:, t * self.n_reduced:(t + 1) * self.n_reduced, :] = reco_vectors[:, :, i -h + t].view(
                        (bs, self.n_reduced, 1))
                reco_vectors[:, :, i] = (
                    torch.bmm(A_prox, history_vector).view((bs, self.n_reduced)))



        all_features_dict = dict()
        if self.config['dtype'] == 'float':
            all_features_dict['latent'] = reco_vectors.transpose(1, 2).float()  # of shape (bs, m, self.n_reduced)
        else:
            all_features_dict['latent'] = reco_vectors.transpose(1, 2)
        #import pdb;pdb.set_trace()
        x_reco = self.decode(all_features_dict)
        reco = dict()
        reco['features'] = all_features_dict['latent']  # of shape (bs, m, self.n_reduced) Advanced features
        reco['A_matrix'] = A_prox
        reco['x'] = x_reco
        return reco

    def online_reconstruct_k(self, all_features, A_aug_matrix, measured_states, skip_connections=None, U=None):

        # If idx of update frame is 25, it means we have acquired 25, 24, 23, ..., 25 - h (h+1 new measures).
        # If we don't use GT measures, A will be updated partially with wrong recos.
        # It could be tried also.

        bs, n, m = all_features.shape

        h = self.config['history']
        d = self.config['control_d']

        reco_vectors = torch.zeros((bs, n, m), dtype=all_features.dtype, device=all_features.device)
        if self.config['longterm_pen']:
            reco_vectors2 = torch.zeros((bs, n, m), dtype=all_features.dtype, device=all_features.device)

        # Define Y_h and Yh
        Y_h, Yh = build_shifted_matrices(Y=all_features[:, :, :self.n_points_A], h=h)
        n_aug = h * n + d
        Y_h_aug = torch.zeros((bs, n_aug, self.n_points_A - h), dtype=all_features.dtype, device=all_features.device)
        Y_h_aug[:, :h*n, :] = Y_h
        if d:
            Y_h_aug[:, h*n:, :] = U[:, :, h:self.n_points_A]
        else:
            U = torch.zeros((bs, n, self.n_points_A), dtype=all_features.dtype, device=all_features.device)
        Y_h = Y_h_aug

        reco_vectors[:, :, :self.n_points_A] = all_features[:, :, :self.n_points_A]
        if self.config['longterm_pen']:
            reco_vectors2[:, :, :self.n_points_A] = all_features[:, :, :self.n_points_A]
        updated_ids = []

        rho = 1 / self.config['lam_prox']
        # Compute A_prox with history
        if self.config['longterm_pen']:
            for i in range(self.n_points_A, m):
                history_vector = torch.zeros((bs, h * n + d, 1), dtype=all_features.dtype, device=all_features.device)
                for r in range(h):
                    history_vector[:, r * n:(r + 1) * n, :] = reco_vectors2[:, :, i - h + r].reshape((bs, n, 1))
                    # Add control input to history vector
                    if d:
                        history_vector[:, (r + 1) * n: (r + 1) * n + d, :] = U[:, :, i].reshape((bs, d, 1))
                reco_vectors2[:, :, i] = (A_aug_matrix @ history_vector).reshape((bs, n))

        for i in range(self.n_points_A, m):
            history_vector = torch.zeros((bs, n * h + d, 1), device=all_features.device, dtype=all_features.dtype)
            count_updated = 0
            for r in range(h):
                if i - h + r in updated_ids:
                    count_updated += 1

            for r in range(h):
                if i - h + r in updated_ids and count_updated == h:
                    history_vector[:, r * n:(r + 1) * n, :] = all_features[:, :, i - h + r].reshape((bs, n, 1))
                else:
                    history_vector[:, r * n:(r + 1) * n, :] = reco_vectors[:, :, i - h + r].reshape((-1, n, 1))
            if d:
                history_vector[:, (r + 1) * n: (r + 1) * n + d, :] = U[:, :, i].reshape((-1, d, 1))
            reco_vectors[:, :, i] = (A_aug_matrix @ history_vector).reshape((-1, n))

            if str(i) in measured_states or i in measured_states:
                y1 = torch.zeros((bs, h * n + d, 1), device=all_features.device, dtype=all_features.dtype)
                for r in range(h):
                    y1[:, r * n:(r + 1) * n, :] = all_features[:, :, i + r].reshape((-1, n, 1))
                if d:
                    y1[:, (r + 1) * n:(r + 1) * n + d, :] = U[:, :, i + h].reshape(
                        (-1, d, 1))  # the idx of U should be the one in y2

                # y2 will be the i + h th vector
                y2 = all_features[:, :, i + h]

                Y_h_temp = Y_h.clone()
                bs, hn_temp, m_temp = Y_h.shape
                Y_h = torch.zeros((bs, hn_temp, m_temp + 1), device=all_features.device, dtype=all_features.dtype)
                Y_h[:, :, :-1] = Y_h_temp
                for r in range(h):
                    Y_h[:, r * n:(r + 1) * n, -1] = y1[:, r * n:(r + 1) * n, :].reshape((-1, n))
                if d:
                    Y_h[:, (r + 1) * n: (r + 1) * n + d, -1] = y1[:, (r + 1) * n: (r + 1) * n + d, :].reshape((-1, d))
                Yh_temp = Yh.clone()
                bs, n_temp, m_temp = Yh_temp.shape
                Yh = torch.zeros((bs, n_temp, m_temp + 1), device=all_features.device, dtype=all_features.dtype)
                Yh[:, :, :-1] = Yh_temp
                Yh[:, :, -1] = y2
                # Recompute A using new measurements
                A_aug_matrix = estimate_from_history_matrices_pytorch(a_method=self.config['a_method'], n=n,
                                                                      n_aug=h * n + d, Y_h=Y_h, Yh=Yh, rho=rho,
                                                                      reg=self.config['reg'], m=m_temp + h + 1, h=h,
                                                                      d=d, return_P_inv=0,
                                                                      max_it=self.config['m_prox'], verbose=0, return_res=0)
                for r in range(h + 1):
                    updated_ids.append(i + r)

        all_features_dict = dict()
        if self.config['dtype'] == 'float':
            all_features_dict['latent'] = reco_vectors.transpose(1, 2).float()  # of shape (bs, m, self.n_reduced)
        elif self.config['dtype'] == 'double':
            all_features_dict['latent'] = reco_vectors.transpose(1, 2)

        if self.config['longterm_pen']:
            pen_dict = dict()
            if self.config['dtype'] == 'float':
                pen_dict['latent'] = reco_vectors2[:, :, self.n_points_A:].transpose(1, 2).float()
            elif self.config['dtype'] == 'double':
                pen_dict['latent'] = reco_vectors2[:, :, self.n_points_A:].transpose(1, 2)

            pen_reco = self.decode(pen_dict)
        x_reco = self.decode(all_features_dict)
        reco = dict()
        reco['features'] = all_features_dict['latent']  # of shape (bs, m, self.n_reduced) Advanced features
        reco['A_matrix'] = A_aug_matrix
        reco['x'] = x_reco
        if self.config['longterm_pen']:
            reco['penalized'] = pen_reco
        return reco

    def register_gradients(self):
        my_register_backward_hook(model=self, module=self.autoencoder.fc1, name='autoencoder.fc1')
        my_register_backward_hook(model=self, module=self.autoencoder.fc21, name='autoencoder.fc21')
        my_register_backward_hook(model=self, module=self.autoencoder.fc22, name='autoencoder.fc22')
        my_register_backward_hook(model=self, module=self.autoencoder.fc3, name='autoencoder.fc3')
        my_register_backward_hook(model=self, module=self.autoencoder.fc4, name='autoencoder.fc4')
        my_register_backward_hook(model=self, module=self.dmd, name='dmd')
        my_register_backward_hook(model=self, module=self.dmd.svd, name='dmd.svd')
        my_register_backward_hook(model=self, module=self.dmd_modes.eig, name='dmd_modes.eig')
        my_register_backward_hook(model=self, module=self.dmd_modes, name='dmd_modes')
        my_register_backward_hook(model=self, module=self.dmd_reco, name='dmd_reco')

    def return_gradients(self):
        return self.gradients