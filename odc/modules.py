import torch
import torch.nn as nn


from odc.model_utils import remove_state_from_pair, estimate_A, estimate_A_b_learned, batch_eye


class SVD(nn.Module):
    """
    Module version of torch.svd https://pytorch.org/docs/stable/generated/torch.svd.html
    """
    def __init__(self):
        super(SVD, self).__init__()

    def forward(self, input, some=True, compute_uv=True, out=None):
        """
        Params
        ------
        input: torch.Tensor
            Shape: (bs, n, m) or (n, m)
        some: bool
            If some is True (default), the method returns the reduced singular value decomposition
            i.e., if the last two dimensions of input are m and n, then the returned U and V matrices
            will contain only min(n, m)min(n,m) orthonormal columns.
        compute_uv: bool
            If compute_uv is False, the returned U and V matrices will be zero matrices of shape
            m \times m)(m×m) and (n \times n)(n×n) respectively. some will be ignored here.
        out:
        Returns
        -------
        U, S, V
        """
        return torch.svd(input, some=some, compute_uv=compute_uv, out=out)


class DMD(nn.Module):
    def __init__(self, a_method):
        super(DMD, self).__init__()
        if a_method == "svd":
            self.svd = SVD()

    def forward(self, input, r, return_USV=False, a_method='svd', a_return='reduced', m_prox=10, lam_prox=1e6,
                n_reduced=None, missing_states=0):
        """
        Params
        -------
        input: torch.Tensor
            shape: (bs, n, m). X data matrix
        r: int
            SVD truncation rank
        return_USV: bool
            whether or not to return U, S, V matrices (from SVD of X1 = X[:, :, :-1])
            See returns
        Returns
        -------
        If return_USV:
            A_tilde, U_rt, S_r_inv, V_r
        else:
            A_tilde
        """
        bs, _, _ = input.shape
        if a_method == 'prox':
            X1 = input[:, :, :-1].double()
            X2 = input[:, :, 1:].double()
        else:
            X1 = input[:, :, :-1]
            X2 = input[:, :, 1:]
        if missing_states:
            for k in missing_states:
                # k = missing_states[i]
                # X1 = torch.cat((X1[:, :, :k-1], X1[:, :, k+1:]), dim=-1)  # we remove x_k-1 and x_k.
                # X2 = torch.cat((X2[:, :, :k-1], X2[:, :, k+1:]), dim=-1) # we remove x_k and x_k+1
                X1, X2 = remove_state_from_pair(k=k, old_X1=X1, old_X2=X2)
            # X1 = X[:, :, :-1]
            # X2 = X[:, :, 1:]
        if a_method == 'pinv':
            A = X2.bmm(torch.pinverse(X1))
            return A
        elif a_method == 'still':
            return batch_eye(n=n_reduced, bs=bs, device=input.device, dtype=input.dtype)
        elif a_method == 'qr':
            Q, R = torch.qr(X1)
            s = torch.min(torch.stack((torch.tensor(X1.shape[1]), torch.tensor(X1.shape[2]))))
            A = X2.bmm((R + 1e-6 * batch_eye(n=s, bs=bs, device=input.device, dtype=input.dtype)).inverse().bmm(
                Q.transpose(1, 2)))
            return A
        elif a_method == 'svd':
            U, S, V = self.svd(X1)  # the returned V is the V in X1 = USV*
            U_r = U[:, :, :r]
            S_r = S[:, :r]
            S_r_inv = 1 / (S_r + 1e-6)
            S_r_inv = torch.diag_embed(S_r_inv, dim1=-1, dim2=1)
            V_r = V[:, :, :r]
            U_rt = U_r.permute(0, 2, 1)
            A_tilde = U_rt.bmm(X2.bmm(V_r.bmm(S_r_inv)))
            A_full = U_r.bmm(A_tilde).bmm(U_rt)
            if a_return == 'reduced':
                A = A_tilde
            elif a_return == 'full':
                A = A_full

            if return_USV:
                return A, U_rt, S_r_inv, V_r
            else:
                return A
        elif a_method == 'prox':
            rho = 1 / lam_prox
            A0 = batch_eye(bs=bs, n=n_reduced, dtype=torch.double, device=X1.device)
            A_t_s = []
            A_t_s.append(A0)
            P_b = X1 @ X1.transpose(1, 2) + rho * batch_eye(bs=bs, n=n_reduced, dtype=torch.double, device=X1.device)
            L_b_sp = torch.cholesky(P_b)
            for k in range(m_prox):
                A_ = A_t_s[-1] + torch.cholesky_solve(X1 @ (X2.transpose(1, 2) - X1.transpose(1, 2) @ A_t_s[-1]),
                                                      L_b_sp)
                A_t_s.append(A_)

            return A_t_s[-1].transpose(1, 2).float()


class DelayedDMD(nn.Module):
    def __init__(self, a_method):
        super(DelayedDMD, self).__init__()
        if a_method == "svd":
            self.svd = SVD()

    def forward(self, input, r, B=None, U=None, d=None, return_USV=False, a_method='prox', a_return='reduced', m_prox=10, lam_prox=1e8,
                n_reduced=None, missing_states=0, history=2, a_dtype='float', A_reg=1e-8):
        """
        Params
        -------
        input: torch.Tensor
            shape: (bs, n, m). X data matrix
        return_USV: bool
            whether or not to return U, S, V matrices (from SVD of X1 = X[:, :, :-1])
            See returns
        Returns
        -------
        If return_USV:
            A_tilde, U_rt, S_r_inv, V_r
        else:
            A of shape (bs, history * n_reduced, history * n_reduced)
        """
        bs, n, m = input.shape

        rho = 1 / lam_prox
        if not B:
            A = estimate_A(X=input, rho=rho, reg=A_reg, max_it=m_prox, h=history, a_dtype=a_dtype, a_method=a_method,
                       missing_states=missing_states)
        else:
            A = estimate_A_b_learned(X=input, B=B, U=U, d=d, rho=rho, reg=A_reg, max_it=m_prox, h=history, a_dtype=a_dtype,
                                     a_method=a_method,
                       missing_states=missing_states)
        return A

    def update(self, A_start, Y1, Y2, y1, y2, P_inv_start, h, m_prox=10):
        """
            Initialize Am+1,0 with A_start (last Am)
            You need Y1 and Y2 of previous step, + y1 and y2 + P_inv_start
        """

        A_t_0 = A_start.transpose(1, 2)
        A_t_s = []
        A_t_s.append(A_t_0)
        for k in range(m_prox):
            A_ = A_t_s[-1] + P_inv_start @ (Y1 @ (Y2.transpose(1, 2) - Y1.transpose(1, 2) @ A_t_s[-1])) + y1 @ \
                 (y2.transpose(1, 2) - y1.transpose(1, 2) @ (
                             A_t_s[-1] + P_inv_start @ (Y1 @ (Y2.transpose(1, 2) - Y1.transpose(1, 2) @ A_t_s[-1])))) / \
                 (1 + y1.transpose(1, 2) @ P_inv_start @ y1)
            A_t_s.append(A_)
        P_inv_new = P_inv_start - P_inv_start @ y1 @ y1.transpose(1, 2) @ P_inv_start \
                    / (1 + y1.transpose(1, 2) @ P_inv_start @ y1)
        return A_t_s[-1].transpose(1, 2), P_inv_new


class DelayedDMDc(DelayedDMD):
    def __init__(self, a_method):
        DelayedDMD.__init__(self, a_method)

    def forward(self, input, r, d=0, U=None, return_USV=False, a_method='prox', a_return='reduced', m_prox=10, lam_prox=1e8,
                A_reg=1e-8, n_reduced=None, missing_states=0, history=2, a_dtype='float'):
        """
               Params
               -------
               input: torch.Tensor
                   shape: (bs, n, m). X data matrix

                U: torch.Tensor
                    shape: (bs, d, m-h). Matrix of control inputs

                d: torch.Tensor

               return_USV: bool
                   whether or not to return U, S, V matrices (from SVD of X1 = X[:, :, :-1])
                   See returns
               Returns
               -------
               If return_USV:
                   A_tilde, U_rt, S_r_inv, V_r
               else:
                   A of shape (bs, history * n_reduced, history * n_reduced)
               """
        rho = 1 / lam_prox
        A = estimate_A(X=input, rho=rho, reg=A_reg, max_it=m_prox, h=history, d=d, U=U, a_dtype=a_dtype,
                       a_method=a_method, missing_states=missing_states)
        return A

    def update(self, A_start, Y1, Y2, y1, y2, P_inv_start, h, m_prox=10):
        print("Update not implemented for DMDc")
        raise NotImplementedError