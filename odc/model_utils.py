import torch
import numpy as np
import time


def batch_eye(n, bs, device=None, dtype=torch.double):
    x = torch.eye(n, dtype=dtype, device=device)
    x = x.reshape((1, n, n))
    y = x.repeat(bs, 1, 1)

    return y


def get_time(t):
    year = str(time.localtime(t).tm_year)
    month = str(time.localtime(t).tm_mon)
    day = str(time.localtime(t).tm_mday)
    hour = str(time.localtime(t).tm_hour)
    minute = str(time.localtime(t).tm_min)
    formatted_t = year + '_' + month + '_' + day + '_' + hour + '_' + minute
    return formatted_t


def estimate_A_prox_pytorch(n, n_aug, Y_h, Yh, rho, h, d,  return_P_inv, max_it, verbose,return_res=0):
    bs, _, _ = Y_h.shape
    A_t_s = []
    # Initialize A with identity, B with zeros
    A0 = torch.zeros((bs, n, n_aug), dtype=Yh.dtype, device=Yh.device)
    for i in range(h):
        A0[:,:, i * n:(i + 1) * n] = batch_eye(n=n, bs=bs, dtype=Yh.dtype, device=Yh.device)
    if d:
        A0[:, :,  -d:] = torch.zeros((bs, n, d), dtype=Yh.dtype, device=Yh.device)  # Initialize B component

    A_t_s.append(A0.transpose(1, 2))
    rho = rho

    P_b = Y_h@(Y_h.transpose(1, 2)) + rho * batch_eye(n=n_aug, bs=bs, dtype=Yh.dtype, device=Yh.device)
    L_b = torch.cholesky(P_b)
    if return_P_inv:
        P_inv = torch.cholesky_solve(batch_eye(n=h*n, bs=bs, dtype=Yh.dtype, device=Yh.device), L_b)
    for k in range(max_it):
        res = torch.norm(A_t_s[-1].transpose(1, 2) @ Y_h - Yh, np.inf)
        A_ = A_t_s[-1] + torch.cholesky_solve(Y_h.bmm(Yh.transpose(1, 2) - Y_h.transpose(1, 2).bmm(A_t_s[-1])), L_b)
        A_t_s.append(A_)
    A = A_t_s[-1].transpose(1, 2)
    if return_res:
        return A, res
    else:
        return A


def estimate_A_lag_pytorch(n, n_aug, Y_h, Yh, rho, h, d, m, max_it, reg, verbose,return_res=0):
    bs, _, _ = Y_h.shape
    H = torch.zeros((bs, n_aug + (m - h), n_aug + (m - h)), dtype=Yh.dtype, device=Yh.device)
    H[:, :n_aug, :n_aug] = reg * batch_eye(bs=bs, n=n_aug, dtype=Yh.dtype, device=Yh.device)
    H[:, n_aug:, :n_aug] = Y_h.transpose(1, 2)
    H[:, :n_aug, n_aug:] = Y_h
    H[:, n_aug:, n_aug:] = - rho * batch_eye(n=m - h, bs=bs, dtype=Yh.dtype, device=Yh.device)
    H_inv = torch.inverse(H)
    Ats = []
    Yts = []
    At = torch.zeros((bs, n_aug, n), dtype=Yh.dtype, device=Yh.device)
    Yt = torch.zeros((bs, m - h, n), dtype=Yh.dtype, device=Yh.device)
    Ats.append(At)
    Yts.append(Yt)
    for i in range(max_it):
        if verbose:
            res = torch.norm(Ats[-1].transpose(1, 2) @ Y_h - Yh, np.inf)
            print("res: {}".format(res))
        rhs = torch.zeros((bs, n_aug + m - h, n), dtype=Yh.dtype, device=Yh.device)
        rhs[:, :n_aug, :] = -Y_h @ Yts[-1] - reg * Ats[-1]
        rhs[:, n_aug:, :] = Yh.transpose(1, 2) - Y_h.transpose(1, 2) @ Ats[-1]
        dZ = H_inv @ rhs
        Ats.append(Ats[-1] + dZ[:, :h * n + d, :])
        Yts.append(Yts[-1] + dZ[:, h * n + d:, :])
    A = Ats[-1].transpose(1, 2)
    if return_res:
        return A, res
    else:
        return A


def estimate_from_history_matrices_pytorch(a_method, n, n_aug, Y_h, Yh, rho, reg, m, h, d,  return_P_inv, max_it,
                                   verbose,return_res=0):
    if a_method == 'prox':
        return estimate_A_prox_pytorch(Y_h=Y_h, Yh=Yh, n=n, n_aug=n_aug, rho=rho, verbose=verbose,
                                  return_P_inv=return_P_inv, h=h, d=d, max_it=max_it)
    elif a_method == 'lagrangian':
        return estimate_A_lag_pytorch(Y_h=Y_h, Yh=Yh, n=n, n_aug=n_aug, rho=rho, verbose=verbose,
                                     return_P_inv=return_P_inv, h=h, d=d, max_it=max_it, m=m, reg=reg)


def build_shifted_matrices(Y, h):
    bs, n, T = Y.shape
    Y_h = torch.zeros((bs, h * n, T - h), dtype=Y.dtype, device=Y.device)
    for i in range(h):
        Y_h[:, i * n:(i + 1) * n, :] = Y[:, :, i:T - h + i]
    Yh = Y[:, :, h:T]

    return Y_h, Yh


def update_shifted_matrices(Yh, Y_h, h, n, new_measures):
    """
    Update shifted matrices with h+1 new vectors
    """
    bs, n_, m_ = Yh.shape
    new_Yh = torch.zeros((bs, n_, m_ + 1), dtype=new_measures.dtype, device=new_measures.device)
    new_Yh[:, :, :-1] = Yh
    bs, n_, m_ = Y_h.shape
    new_Y_h = torch.zeros((bs, n_, m_ + 1), dtype=new_measures.dtype, device=new_measures.device)
    new_Y_h[:, :, :-1] = Y_h
    for r in range(h):
        new_Y_h[:, r * n:(r + 1) * n, -1] = new_measures[:, :, r]
    new_Yh[:, :, -1] = new_measures[:, :, -1]

    return new_Y_h, new_Yh

def build_shifted_matrices(Y, h):
    bs, n, T = Y.shape
    Y_h = torch.zeros((bs, h * n, T - h), dtype=torch.double, device=Y.device)
    for i in range(h):
        Y_h[:, i * n:(i + 1) * n, :] = Y[:, :, i:T - h + i]
    Yh = Y[:, :, h:T]

    return Y_h, Yh


def estimate_A(X, rho, reg, max_it=10, h=2, d=0, U=None, a_dtype='double', a_method='prox', missing_states=[]):
    """

    :param X:
    :param rho:
    :param reg:
    :param max_it:
    :param h: history
    :param d: Dimension of control. If 0, it means no control
    :param U: Control matrix. If d=0, will not be considered
    :param a_dtype:
    :param a_method:
    :return:
    """
    bs, n, m = X.shape
    n_aug = h * n + d  # Dimension of augmented state
    # Build Y_h and Yh matrices
    Y_h, Yh = build_shifted_matrices(Y=X, h=h)
    Y_h_aug = torch.zeros((bs, n_aug, m - h), dtype=torch.double, device=X.device)
    Y_h_aug[:, :h * n, :] = Y_h
    if d:
        Y_h_aug[:, h * n:, :] = U[:, :, h:m]
    Y_h = Y_h_aug

    # Solve using proximal lagrangien
    if a_method == 'lagrangian':
        H = torch.zeros((bs, n_aug+ (m - h), n_aug + (m - h)), dtype=torch.double, device=X.device)
        H[:, :n_aug, :n_aug] = reg * batch_eye(bs=bs, n=n_aug, dtype=torch.double, device=X.device)
        H[:, n_aug:, :n_aug] = Y_h.transpose(1, 2)
        H[:, :n_aug, n_aug:] = Y_h
        H[:, n_aug:, n_aug:] = - rho * batch_eye(n=m - h, bs=bs, dtype=torch.double, device=X.device)
        H_inv = torch.inverse(H)
        Ats = []
        Yts = []
        At = torch.zeros((bs, n_aug, n), dtype=torch.double, device=X.device)
        Yt = torch.zeros((bs, m - h, n), dtype=torch.double, device=X.device)
        Ats.append(At)
        Yts.append(Yt)
        for i in range(max_it):
            rhs = torch.zeros((bs, n_aug + m - h, n), dtype=torch.double, device=X.device)
            rhs[:, :n_aug, :] = -Y_h @ Yts[-1] - reg * Ats[-1]
            rhs[:, n_aug:, :] = Yh.transpose(1, 2) - Y_h.transpose(1, 2) @ Ats[-1]
            dZ = H_inv @ rhs
            Ats.append(Ats[-1] + dZ[:, :h * n + d, :])
            Yts.append(Yts[-1] + dZ[:, h * n + d:, :])
        A = Ats[-1].transpose(1, 2)

    # Solve using proximal
    elif a_method == 'prox':
        A_t_s = []
        # Initialize A with identity, B with zeros
        A0 = torch.zeros((bs, n, n_aug), dtype=torch.double, device=X.device)
        for i in range(h):
            A0[:, :, i * n:(i + 1) * n] = batch_eye(bs=bs, n=n, dtype=torch.double, device=X.device)
        if d:
            A0[:, :, -d:] = torch.zeros((bs, n, d), dtype=torch.double, device=X.device)  # Initialize B component
        Yh = X[:, :, h:]
        A_t_s.append(A0.transpose(1, 2))
        rho = rho
        P_b = Y_h.bmm(Y_h.transpose(1, 2)) + rho * batch_eye(bs=bs, n=n_aug, dtype=torch.double,
                                                             device=X.device)
        L_b = torch.cholesky(P_b)
        for k in range(max_it):
            A_ = A_t_s[-1] + torch.cholesky_solve(Y_h.bmm(Yh.transpose(1, 2) - Y_h.transpose(1, 2).bmm(A_t_s[-1])), L_b)
            A_t_s.append(A_)
        A = A_t_s[-1].transpose(1, 2)

    if a_dtype == 'float':
        return A.float()
    elif a_dtype == 'double':
        return A


def estimate_A_b_learned(X, B, rho, reg, max_it=10, h=2, d=0, U=None, a_dtype='double', a_method='prox', missing_states=[]):
    """

    :param X:
    :param rho:
    :param reg:
    :param max_it:
    :param h: history
    :param d: Dimension of control. If 0, it means no control
    :param U: Control matrix. If d=0, will not be considered
    :param a_dtype:
    :param a_method:
    :return:
    """
    bs, n, m = X.shape
    #n_aug = h * n + d  # Dimension of augmented state
    n_aug = h*n
    # Build Y_h and Yh matrices
    Y_h, Yh = build_shifted_matrices(Y=X, h=h)
    """Y_h_aug = torch.zeros((bs, n_aug, m - h), dtype=torch.double, device=X.device)
    Y_h_aug[:, :h * n, :] = Y_h
    if d:
        Y_h_aug[:, h * n:, :] = U[:, :, h:m]"""
    Yh -= B@U[:, :, h:m]

    # Solve using proximal lagrangien
    if a_method == 'lagrangian':
        H = torch.zeros((bs, n_aug+ (m - h), n_aug + (m - h)), dtype=torch.double, device=X.device)
        H[:, :n_aug, :n_aug] = reg * batch_eye(bs=bs, n=n_aug, dtype=torch.double, device=X.device)
        H[:, n_aug:, :n_aug] = Y_h.transpose(1, 2)
        H[:, :n_aug, n_aug:] = Y_h
        H[:, n_aug:, n_aug:] = - rho * batch_eye(n=m - h, bs=bs, dtype=torch.double, device=X.device)
        H_inv = torch.inverse(H)
        Ats = []
        Yts = []
        At = torch.zeros((bs, n_aug, n), dtype=torch.double, device=X.device)
        Yt = torch.zeros((bs, m - h, n), dtype=torch.double, device=X.device)
        Ats.append(At)
        Yts.append(Yt)
        for i in range(max_it):
            rhs = torch.zeros((bs, n_aug + m - h, n), dtype=torch.double, device=X.device)
            rhs[:, :n_aug, :] = -Y_h @ Yts[-1] - reg * Ats[-1]
            rhs[:, n_aug:, :] = Yh.transpose(1, 2) - Y_h.transpose(1, 2) @ Ats[-1]
            dZ = H_inv @ rhs
            Ats.append(Ats[-1] + dZ[:, :h * n + d, :])
            Yts.append(Yts[-1] + dZ[:, h * n + d:, :])
        A = Ats[-1].transpose(1, 2)

    # Solve using proximal
    elif a_method == 'prox':
        A_t_s = []
        # Initialize A with identity, B with zeros
        A0 = torch.zeros((bs, n, n_aug), dtype=torch.double, device=X.device)
        for i in range(h):
            A0[:, :, i * n:(i + 1) * n] = batch_eye(bs=bs, n=n, dtype=torch.double, device=X.device)
        #if d:
        #    A0[:, :, -d:] = torch.zeros((bs, n, d), dtype=torch.double, device=X.device)  # Initialize B component
        Yh = X[:, :, h:]
        A_t_s.append(A0.transpose(1, 2))
        rho = rho
        P_b = Y_h.bmm(Y_h.transpose(1, 2)) + rho * batch_eye(bs=bs, n=n_aug, dtype=torch.double,
                                                             device=X.device)
        L_b = torch.cholesky(P_b)
        for k in range(max_it):
            A_ = A_t_s[-1] + torch.cholesky_solve(Y_h.bmm(Yh.transpose(1, 2) - Y_h.transpose(1, 2).bmm(A_t_s[-1])), L_b)
            A_t_s.append(A_)
        A = A_t_s[-1].transpose(1, 2)

    A_matrix = torch.zeros((bs, n, h*n + d), dtype=A.dtype, device=A.device)
    A_matrix[:, :, :h*n] = A
    A_matrix[:, :, h*n:] = B

    if a_dtype == 'float':
        return A_matrix.float()
    elif a_dtype == 'double':
        return A_matrix

def remove_state_from_pair(k, old_X1, old_X2):
    bs, n, m = old_X1.shape
    X1 = torch.zeros((bs, n, m - 2), dtype=old_X1.dtype, device=old_X1.device)
    X2 = torch.zeros((bs, n, m - 2), dtype=old_X2.dtype, device=old_X2.device)
    X1[:, :, :k - 1] = old_X1[:, :, :k - 1]
    X1[:, :, k - 1:] = old_X1[:, :, k + 1:]
    X2[:, :, :k - 1] = old_X2[:, :, :k - 1]
    X2[:, :, k - 1:] = old_X2[:, :, k + 1:]
    return X1, X2


def remove_missing_states(k, old_matrices, h, n):
    """
    :param k:
    :param old_matrices: list of F1, ... Fh+1
    :param h:
    :return:
    """
    bs, _, m = old_matrices.shape
    new_matrices = torch.zeros((bs, (h+1)*n, m - h - 1), dtype=old_matrices[0].dtype, device=old_matrices[0].device)
    new_matrices[:, :, :k - h] = old_matrices[:, :, :k - h]
    new_matrices[:, :, k - h:] = old_matrices[:, :, k + 1:]
    return new_matrices