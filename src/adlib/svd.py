'''
PyTorch has its own implementation of backward function for SVD https://github.com/pytorch/pytorch/blob/291746f11047361100102577ce7d1cfa1833be50/tools/autograd/templates/Functions.cpp#L1577 
We reimplement it with a safe inverse function in light of degenerated singular values
'''

import numpy as np
import torch




def randomized_svd(A, rank, n_oversamples=5, n_iter=4):
    """
    Approximate SVD using randomized algorithm.
    A: matrix (m x n)
    rank: target rank k (number of singular vectors/values to compute)
    """
    m, n = A.shape


    # Step 1: Random projection
    P = torch.randn(n, rank + n_oversamples, device=A.device, dtype = A.dtype)
    Z = A @ P  # (m x (k+p))

    # Step 2: Power iterations (optional but improves accuracy)
    for _ in range(n_iter):
        Z = A @ (A.T @ Z)

    # Step 3: Orthonormal basis via QR
    Q, _ = torch.linalg.qr(Z)

    # Step 4: Project A to smaller space
    B = Q.T @ A  # (k+p) x n

    # Step 5: SVD in small space
    try:
        U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    except RuntimeError as e:
        print("SVD failed:", e)
        eps = 1e-10
        B = B + eps * torch.randn_like(B)
        U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)

    U = Q @ U_hat  # back to original space

    return U[:, :rank], S[:rank], Vh[:rank, :]


def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):

        # size = A.shape
        # U, S, V = randomized_svd(A, size[0], n_oversamples=size[0], n_iter=4)
        # print(U.shape)
        try:
            U, S, V = torch.svd(A)
        except RuntimeError as e:
            U, S, V = torch.svd(A + 1e-15)
            print("SVD failed:", e)
            print(A)

        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G.diagonal().fill_(np.inf)
        G = 1/G 

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        return dA

