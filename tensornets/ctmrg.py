import torch
from torch.utils.checkpoint import checkpoint

from variational_iPEPS import get_obs_honeycomb

from .adlib import SVD 
svd = SVD.apply



def renormalize_honeycomb(*tensors):

    # C(d,r), EL(u,r,d), EU(l,d,r)

    C, Ea, Eb, Ta, Tb, chi = tensors 

    #    C -f--- Ea -- a
    #    |e      |g
    #    |       Tb -- p
    #    |      / l  
    #    Eb-j-Ta 
    #    |    |
    #    i    k

    dimEa3 = Ea.shape[2] # a
    dimEb1 = Eb.shape[0] # i

    dimT = Ta.shape[0]
    D_new = min(dimEb1*dimT, chi) 

    CEETT = torch.einsum('ije,ef,fga,jlk,lgp->ikap',(Eb,C,Ea,Ta,Tb))
    CEETT = CEETT.reshape(dimEb1*dimT, dimEa3*dimT)  # Rho(i,a,k,p) => Rho(i,k,a,p) => Rho(ki, pa)
    CEETT = CEETT/CEETT.norm()

    # U, S, V = svd(CEETT@CEETT@CEETT) 
    U, S, V = svd(CEETT) 
   
    truncation_error = S[D_new:].sum()/S.sum()
    P = U[:, :D_new] 
    C = P.t()@CEETT@P

    

    # i--Ea -k- |
    #    | j     P - q   = Eb(ilq)
    #    Tb -m- |
    #   / l

    #   | -i- Eb -- k 
    # q - P   |j         = Ea(qmk)
    #   | -l- Ta 
    #           \ m

    P = P.view(dimEb1, dimT, D_new)       # P(i,k,D_new), chi d chi_new
    Ebtmp = torch.einsum('ijk,ljm,kmq->ilq',(Ea,Tb,torch.conj(P)))
    Eatmp = torch.einsum('ijk,ljm,ilq->qmk',(Eb,Ta,P))

    Eb = Ebtmp.clone()
    Ea = Eatmp.clone()

    C = C/C.norm() 
    Ea = Ea/Ea.norm()
    Eb = Eb/Eb.norm()

    return C, Ea, Eb, S/S.norm(), truncation_error



def CTMRG_honeycomb(Ta, Tb, H, M, A1symm, A2symm, chi, max_iter, dtype, use_checkpoint=False):

    threshold = 1E-7 

    C0 = torch.ones((1,1), dtype=dtype, device= Ta.device)
    Ea0 = torch.ones((1,1,1),dtype=dtype, device= Ta.device)
    Eb0 = torch.ones((1,1,1),dtype=dtype, device= Ta.device)
    C1 = torch.ones((1,1), dtype=dtype, device= Ta.device)
    Ea1 = torch.ones((1,1,1),dtype=dtype, device= Ta.device)
    Eb1 = torch.ones((1,1,1),dtype=dtype, device= Ta.device)
    C2 = torch.ones((1,1), dtype=dtype, device= Ta.device)
    Ea2 = torch.ones((1,1,1),dtype=dtype, device= Ta.device)
    Eb2 = torch.ones((1,1,1),dtype=dtype, device= Ta.device)

    diff = 1E1
    ener = 0
    ener1 = 0
    ener2 = 0


    tensors0 = C0, Ea0, Eb0, Ta, Tb, torch.tensor(chi)

    for n in range(max_iter):

        Etmp = ener + ener1 + ener2
        ener, enerf, Mx, My, Mz = get_obs_honeycomb(A1symm, A2symm, H[0], M[0], M[1], M[2], C0, Ea0, Eb0)
        ener1, enerf, Mx, My, Mz = get_obs_honeycomb(A1symm, A2symm, H[0], M[0], M[1], M[2], C1, Ea1, Eb1)
        ener2, enerf, Mx, My, Mz = get_obs_honeycomb(A1symm, A2symm, H[0], M[0], M[1], M[2], C2, Ea2, Eb2)
        
        diff = abs(ener + ener1 + ener2 - Etmp)
        print(diff)
        if use_checkpoint: # use checkpoint to save memory 
            C, Ea, Eb, s, error = checkpoint(renormalize_honeycomb, *tensors0) 
        else:
            C1, Ea1, Eb1, s, error = renormalize_honeycomb(*tensors0)
            tensors1 = C1, Ea1, Eb1, Ta, Tb, torch.tensor(chi)
            C2, Ea2, Eb2, s, error = renormalize_honeycomb(*tensors1)
            tensors2 = C2, Ea2, Eb2, Ta, Tb, torch.tensor(chi)
            C0, Ea0, Eb0, s, error = renormalize_honeycomb(*tensors2)
            tensors0 = C0, Ea0, Eb0, Ta, Tb, torch.tensor(chi)


        if (diff < threshold):
            break
        if n == max_iter-1:
            print('ctm not converged')


    return C0, Ea0, Eb0, C1, Ea1, Eb1, C2, Ea2, Eb2


