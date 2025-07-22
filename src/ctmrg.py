import torch
from src.utils import save_checkpoint
from src.measure import get_obs_honeycomb
from src.measure import get_energy_pess
from src.adlib import SVD 
from src.adlib import EigenSolver

svd = SVD.apply
eig = EigenSolver.apply

def renormalize_honeycomb(*tensors):

    """
        Performs one iteration of the C3-CTMRG alogorithm

        Args:  
            - *tensors: environment tensors, and the local ones. 
    """

    C, Ea, Eb, Ta, Tb, chi = tensors 

    """
       C -f--- Ea -- a
       |e      |g
       |       Tb -- p
       |      / l  
       Eb-j-Ta 
       |    |
       i    k
    """
    dimEa3 = Ea.shape[2] # a
    dimEb1 = Eb.shape[0] # i

    dimT = Ta.shape[0]
    D_new = min(dimEb1*dimT, chi) 

    CEETT = torch.einsum('ije,ef,fga,jlk,lgp->ikap',(Eb,C,Ea,Ta,Tb))
    CEETT = CEETT.reshape(dimEb1*dimT, dimEa3*dimT)  # Rho(i,a,k,p) => Rho(i,k,a,p) => Rho(ki, pa)
    CEETT = CEETT/CEETT.norm()

    U, S, V = svd(CEETT) 
    truncation_error = S[D_new:].sum()/S.sum() 
    P = U[:, :D_new] 

    C = P.t()@CEETT@P


    """
    i--Ea -k- |
       | j     P - q   = Eb(ilq)
       Tb -m- |
      / l

      | -i- Eb -- k 
    q - P   |j         = Ea(qmk)
      | -l- Ta 
              \ m
    """

    P = P.view(dimEb1, dimT, D_new)                     # P(i,k,D_new), chi d chi_new

    Ebtmp = torch.einsum('ijk,ljm,kmq->ilq',(Ea, Tb, torch.conj(P)))
    Eatmp = torch.einsum('ijk,ljm,ilq->qmk',(Eb, Ta, P))

    Eb = Ebtmp.clone()
    Ea = Eatmp.clone()
    
    C = C/C.max() 
    Ea = Ea/Ea.max()
    Eb = Eb/Eb.max()

    return C, Ea, Eb, S/S.norm(), truncation_error

def CTMRG_honeycomb(Ta, Tb, H, M, A1symm, A2symm, chi, max_iter, dtype, use_checkpoint=False):


    """
        Computes the contraction of the 2D tensor network on the honeycomb lattice, for iPEPS wavefunctions

        Args: 
            - Ta & Tb : local tensors = < A1symm | A1symm > & < A2symm | A2symm >
            - H : energy to minimise
            - M : Mx, My, Mz
            - A1symm & A2symm : local tensors
    """

    threshold = 1E-7 

    C0 = torch.ones((1,1), dtype=dtype, device= Ta.device)
    Ea0 = torch.ones((1,1,1),dtype=dtype, device= Ta.device)
    Eb0 = torch.ones((1,1,1),dtype=dtype, device= Ta.device)
   
    diff = 1E1
    ener = 0
    ener1 = 0
    ener2 = 0


    tensors0 = C0, Ea0, Eb0, Ta, Tb, torch.tensor(chi)

    for n in range(max_iter):

        Etmp = ener + ener1 + ener2
        ener, enerf, Mx, My, Mz = get_obs_honeycomb(A1symm, A2symm, H[0], M[0], M[1], M[2], C0, Ea0, Eb0)
        diff = abs(ener + ener1 + ener2 - Etmp)

        if use_checkpoint: # use checkpoint to save memory 
            C, Ea, Eb, s, error = save_checkpoint(renormalize_honeycomb, *tensors0) 
        else:
            C0, Ea0, Eb0, s, error = renormalize_honeycomb(*tensors0)
            tensors0 = C0, Ea0, Eb0, Ta, Tb, torch.tensor(chi)


        if (diff < threshold):
            break
        if n == max_iter-1:
            print('ctm not converged')


    return C0, Ea0, Eb0

def CTMRG_honeycomb_pess(Ta, Tb, H, M, A1symm, A2symm, chi, max_iter, dtype, use_checkpoint=False):


    """
        Computes the contraction of the 2D tensor network on the honeycomb lattice, for PESS wavefunctions

        Args: 
            - Ta & Tb : local tensors = < A1symm | A1symm > & < A2symm | A2symm >
            - H : energy to minimise
            - M : Mx, My, Mz
            - A1symm & A2symm : local tensors
    """


    threshold = 1E-7 
    D2 = Ta.shape[0]
    C = torch.ones((1,1), dtype=dtype, device= Ta.device)
    Ea = torch.ones((1,D2,1),dtype=dtype, device= Ta.device)
    Eb = torch.ones((1,D2,1),dtype=dtype, device= Ta.device)
    
    Mpx, Mpy, Mpz = M
    diff = 1E1
    ener = 0
    tensors = C, Ea, Eb, Ta, Tb, torch.tensor(chi)

    for n in range(max_iter):

        Etmp = ener 
        ener, mx, my, mz = get_energy_pess(A1symm, A2symm, H, Mpx, Mpy, Mpz, C, Ea, Eb)
        
        # print(f'energy: {ener}')
        diff = abs(ener - Etmp)
        C, Ea, Eb, s, error = renormalize_honeycomb(*tensors)
        tensors = C, Ea, Eb, Ta, Tb, torch.tensor(chi)
            
        if (diff < threshold):
            break
        if n == max_iter-1:
            print('ctm not converged')


    return C, Ea, Eb


