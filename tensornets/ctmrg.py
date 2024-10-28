import torch
from torch.utils.checkpoint import checkpoint

from variational_iPEPS import get_obs_honeycomb

from .adlib import SVD 
svd = SVD.apply
#from .adlib import EigenSolver
#symeig = EigenSolver.apply

def renormalize_honeycomb(*tensors):
    # T(up,left,down,right), u=up, l=left, d=down, r=right
    # C(d,r), EL(u,r,d), EU(l,d,r)

    C, Ea, Eb, Ta, Tb, chi = tensors 

    #    C -f--- Ea -- a
    #    |e      |g
    #    |       Tb -- p
    #    |      / l  
    #    Eb-j-Ta 
    #    |    |
    #    i    k

    dimEa1 = Ea.shape[0] # f
    dimEa2 = Ea.shape[1] # g
    dimEa3 = Ea.shape[2] # a
    dimEb1 = Eb.shape[0] # i
    dimEb2 = Eb.shape[1] # j
    dimEb3 = Eb.shape[2] # e

    # dimT, dimE = Ta.shape[0], Ea.shape[0] # to modify
    dimT = Ta.shape[0]
    D_new = min(dimEb1*dimT, chi)  # to modify

    CEETT = torch.einsum('ije,ef,fga,jkl,lpg->ikap',(Eb,C,Ea,Ta,Tb))
    CEETT = CEETT.reshape(dimEb1*dimT, dimEa3*dimT)  # Rho(i,a,k,p) => Rho(i,k,a,p) => Rho(ki, pa)
    CEETT = CEETT/CEETT.norm()

    # U, S, V = svd(CEETT@CEETT@CEETT) 
    U, S, V = svd(CEETT) 
    
    # U0, S0, V0 = svd(CEETT@CEETT) 
    # V0 = V0.t()
    # # print('norm A - USV 2:', ( U0@torch.diag(S0)@V0 - CEETT@CEETT).norm() )
    truncation_error = S[D_new:].sum()/S.sum()

    # U0 = U0[:, :D_new]
    # V0 = V0[:D_new, :]
    # S0 = S0[:D_new]

    # inv_sqrt_s = torch.zeros((D_new,D_new), dtype=CEETT.dtype, device=C.device)
    # for i in range(D_new):
    #     inv_sqrt_s[i,i] = 1/torch.sqrt(S0[i])

    # P1bar = inv_sqrt_s@torch.conj(U0).t()@CEETT # chi_new dchi
    # P1 = CEETT@V0.t()@inv_sqrt_s # dchi chi_new

    '''
    print('USV^T: ',(U0@torch.diag(S0)@V0.t()-A).norm()) # 1e-8 with USV = svd(A)
    '''

    # truncation_error = S[D_new:].sum()/S.sum() 
    P = U[:, :D_new] 

    # projection operator  P(ki, D_new) , dchi chi_new
    # P = V[:D_new,:].t()
    # print('imaginary part of U ', torch.conj(P).norm())

    # C = (torch.conj(P).t() @ CEETT @ P) #C(D_new, D_new)
    # N = (P1@P1bar).size()[0]
    # id1 = torch.zeros(N,N)
    # M = (P1bar@P1).size()[0]
    # id2 = torch.zeros(M,M)

    # for i in range(N):
    #     id1[i,i] = 1
    # for i in range(M):
    #     id2[i,i] = 1
    # # print('PPbar - id ',(P1@P1bar - id1).norm())
    # # print('PbarP - id ',(P1bar@P1 - id2).norm()) = 0


    C = P.t()@CEETT@P
    # C = P1bar@CEETT@P1
    
    # print('PPt - id ',(P@P.t() - id1).norm())
    # print('PtP - id ',(P.t()@P - id2).norm())  = 0
  

    # i--Ea -k- |
    #    | j     P - q   = Eb(ilq)
    #    Tb -m- |
    #   / l

    #   | -i- Eb -- k 
    # q - P   |j         = Ea(qmk)
    #   | -l- Ta 
    #           \ m

    P = P.view(dimEb1, dimT, D_new)                # P(i,k,D_new), chi d chi_new
    '''
      P1bar chi_new dchi
      P1    dchi chi_new
    '''
    # P1bar = P1bar.reshape(D_new, dimT, dimEb1)  # chi_new d chi
    # P1bar = P1bar.permute((2,1,0))              # chi d chi_new
    # P1 = P1.reshape(dimEb1,dimT,D_new)          # chi d chi_new

    Ebtmp = torch.einsum('ijk,lmj,kmq->ilq',(Ea,Tb,torch.conj(P)))
    Eatmp = torch.einsum('ijk,lmj,ilq->qmk',(Eb,Ta,P))

    # Ebtmp = torch.einsum('ijk,lmj,kmq->ilq',(Ea,Tb,P1))
    # Eatmp = torch.einsum('ijk,lmj,ilq->qmk',(Eb,Ta,P1bar))

    Eb = Ebtmp.clone()
    Ea = Eatmp.clone()

    C = C/C.norm() 
    Ea = Ea/Ea.norm()
    Eb = Eb/Eb.norm()

    return C, Ea, Eb, S/S.norm(), truncation_error



def CTMRG_honeycomb(Ta, Tb, H, M, A1symm, A2symm, chi, max_iter, dtype, use_checkpoint=False):

    threshold = 1E-12 

    ## Declaration of C, Ea, Eb => C3-symmetric state
    # C(down, right), E(up,right,down

    C = torch.ones((1,1), dtype=dtype, device= Ta.device)
    Ea = torch.ones((1,1,1),dtype=dtype, device= Ta.device)
    Eb = torch.ones((1,1,1),dtype=dtype, device= Ta.device)

    diff = 1E1
    ener = 0

    for n in range(max_iter):

        tensors = C, Ea, Eb, Ta, Tb, torch.tensor(chi) ## gives a tuple

        Etmp = ener
        ener, Mx, My, Mz = get_obs_honeycomb(A1symm, A2symm, H[0], M[0], M[1], M[2], C, Ea, Eb)
        # print('n', n, ' ', ener - Etmp)
        diff = abs(ener - Etmp)

        if use_checkpoint: # use checkpoint to save memory 
             C, Ea, Eb, s, error = checkpoint(renormalize_honeycomb, *tensors) 
        else:
            C, Ea, Eb, s, error = renormalize_honeycomb(*tensors)

        if (diff < threshold):
            break

        if n == max_iter-1:
            print('ctm not converged')


    return C, Ea, Eb


