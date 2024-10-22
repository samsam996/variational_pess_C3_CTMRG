import torch
from torch.utils.checkpoint import checkpoint

from .adlib import SVD 
svd = SVD.apply
#from .adlib import EigenSolver
#symeig = EigenSolver.apply

def renormalize_honeycomb(*tensors):
    # T(up,left,down,right), u=up, l=left, d=down, r=right
    # C(d,r), EL(u,r,d), EU(l,d,r)

    # print('BEGINNING OF RENORM HONEYCOMB')

    # C, Ea, Eb, Ta, Tb, chi = tensors 
    C, Ea, Eb, T, chi = tensors 

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
    dimT = T.shape[0]
    D_new = min(dimEb1*dimT, chi)  # to modify

    CE = torch.tensordot(C, Ea, ([1],[0]))   # C(ef)*Ea(fga)=Rho(ega)
    CEE = torch.tensordot(Eb, CE, ([2],[0])) # Eb(ije)*Rgo(ega)=Rho(ijga)
    TaTb = torch.tensordot(T, T, ([2],[0])) # Ta(jkl)*Tb(lpg) = Tab(jkpg)

    CEETT = torch.tensordot(CEE,TaTb, ([1,2],[0,3])) # Rho(ijga)*Tab(jkpg) = Rho(i,a,k,p)
    CEETT = CEETT.permute(0,2,1,3).contiguous().view(dimEb1*dimT, dimEa3*dimT)  # Rho(i,a,k,p) => Rho(i,k,a,p) => Rho(ki, pa)
    CEETT = CEETT/CEETT.norm()

    # step 2: Get Isometry P 
    # print((Rho.norm()))
    # print(C.norm())
    # print(T.norm())
    # print(Ea.norm())
    # print(Eb.norm())

    U0, S0, V0 = svd(CEETT@CEETT@CEETT) 
    U, S, V = svd(CEETT) 

    # print((CEETT - torch.conj(CEETT).t()).norm())
    # print((S - S0).norm())

    truncation_error = S[D_new:].sum()/S.sum() 
    P = U[:, :D_new] # projection operator  P(ki, D_new)

    # print('imaginary part of U ', torch.conj(P).norm())
    C = (torch.conj(P).t() @ CEETT @ P) #C(D_new, D_new)


    # i--Ea -k- |
    #    | j     P - q   = Eb(ilq)
    #    Tb -m- |
    #   / l

    #   | -i- Eb -- k 
    # q - P   |j         = Ea(qmk)
    #   | -l- Ta 
    #           \ m
    # A = P@torch.conj(P).t()
    A = torch.conj(P).t()@P # \simeq id
    N = A.size()[0]
    id = torch.zeros((N,N))
    for i in range(N):
        id[i,i] = 1

    # print('PP^T: ',(A - id).norm())
    
    P = P.view(dimEb1, dimT, D_new)                # P(i,k,D_new)
    Ebtmp = torch.tensordot(Ea,T,([1],[2]))       # Ea(ijk) Tb(lmj) = Ebtmp(iklm) 
    Ebtmp = torch.tensordot(Ebtmp,torch.conj(P),([1,3],[0,1])) # Ebtmp(iklm) P(kmq) = Ebtmp(ilq) 

    Eatmp = torch.tensordot(Eb,T, ([1],[2]))      # Eb(ijk)*Ta(lmj) = Eatmp(iklm)
    Eatmp = torch.tensordot(Eatmp,P, ([0,2],[0,1]))  # Eatmp(iklm)*P(ilq) = Eatmp(kmq)

    # Ebtmp(ilq) => ilq
    Eb = Ebtmp
    # Eatmp(kmq) => qmk
    Ea = Eatmp.permute(2,1,0)

  
    return C/C.norm(), Ea/Ea.norm(), Eb/Eb.norm(), S/S.max(), truncation_error


## For the ruby lattice, I could ask for the state to be C3-symmetric. 
## Ea, Eb, C // Ta and Tb are the local tensors
def CTMRG_honeycomb(T, chi, max_iter, dtype, use_checkpoint=False):

    threshold = 1E-7 #if T.dtype is torch.float64 else 1E-6 # ctmrg convergence threshold

    ## Declaration of C, Ea, Eb => C3-symmetric state
    # C(down, right), E(up,right,down

    C = torch.ones((1,1), dtype=dtype)
    Ea = torch.ones((1,1,1),dtype=dtype)
    Eb = torch.ones((1,1,1),dtype=dtype)

    truncation_error = 0.0
    sold = torch.zeros(chi, dtype=T.dtype, device=T.device)
    diff = 1E1
    for n in range(max_iter):

        tensors = C, Ea, Eb, T, torch.tensor(chi)  ## gives a tuple

        if use_checkpoint: # use checkpoint to save memory
            C, Ea, Eb, s, error = checkpoint(renormalize_honeycomb, *tensors) 
        else:
            C, Ea, Eb, s, error = renormalize_honeycomb(*tensors)

        Ea = Ea/Ea.norm()
        Eb = Eb/Eb.norm()
        C = C/C.norm()

        truncation_error += error.item()
        if (s.numel() == sold.numel()):
            diff = (s-sold).norm().item()
            # print(n,'  :  ',diff)
            #print( s, sold )
            # print( 'n: %d, Enorm: %g, error: %e, diff: %e' % (n, Enorm, error.item(), diff) )
        if (diff < threshold):
            break
        sold = s
    #print ('ctmrg converged at iterations %d to %.5e, truncation error: %.5f'%(n, diff, truncation_error/n))

    return C, Ea, Eb


