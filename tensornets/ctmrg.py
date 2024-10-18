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

    Rho = torch.tensordot(C, Ea, ([1],[0]))   # C(ef)*Ea(fga)=Rho(ega)
    Rho = torch.tensordot(Eb, Rho, ([2],[0])) # Eb(ije)*Rgo(ega)=Rho(ijga)
    TaTb = torch.tensordot(T, T, ([2],[0])) # Ta(jkl)*Tb(lpg) = Tab(jkpg)

    # print((T-T.permute(2,1,0)).norm()) 
    # print((T-T.permute(1,2,0)).norm()) 

    Rho = torch.tensordot(Rho,TaTb, ([1,2],[0,3])) # Rho(ijga)*Tab(jkpg) = Rho(i,a,k,p)
    Rho = Rho.permute(0,2,1,3).contiguous().view(dimEb1*dimT, dimEa3*dimT)  # Rho(i,a,k,p) => Rho(i,k,a,p) => Rho(ki, pa)
    Rho = Rho/Rho.norm()

    # print((Rho - Rho.t()).norm()) # 
    # step 2: Get Isometry P 
    U, S, V = svd(Rho@Rho@Rho) 
    # U, S, V = svd(Rho) 

    # Rho is not symmetric, although it should!    
    # print(Rho)
    # x[3] = 0

    truncation_error = S[D_new:].sum()/S.sum() 
    P = U[:, :D_new] # projection operator  P(ki, D_new)

    # step 3: renormalize C and E
    C = (P.t() @ Rho @ P) #C(D_new, D_new)

    # i--Ea -k- |
    #    | j     P - q   = Eb(ilq)
    #    Tb -m- |
    #   / l

    #   | -i- Eb -- k 
    # q - P   |j         = Ea(qmk)
    #   | -l- Ta 
    #           \ m

    # print('size P')
    # print(P.size())

    
    P = P.view(dimEb1, dimT, D_new)                # P(i,k,D_new)
    Ebtmp = torch.tensordot(Ea,T,([1],[2]))       # Ea(ijk) Tb(lmj) = Ebtmp(iklm) 
    Ebtmp = torch.tensordot(Ebtmp,P,([1,3],[0,1])) # Ebtmp(iklm) P(kmq) = Ebtmp(ilq) 

    Eatmp = torch.tensordot(Eb,T, ([1],[2]))      # Eb(ijk)*Ta(lmj) = Eatmp(iklm)
    Eatmp = torch.tensordot(Eatmp,P, ([0,2],[0,1]))  # Eatmp(iklm)*P(ilq) = Eatmp(kmq)

    # Ebtmp(ilq) => ilq
    Eb = Ebtmp
    # Eatmp(kmq) => qmk
    Ea = Eatmp.permute(2,1,0)

    return C/C.norm(), Ea/Ea.norm(), Eb/Eb.norm(), S.abs()/S.abs().max(), truncation_error


## For the ruby lattice, I could ask for the state to be C3-symmetric. 
## Ea, Eb, C // Ta and Tb are the local tensors
def CTMRG_honeycomb(T, chi, max_iter, use_checkpoint=False):
    # T(up, left, down, right)

    threshold = 1E-12 if T.dtype is torch.float64 else 1E-6 # ctmrg convergence threshold

    ## Declaration of C, Ea, Eb => C3-symmetric state
    # C(down, right), E(up,right,down)
    # C = Ta.sum(1)  
    # Ea = Ta #.sum(1).permute(0,2,1)
    # Eb = Tb #.sum(1).permute(0,2,1)


    C = torch.ones((1,1), dtype=torch.float64)
    Ea = torch.ones((1,1,1),dtype=torch.float64)
    Eb = torch.ones((1,1,1),dtype=torch.float64)

    truncation_error = 0.0
    sold = torch.zeros(chi, dtype=T.dtype, device=T.device)
    diff = 1E1
    for n in range(max_iter):
        # tensors = C, Ea, Eb, Ta, Tb, torch.tensor(chi)  ## gives a tuple
        tensors = C, Ea, Eb, T, torch.tensor(chi)  ## gives a tuple

        # print((T-T.permute(0,2,1)).norm())

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
            #print( s, sold )
            #print( 'n: %d, Enorm: %g, error: %e, diff: %e' % (n, Enorm, error.item(), diff) )
        if (diff < threshold):
            break
        sold = s
    #print ('ctmrg converged at iterations %d to %.5e, truncation error: %.5f'%(n, diff, truncation_error/n))

    return C, Ea, Eb


