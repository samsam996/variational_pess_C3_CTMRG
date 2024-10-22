import sys
sys.path.insert(0, '../')

import torch 
import time
from tensornets import CTMRG_honeycomb
from measure import get_obs_honeycomb
from args import args

# could try just one C3-symmetric A -> C, E.
class honeycombiPEPS(torch.nn.Module):

    def __init__(self, args, dtype=torch.float64, device='cpu', use_checkpoint=False):
        super(honeycombiPEPS, self).__init__()
        self.d = args.d
        self.D = args.D
        self.chi = args.chi
        self.Niter = args.Niter
        self.use_checkpoint = use_checkpoint 
        
        d, D = self.d, self.D
        
        B1 = torch.rand(d, D, D, D, dtype=dtype, device=device)
        # for i in range(self.d):
        #     B1[i,i,i,i] = torch.rand(1)

        B1 = B1/B1.norm()

        # A1 is the tensor that will be optimised.
        self.A1 = torch.nn.Parameter(B1)
        
    def forward(self, H, Mpx, Mpy, Mpz, chi, dtype):
        
        d, D, chi, Niter = self.d, self.D, self.chi, self.Niter

        # We make (in this particular case for the C3 CTMRG) the local tensor C3 symmetric
        # A1symm = self.A1.permute(0,1,2,3) +  self.A1.permute(0,1,3,2) + \
        #          self.A1.permute(0,2,3,1) + self.A1.permute(0,3,2,1) + \
        #          self.A1.permute(0,3,1,2) + self.A1.permute(0,2,1,3) 
        
        # We make (in this particular case for the C3v CTMRG) the local tensor C3v symmetric
        A1symm = self.A1.permute(0,1,2,3) +  self.A1.permute(0,2,3,1) +  self.A1.permute(0,3,1,2) 
        
        # A1symm d, D, D, D
        # We compute the new local tensor T

        T = torch.tensordot(A1symm,torch.conj(A1symm),([0],[0])) # T(d 123) T(d 345) = T(123456)
        T = T.permute(0,3,1,4,2,5)
        T = T.contiguous().view(D*D,D*D,D*D)
        T = T/T.norm()

        # print(T.norm())
        C, Ea, Eb = CTMRG_honeycomb(T, chi, Niter, dtype, self.use_checkpoint) 
        loss, Mx, My, Mz = get_obs_honeycomb(A1symm, A1symm, H, Mpx, Mpy, Mpz, C, Ea, Eb)

        loss = torch.real(loss)

        return loss, Mx, My, Mz 