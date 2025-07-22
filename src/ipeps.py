import sys
sys.path.insert(0, '../')

import torch 
import time
from src.ctmrg import CTMRG_honeycomb
from src.ctmrg import CTMRG_honeycomb_pess
from src.measure import get_obs_honeycomb
from src.measure import get_energy_pess
from src.args import args



class honeycombiPEPS(torch.nn.Module):

    def __init__(self, args, dtype=torch.float64, device='cpu', use_checkpoint=False):
        super(honeycombiPEPS, self).__init__()
        self.d = args.d
        self.D = args.D
        self.chi = args.chi
        self.Niter = args.Niter
        self.use_checkpoint = use_checkpoint 
        
        d, D = self.d, self.D
        
        B1 = torch.abs(torch.rand(d, D, D, D, dtype=dtype, device=device))
        B1 = B1/B1.norm()

        self.A1 = torch.nn.Parameter(B1)
        
    def forward(self, H, Mpx, Mpy, Mpz, chi, dtype):
        
        d, D, chi, Niter = self.d, self.D, self.chi, self.Niter


       
        A1symm = self.A1.permute(0,1,2,3) 
        A1symm = A1symm + self.A1.permute(0,2,3,1) 
        A1symm = A1symm + self.A1.permute(0,3,1,2) 
        A1symm = A1symm/A1symm.norm()
        A2symm = A1symm
            
        Ta = torch.einsum('mefg,mabc->eafbgc',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2)
        Tb = torch.einsum('mefg,mabc->eafbgc',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2)

        M = [Mpx, Mpy, Mpz]
        C0, Ea0, Eb0 = CTMRG_honeycomb(Ta, Tb, H, M, A1symm, A2symm, chi, Niter, dtype, self.use_checkpoint) 

        loss = 0
        ener_array = []
        for i in range(len(H)):
            enerab0, enerba0, Mx, My, Mz = get_obs_honeycomb(A1symm, A2symm, H[i], Mpx, Mpy, Mpz, C0, Ea0, Eb0)
            ener_array.append(enerab0)
        
        loss = torch.real(sum(ener_array)/len(ener_array))
        
        
        return loss, Mx, My, Mz 
    
class honeycombiPESS(torch.nn.Module):

    def __init__(self, args, dtype=torch.float64, device='cpu', use_checkpoint=False):
        super(honeycombiPESS, self).__init__()
        self.d = args.d
        self.D = args.D
        self.chi = args.chi
        self.Niter = args.Niter
        self.use_checkpoint = use_checkpoint 
        
        d, D = self.d, self.D
        
        B1 = (torch.rand(d, D, D, D, dtype=dtype, device=device))
        B1 = B1/B1.norm()
        B2 = (torch.rand(D, D, D, dtype=dtype, device=device))
        B2 = B2/B2.norm()

        self.A1 = torch.nn.Parameter(B1)
        self.A2 = torch.nn.Parameter(B2)

        self.register_buffer("final_A1", torch.empty(d, D, D, D, dtype=dtype, device=device))
        self.register_buffer("final_A2", torch.empty(D, D, D, dtype=dtype, device=device))
        
    def forward(self, H, Mpx, Mpy, Mpz, chi, dtype):
        
        d, D, chi, Niter = self.d, self.D, self.chi, self.Niter


        """
            Symmetrise the local tensors under C3 permutation.
        """
        A1symm = self.A1.permute(0,1,2,3) + self.A1.permute(0,2,3,1) + self.A1.permute(0,3,1,2) 
        A2symm = self.A2.permute(0,1,2) + self.A2.permute(1,2,0) + self.A2.permute(2,0,1) 



        A1symm = A1symm/A1symm.norm() 
        A2symm = A2symm/A2symm.norm() 
            
        self.final_A1.copy_(A1symm) 
        self.final_A2.copy_(A2symm) 
        
        
        Ta = torch.einsum('mefg,mabc->eafbgc',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2)
        Tb = torch.einsum('efg,abc->eafbgc',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2)

        M = [Mpx, Mpy, Mpz]
        C, Ea, Eb = CTMRG_honeycomb_pess(Ta, Tb, H[0], M, A1symm, A2symm, chi, Niter, dtype, self.use_checkpoint) 

        loss = 0
        ener, Mx, My, Mz = get_energy_pess(A1symm, A2symm, H[0], Mpx, Mpy, Mpz, C, Ea, Eb)
            
        loss = torch.real(ener)
        
        return loss , Mx, My, Mz 
    
