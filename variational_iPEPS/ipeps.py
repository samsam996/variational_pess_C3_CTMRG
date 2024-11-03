import sys
sys.path.insert(0, '../')

import torch 
import time
from tensornets import CTMRG_honeycomb
from measure import get_obs_honeycomb
from args import args

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
        B2 = torch.abs(torch.rand(d, D, D, D, dtype=dtype, device=device))
        B2 = B2/B2.norm()

        self.A1 = torch.nn.Parameter(B1)
        self.A2 = torch.nn.Parameter(B2)
        
    def forward(self, H, Mpx, Mpy, Mpz, chi, dtype):
        
        d, D, chi, Niter = self.d, self.D, self.chi, self.Niter

        #  We make (in this particular case for the C3 CTMRG) the local tensor C3 symmetric
        # A1symm = self.A1.permute(0,1,2,3) +  self.A1.permute(0,1,3,2) + \
        #          self.A1.permute(0,2,3,1) + self.A1.permute(0,3,2,1) + \
        #          self.A1.permute(0,3,1,2) + self.A1.permute(0,2,1,3) 
        
        if d == 8:
            A1symm = self.A1
            # A2symm = self.A2
            A1symm = self.A1.reshape(2,2,2,D,D,D)
            A1symm = A1symm.permute(0,1,2, 3,4,5) +  A1symm.permute(1,2,0 , 4,5,3) +  A1symm.permute(2,0,1, 5,3,4) 
            A1symm = A1symm.reshape(8, D,D,D)

            A2symm = A1symm
            # A2symm = self.A2.reshape(2,2,2,D,D,D)
            # A2symm = A2symm.permute(0,1,2, 3,4,5) +  A2symm.permute(1,2,0 , 4,5,3) +  A2symm.permute(2,0,1, 5,3,4) 
            # A2symm = A2symm.reshape(8, D,D,D)
        elif d==4:
            A1symm = self.A1
            A2symm = self.A2
            A1symm = A1symm.permute(0, 1,2,3) +  A1symm.permute(0 , 2,3,1) +  A1symm.permute(0, 3,1,2) 
            A2symm = A2symm.permute(0, 1,2,3) +  A2symm.permute(0 , 2,3,1) +  A2symm.permute(0, 3,1,2) 



        else:
            A1symm = self.A1.permute(0,1,2,3) +  self.A1.permute(0,2,3,1) + self.A1.permute(0,3,1,2) 
            A2symm = self.A2.permute(0,1,2,3) +  self.A2.permute(0,2,3,1) + self.A2.permute(0,3,1,2) 

            A1symm = A1symm/A1symm.norm()
            A2symm = A2symm/A2symm.norm()
            # A2symm = A1symm
            
        Ta = torch.einsum('mefg,mabc->eafbgc',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2)
        Tb = torch.einsum('mefg,mabc->eafbgc',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2)


        M = [Mpx, Mpy, Mpz]
        C, Ea, Eb = CTMRG_honeycomb(Ta, Tb, H, M, A1symm, A2symm, chi, Niter, dtype, self.use_checkpoint) 

        loss = 0
        ener_array = []
        for i in range(len(H)):
            enerab, enerba, Mx, My, Mz = get_obs_honeycomb(A1symm, A2symm, H[i], Mpx, Mpy, Mpz, C, Ea, Eb)
            ener_array.append((enerab))
            # print(f'Ener{i} :',ener)
            # print(ener)

        loss = torch.real(sum(ener_array)/len(ener_array))

        
        return loss, Mx, My, Mz 
    

    class honeycombiPEPSNoSymm(torch.nn.Module):

        def __init__(self,chi):
            self.chi = chi



