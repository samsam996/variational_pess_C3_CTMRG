 



import sys
import io
import torch
import numpy as np
torch.set_num_threads(1)
torch.manual_seed(1879)
import subprocess
from utils import kronecker_product as kron
from utils import save_checkpoint, load_checkpoint
from ipeps import honeycombiPEPS
from ipeps import honeycombiPESS
from tensornets import CTMRG_honeycomb_pess
from measure import get_energy_pess
from args import args



def project_C3v_ireps(T):
    """
    Project a rank-3 tensor T onto A1, A2 and E irreducible representations of C3v.
    Returns:
        T_A1, T_A2, T_E
    """
    G_order = 6

    E = T

    C3 = T.permute(1,2,0)
    C3_2 = T.permute(2,0,1)

    S1 = T.permute(0,2,1)
    S2 = T.permute(2,1,0)
    S3 = T.permute(1,0,2)

    T_A1 = (1/6)*(E + C3 + C3_2 + S1 + S2 + S3)
    T_A2 = (1/6)*(E + C3 + C3_2 - S1 - S2 - S3)
    T_E  = 1/3*(E + C3 + C3_2)

    return T_A1, T_A2, T_E

def identify_C3v_irep(T, tol=1e-10):

    T_A1, T_A2, T_E = project_C3v_ireps(T)

    if torch.norm(T - T_A1) < tol:
        return 'A1'
    elif torch.norm(T - T_A2) < tol:
        return 'A2'
    elif torch.norm(T - T_E) < tol:
        return 'waza'
    else:
        return 'E (or linear combination)'



data = torch.load('final_tensors_D4.pt', map_location='cpu')
A1symm = data['A1symm']
A2symm = data['A2symm']



D = 4
Ta = torch.einsum('mefg,mabc->eafbgc',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2)
Tb = torch.einsum('efg,abc->eafbgc',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2)

irrepT1 = identify_C3v_irep(Tb)
print(irrepT1)

"""

if __name__=='__main__':
    
    dtype = args.dtype
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

    if args.dtype == "float32":
        dtype = torch.float32   
    elif args.dtype == "float64":
        dtype = torch.float64
    elif args.dtype =="float128":
        print('no possible jose')
        sys.exit()
    else:
        dtype = torch.cfloat


    model = honeycombiPESS(args, dtype, device, args.use_checkpoint)

    file_name = f'Heisenberg_D{args.D}_chi{args.chi}_float64/peps.tensor'
    check_point_path = '../data/'+file_name
    load_checkpoint(check_point_path,args, model)

    sy = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device)*0.5
    sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)*0.5
    sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
    sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
    sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)*0.5
    id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)


    def Ry(theta):
        return np.cos(theta/2)*id2 + np.sin(theta/2)*sy*2



    theta1 = 0.
    theta2 = -2*np.pi/3
    theta3 = -4*np.pi/3
        
    sm1 = Ry(theta1)@sm@Ry(theta1).t()
    sp1 = Ry(theta1)@sp@Ry(theta1).t()
    sz1 = Ry(theta1)@sz@Ry(theta1).t()

    sm2 = Ry(theta2)@sm@Ry(theta2).t()
    sp2 = Ry(theta2)@sp@Ry(theta2).t()
    sz2 = Ry(theta2)@sz@Ry(theta2).t()

    sm3 = Ry(theta3)@sm@Ry(theta3).t()
    sp3 = Ry(theta3)@sp@Ry(theta3).t()
    sz3 = Ry(theta3)@sz@Ry(theta3).t()

    h =  2*kron(kron(id2, sz2), sz3) + kron(kron(id2, sm2), sp3) + kron(kron(id2, sp2), sm3) # 23
    h += 2*kron(kron(sz1, id2), sz3) + kron(kron(sm1, id2), sp3) + kron(kron(sp1, id2), sm3) # 13
    h += 2*kron(kron(sz1, sz2), id2) + kron(kron(sm1, sp2), id2) + kron(kron(sp1, sm2), id2) 

    H =[h/2]
    
    Mpx = kron(sx, id2)
    Mpy = kron(sp, id2)
    Mpz = kron(sm, id2)
    M = [Mpx, Mpy, Mpz]

    A1symm = model.final_A1 
    A2symm = model.final_A2

    D = args.D 
    chi = [5,10,15,20,25,30,35,40,45,50]
    Niter = 100

    Ta = torch.einsum('mefg,mabc->eafbgc',(A1symm,torch.conj(A1symm))).reshape(D**2, D**2, D**2)
    Tb = torch.einsum('efg,abc->eafbgc',(A2symm,torch.conj(A2symm))).reshape(D**2, D**2, D**2)

    ener = []
    for chi_ in chi:
        C, Ea, Eb = CTMRG_honeycomb_pess(Ta, Tb, H[0], M, A1symm, A2symm, chi_, Niter, dtype, args.use_checkpoint) 
        ener.append((get_energy_pess(A1symm, A2symm, H[0], C, Ea, Eb)).item())

    print(f'energy: {ener}')
    print(f'chi: {chi}')

    torch.save({'A1symm': model.final_A1, 'A2symm': model.final_A2}, f'final_tensors_D{D}.pt')

    
"""