'''
Variational PEPS with automatic differentiation and GPU support
'''

import io
import torch
import numpy as np
torch.set_num_threads(1)
torch.manual_seed(1879)
import subprocess
from utils import kronecker_product as kron
from utils import save_checkpoint, load_checkpoint
from ipeps import honeycombiPEPS
from args import args



def R(theta,dtype,device):
    sy = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device)*0.5
    id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

    return np.cos(theta/2)*id2 + np.sin(theta/2)*sy




if __name__=='__main__':
    import time
    device = torch.device("cpu" if args.cuda<0 else "cuda:"+str(args.cuda))

    if args.dtype == "float32":
        dtype = torch.float32   # if args.float32 else torch.float64
    elif args.dtype == "float64":
        dtype = torch.float64
    else:
        dtype = torch.cfloat
    
    print ('use', dtype)

    model = honeycombiPEPS(args, dtype, device, args.use_checkpoint)

    if args.load is not None:
        try:
            load_checkpoint(args.load, args, model)
            print('load model', args.load)
        except FileNotFoundError:
            print('not found:', args.load)

    # optimizer = torch.optim.LBFGS(model.parameters(), max_iter=10)
    optimizer =  torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, eps=1e-08)

    params = list(model.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)

    key = args.folder
    key += args.model \
          + '_D' + str(args.D) \
          + '_chi' + str(args.chi)
    if args.dtype=="float32": #(args.float32):
        key += '_float32'
    elif args.dtype=="float64": #(args.float32):
        key += '_float64'
    if args.dtype=="cfloat": #(args.float32):
        key += '_cfloat'    
    cmd = ['mkdir', '-p', key]
    subprocess.check_call(cmd)

    if args.model == 'Heisenberg':
        # Hamiltonian operators on a bond
        # sy is not defined with imaginary numbers!!

        if dtype == torch.float32 or dtype == torch.float64:
            sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)*0.5
            sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
            sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
            sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)*0.5
            id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

            # now assuming Jz>0, Jxy > 0
            # We flip the spin on one of the sub-lattice to get back to a one-site unit cell in the iPEPS wavefunction.
            h = 2*kron(sz,4*sx@sz@sx)-(kron(sm, 4*sx@sp@sx)+kron(sp,4*sx@sm@sx))
            H =[h]
        
        else: 
            sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)*0.5
            sy = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)*0.5
            sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)*0.5
            id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

            Rypi = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device)
            
            # h = 2*args.Jz*kron(sz,4*Rypi@sz@Rypi.t())+2*args.Jxy*(kron(sy, Rypi@sy@Rypi.t()) + kron(sx, Rypi@sx@Rypi.t()))
            h = 2*args.Jz*kron(sz, Rypi@sz@Rypi.t())+2*args.Jxy*(kron(sy, Rypi@sy@Rypi.t()) + kron(sx, Rypi@sx@Rypi.t()))
            H =[h]

        Mpx = kron(sx, id2)
        Mpy = kron(sz, id2)
        Mpz = kron(sz, id2)


    elif args.model == 'maple_leaf':
            
            # For now we assume to be in the Neel phase
            sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
            sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
            sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)*0.5
            id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

            # Rypi = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device)
            Rypi = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
            Rypi = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

            #        1 -- 2
            #       / \B /
            #      4 -- 3
            #    / A\  /
            #   6 -- 5
            #
            #  Jt: 13, 45 // Jd 34 // Jh 14 35 

            # BA
            H1 = args.Jt*(kron(kron(kron(sp,id2),kron(sm,id2)),kron(id2,id2)) + \
                          kron(kron(kron(sm,id2),kron(sp,id2)),kron(id2,id2)) + \
                        2*kron(kron(kron(sz,id2),kron(sz,id2)),kron(id2,id2)) )    # 13
            
            H1 = H1 + args.Jt*(kron(kron(kron(id2,id2),kron(id2,Rypi @ sp @ Rypi.t())),kron(Rypi @ sm @ Rypi.t(), id2)) + \
                               kron(kron(kron(id2,id2),kron(id2,Rypi @ sm @ Rypi.t())),kron(Rypi @ sp @ Rypi.t(), id2)) + \
                             2*kron(kron(kron(id2,id2),kron(id2,Rypi @ sz @ Rypi.t())),kron(Rypi @ sz @ Rypi.t(), id2)) )   # 45
            
            H1 = H1 + args.Jd*(kron(kron(kron(id2,id2),kron(sp, Rypi@sm@Rypi.t())),kron(id2,id2)) + \
                               kron(kron(kron(id2,id2),kron(sm, Rypi@sp@Rypi.t())),kron(id2,id2)) + \
                             2*kron(kron(kron(id2,id2),kron(sz, Rypi@sz@Rypi.t())),kron(id2,id2))) # 34 
            
            H1 = H1 + args.Jh*(kron(kron(kron(sp,id2),kron(id2, Rypi @ sm @ Rypi.t())),kron(id2,id2)) + \
                               kron(kron(kron(sm,id2),kron(id2, Rypi @ sp @ Rypi.t())),kron(id2,id2)) + \
                             2*kron(kron(kron(sz,id2),kron(id2, Rypi @ sz @ Rypi.t())),kron(id2,id2)))  # 14 
            
            H1 = H1 + args.Jh*(kron(kron(kron(id2,id2),kron(sp,id2)),kron( Rypi@sm@Rypi.t(),id2)) + \
                               kron(kron(kron(id2,id2),kron(sm,id2)),kron( Rypi@sp@Rypi.t(),id2)) + \
                             2*kron(kron(kron(id2,id2),kron(sz,id2)),kron( Rypi@sz@Rypi.t(),id2)))  # 35 ok

    
            #    1 -- 2 
            #     \ B/ \    
            #      3  | 4
            #       \ |/ A\  
            #         6 -- 5
            #
            # Jt: 23, 46 // Jd 26 // Jh 24 36
            # BA
            H2 = args.Jt*(kron(kron(kron(id2,sp),kron(sm,id2)),kron(id2,id2)) + \
                          kron(kron(kron(id2,sm),kron(sp,id2)),kron(id2,id2)) + \
                        2*kron(kron(kron(id2,sz),kron(sz,id2)),kron(id2,id2)) )       # 23 ok
            
            H2 = H2 + args.Jt*(kron(kron(kron(id2,id2),kron(id2,Rypi@sp@Rypi.t())),kron(id2, Rypi@sm@Rypi.t())) + \
                               kron(kron(kron(id2,id2),kron(id2,Rypi@sm@Rypi.t())),kron(id2, Rypi@sp@Rypi.t())) + \
                             2*kron(kron(kron(id2,id2),kron(id2,Rypi@sz@Rypi.t())),kron(id2, Rypi@sz@Rypi.t())) )   # 46 ok
            
            H2 = H2 + args.Jd*(kron(kron(kron(id2,sp),kron(id2,id2)),kron(id2, Rypi@sm@Rypi.t())) + \
                               kron(kron(kron(id2,sm),kron(id2,id2)),kron(id2, Rypi@sp@Rypi.t())) + \
                             2*kron(kron(kron(id2,sz),kron(id2,id2)),kron(id2, Rypi@sz@Rypi.t())))    # 26 ok
            
            H2 = H2 + args.Jh*(kron(kron(kron(id2,sp),kron(id2, Rypi@sm@Rypi.t())),kron(id2,id2)) + \
                               kron(kron(kron(id2,sm),kron(id2, Rypi@sp@Rypi.t())),kron(id2,id2)) + \
                             2*kron(kron(kron(id2,sz),kron(id2, Rypi@sz@Rypi.t())),kron(id2,id2)))  # 24 ok
            
            H2 = H2 + args.Jh*(kron(kron(kron(id2,id2),kron(sp,id2)),kron(id2, Rypi@sm@Rypi.t())) + \
                               kron(kron(kron(id2,id2),kron(sm,id2)),kron(id2, Rypi@sp@Rypi.t())) + \
                             2*kron(kron(kron(id2,id2),kron(sz,id2)),kron(id2, Rypi@sz@Rypi.t())))  # 36 ok
            

            #      4 
            #    / A\  
            #   6 -- 5
            #   | /  |
            #   1 -- 2
            #    \ B/ 
            #      3
            #
            # Jt 12 56 // Jd 15 // Jh 16 25
            # BA

            H3 = args.Jt*(kron(kron(kron(sp,sm),kron(id2, id2)),kron(id2,id2)) + \
                          kron(kron(kron(sm,sp),kron(id2, id2)),kron(id2,id2)) + \
                        2*kron(kron(kron(sz,sz),kron(id2, id2)),kron(id2,id2)) )     # 12
            
            H3 = H3 + args.Jt*(kron(kron(kron(id2,id2),kron(id2,id2)),kron(Rypi@sp@Rypi.t(), Rypi@sm@Rypi.t())) + \
                               kron(kron(kron(id2,id2),kron(id2,id2)),kron(Rypi@sm@Rypi.t(), Rypi@sp@Rypi.t())) + \
                             2*kron(kron(kron(id2,id2),kron(id2,id2)),kron(Rypi@sz@Rypi.t(), Rypi@sz@Rypi.t())) )  # 56
            
            H3 = H3 + args.Jd*(kron(kron(kron(sp,id2),kron(id2, id2)),kron(Rypi@sm@Rypi.t(),id2)) + \
                               kron(kron(kron(sm,id2),kron(id2, id2)),kron(Rypi@sp@Rypi.t(),id2)) + \
                             2*kron(kron(kron(sz,id2),kron(id2, id2)),kron(Rypi@sz@Rypi.t(),id2)))  # 15
            
            H3 = H3 + args.Jh*(kron(kron(kron(sp,id2),kron(id2, id2)),kron(id2, Rypi@sm@Rypi.t())) + \
                               kron(kron(kron(sm,id2),kron(id2, id2)),kron(id2, Rypi@sp@Rypi.t())) + \
                             2*kron(kron(kron(sz,id2),kron(id2, id2)),kron(id2, Rypi@sz@Rypi.t())))  # 16
            
            H3 = H3 + args.Jh*(kron(kron(kron(id2,sp),kron(id2,id2)),kron( Rypi@sm@Rypi.t(),id2)) + \
                               kron(kron(kron(id2,sm),kron(id2,id2)),kron( Rypi@sp@Rypi.t(),id2)) + \
                             2*kron(kron(kron(id2,sz),kron(id2,id2)),kron( Rypi@sz@Rypi.t(),id2)))  # 25
            
            # H3 = H3 + 0.01*(kron(kron(kron(sz,id2),kron(id2,id2)),kron( id2,id2)) + \
            #                 kron(kron(kron(id2,sz),kron(id2,id2)),kron( id2,id2)) + \
            #                 kron(kron(kron(id2,id2),kron(sz,id2)),kron( id2,id2)))  #123

            H = [H1,H2,H3]

            Mpx = kron(kron(kron(id2,sz),kron(id2,id2)),kron(id2,id2))
            Mpy = kron(kron(kron(id2,id2),kron(sz,id2)),kron(id2,id2))
            Mpz = kron(kron(kron(id2,id2),kron(id2,id2)),kron(id2,sz))



    else:
        print ('Only support Heisenberg model???')
        sys.exit(1)

    print ('Hamiltonian:\n', H)

    # for the ADAM optimiser
    def train_step(H, Mpx, Mpy, Mpz, args, dtype):
        # Zero the gradients from the previous step
        optimizer.zero_grad()
        # Start the timer for performance measurement
        start = time.time()
        # Perform a forward pass through the model
        loss, Mx, My, Mz = model.forward(H, Mpx, Mpy, Mpz, args.chi, dtype)
        # Perform backpropagation to compute gradients
        loss.backward()
        # Update weights using the optimizer
        optimizer.step()
        # Record the time taken for the forward pass
        forward = time.time()
        # Return loss for monitoring
        return loss.item(), Mx, My, Mz

    # for the LGBS optimiser
    def closure():
        # Zero the gradients from the previous step
        optimizer.zero_grad()
        # Start the timer for performance measurement
        start = time.time()
        # Perform a forward pass through the model
        loss, Mx, My, Mz = model.forward(H, Mpx, Mpy, Mpz, args.chi, dtype)
        # Record the time taken for the forward pass
        forward = time.time()
        # Perform backpropagation to compute gradients
        loss.backward()
        #print (model.A.norm().item(), model.A.grad.norm().item(), loss.item(), Mx.item(), My.item(), Mz.item(), torch.sqrt(Mx**2+My**2+Mz**2).item(), forward-start, time.time()-forward)
        return loss


    # # Logging and training loop
    with io.open(key + '.log', 'a', buffering=1, newline='\n') as logfile:

        print('dddd :',args.d) # ok
        En = 4
        Etmp = 5
        epoch = 0
        while epoch < args.Nepochs:
            epoch = epoch + 1

            # Train step and get loss and magnetization values
            # loss, Mx, My, Mz = train_step(H, Mpx, Mpy, Mpz, args, dtype)
            loss = optimizer.step(closure)
           
            # Save checkpoint periodically
            if (epoch % args.save_period == 0):
                save_checkpoint(key + '/peps.tensor'.format(epoch), model, optimizer)

            with torch.no_grad():
                Etmp = En
                En, Mx, My, Mz = model.forward(H, Mpx, Mpy, Mpz, args.chi if args.chi_obs is None else args.chi_obs, dtype)
                Mg = torch.sqrt(Mx**2 + My**2 + Mz**2)
                message = ('{} ' + 5 * '{:.8f} ').format(epoch, En/2*1.5/9, Mx, My, Mz, Mg)
                print('epoch, En, Mx, My, Mz, Mg', message)
                logfile.write(message + u'\n')


