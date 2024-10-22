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
    optimizer =  torch.optim.Adam(model.parameters(), lr=5e-4)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-08)

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
        key += 'cfloat'    
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
            H = 2*args.Jz*kron(sz,4*sx@sz@sx)-args.Jxy*(kron(sm, 4*sx@sp@sx)+kron(sp,4*sx@sm@sx))
        
        else: 
            sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)*0.5
            sy = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)*0.5
            sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
            sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
            sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)*0.5
            id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

            H = 2*args.Jz*kron(sz,4*sx@sz@sx)-2*args.Jxy*(kron(sy, 4*sx@sy@sx) + kron(sx, 4*sx@sx@sx))

        Mpx = kron(sx, id2)
        Mpy = kron(sp, id2)
        Mpz = kron(sz, id2)

    else:
        print ('Only support Heisenberg model???')
        sys.exit(1)

    print ('Hamiltonian:\n', H)

    # using ADAM
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

    # # Logging and training loop
    with io.open(key + '.log', 'a', buffering=1, newline='\n') as logfile:

        En = 4
        Etmp = 5
        epoch = 0
        while (En < Etmp or abs(En - Etmp) < 1e-5) and epoch < args.Nepochs:
            epoch = epoch + 1
        # for epoch in range(args.Nepochs):
            # Train step and get loss and magnetization values
            loss, Mx, My, Mz = train_step(H, Mpx, Mpy, Mpz, args, dtype)
            
            # Save checkpoint periodically
            if (epoch % args.save_period == 0):
                save_checkpoint(key + '/peps.tensor'.format(epoch), model, optimizer)

            with torch.no_grad():
                Etmp = En
                En, Mx, My, Mz = model.forward(H, Mpx, Mpy, Mpz, args.chi if args.chi_obs is None else args.chi_obs, dtype)
                Mg = torch.sqrt(Mx**2 + My**2 + Mz**2)
                message = ('{} ' + 5 * '{:.8f} ').format(epoch, En/2*1.5, Mx, My, Mz, Mg)
                print('epoch, En, Mx, My, Mz, Mg', message)
                logfile.write(message + u'\n')



    # ## LGBS OPTIMISER 
    # def closure():
    #     # Zero the gradients from the previous step
    #     optimizer.zero_grad()
    #     # Start the timer for performance measurement
    #     start = time.time()
    #     # Perform a forward pass through the model
    #     loss, Mx, My, Mz = model.forward(H, Mpx, Mpy, Mpz, args.chi, dtype)
    #     # Record the time taken for the forward pass
    #     forward = time.time()
    #     # Perform backpropagation to compute gradients
    #     loss.backward()
    #     #print (model.A.norm().item(), model.A.grad.norm().item(), loss.item(), Mx.item(), My.item(), Mz.item(), torch.sqrt(Mx**2+My**2+Mz**2).item(), forward-start, time.time()-forward)
    #     return loss

    # with io.open(key+'.log', 'a', buffering=1, newline='\n') as logfile:
    #     for epoch in range(args.Nepochs):
    #         loss = optimizer.step(closure)
    #         if (epoch%args.save_period==0):
    #             save_checkpoint(key+'/peps.tensor'.format(epoch), model, optimizer)

    #         with torch.no_grad():
    #             En, Mx, My, Mz = model.forward(H, Mpx, Mpy, Mpz, args.chi if args.chi_obs is None else args.chi_obs, dtype)
    #             Mg = torch.sqrt(Mx**2+My**2+Mz**2)
    #             message = ('{} ' + 5*'{:.8f} ').format(epoch, En/2*1.5, Mx, My, Mz, Mg)
    #             print ('epoch, En, Mx, My, Mz, Mg', message)
    #             logfile.write(message + u'\n')
