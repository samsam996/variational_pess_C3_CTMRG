'''
Variational PEPS with automatic differentiation and GPU support
'''

# I don't like the ctm convergence criteria on s(end), I want it on the energy !!


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
from args import args



if __name__=='__main__':

    import time
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
    
    print ('use', dtype)

    model = honeycombiPEPS(args, dtype, device, args.use_checkpoint)

    if args.load is not None:
        try:
            load_checkpoint(args.load, args, model)
            print('load model', args.load)
        except FileNotFoundError:
            print('not found:', args.load)

    optimizer =  torch.optim.Adam(model.parameters(), lr=1e-3)

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

    if args.model=='maple_leaf':
        key += f'_Jd{args.Jd:.2f}_Jt{args.Jt:.2f}'

    cmd = ['mkdir', '-p', key]
    subprocess.check_call(cmd)


    if args.model == 'Heisenberg':
        # Hamiltonian operators on a bond
        # sy is not defined with imaginary numbers!!

        sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)*0.5
        sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
        sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
        sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)*0.5
        id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

        # now assuming Jz>0, Jxy > 0
        # We flip the spin on one of the sub-lattice to get back to a one-site unit cell in the iPEPS wavefunction.
        # h = 2*kron(sz,sz)+(kron(sm, sp)+kron(sp,sm))
        h = 2*kron(sz,4*sx@sz@sx)-(kron(sm, 4*sx@sp@sx)+kron(sp,4*sx@sm@sx))

        H =[h/2]
        
    
        Mpx = kron(sx, id2)
        Mpy = kron(sp, id2)
        Mpz = kron(sm, id2)

    elif args.model == 'maple_leaf':
            
            # For now we assume to be in the Neel phase
            sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)*0.5
            sy = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device)*0.5

            sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
            sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
            sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)*0.5
            id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)

            Rypi = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device)


            def Ry(theta):
                return np.cos(theta/2)*id2 + np.sin(theta/2)*sy*2



            # #        1 -- 2
            # #       / \ A/
            # #      4 -- 3
            # #    / B\  /
            # #   6 -- 5
            # #
            # #  Jt: 13, 45 // Jd 34 // Jh 14 35 
            # # BA 1x1 works
            def Hx(Ry1, Ry2, Ry3, Ry4, Ry5, Ry6):

                H1 = args.Jt*(kron(kron(kron(Ry1@sp@Ry1.t(),id2),kron(Ry3@sm@Ry3.t(),id2)),kron(id2,id2)) + \
                              kron(kron(kron(Ry1@sm@Ry1.t(),id2),kron(Ry3@sp@Ry3.t(),id2)),kron(id2,id2)) + \
                            2*kron(kron(kron(Ry1@sz@Ry1.t(),id2),kron(Ry3@sz@Ry3.t(),id2)),kron(id2,id2)) )    # 13
                
                H1 = H1 + args.Jt*(kron(kron(kron(id2,id2),kron(id2,Ry4@sp@Ry4.t())),kron(Ry5@sm@Ry5.t(), id2)) + \
                                   kron(kron(kron(id2,id2),kron(id2,Ry4@sm@Ry4.t())),kron(Ry5@sp@Ry5.t(), id2)) + \
                                 2*kron(kron(kron(id2,id2),kron(id2,Ry4@sz@Ry4.t())),kron(Ry5@sz@Ry5.t(), id2)) )   # 45
                
                H1 = H1 + args.Jd*(kron(kron(kron(id2,id2),kron(Ry3@sp@Ry3.t(), Ry4@sm@Ry4.t())),kron(id2,id2)) + \
                                   kron(kron(kron(id2,id2),kron(Ry3@sm@Ry3.t(), Ry4@sp@Ry4.t())),kron(id2,id2)) + \
                                 2*kron(kron(kron(id2,id2),kron(Ry3@sz@Ry3.t(), Ry4@sz@Ry4.t())),kron(id2,id2))) # 34 
                
                H1 = H1 + args.Jh*(kron(kron(kron(Ry1@sp@Ry1.t(),id2),kron(id2, Ry4@sm@ Ry4.t())),kron(id2,id2)) + \
                                   kron(kron(kron(Ry1@sm@Ry1.t(),id2),kron(id2, Ry4@sp@ Ry4.t())),kron(id2,id2)) + \
                                 2*kron(kron(kron(Ry1@sz@Ry1.t(),id2),kron(id2, Ry4@sz@ Ry4.t())),kron(id2,id2)))  # 14 
                
                H1 = H1 + args.Jh*(kron(kron(kron(id2,id2),kron(Ry3@sp@Ry3.t(),id2)),kron( Ry5@sm@Ry5.t(),id2)) + \
                                   kron(kron(kron(id2,id2),kron(Ry3@sm@Ry3.t(),id2)),kron( Ry5@sp@Ry5.t(),id2)) + \
                                 2*kron(kron(kron(id2,id2),kron(Ry3@sz@Ry3.t(),id2)),kron( Ry5@sz@Ry5.t(),id2)))  # 35 ok
                return H1

    
            # #    1 -- 2 
            # #     \ A/ \    
            # #      3  | 4
            # #       \ |/ B\  
            # #         6 -- 5
            # #
            # # Jt: 23, 46 // Jd 26 // Jh 24 36
            # # BA WORKS FOR 1x1 ok for 1x1
            
            def Hy(Ry1, Ry2, Ry3, Ry4, Ry5, Ry6):
                H2 = args.Jt*(kron(kron(kron(id2,Ry2@sp@Ry2.t()),kron(Ry3@sm@Ry3.t(),id2)),kron(id2,id2)) + \
                              kron(kron(kron(id2,Ry2@sm@Ry2.t()),kron(Ry3@sp@Ry3.t(),id2)),kron(id2,id2)) + \
                            2*kron(kron(kron(id2,Ry2@sz@Ry2.t()),kron(Ry3@sz@Ry3.t(),id2)),kron(id2,id2)) )       # 23 ok
                
                H2 = H2 + args.Jt*(kron(kron(kron(id2,id2),kron(id2,Ry4@sp@Ry4.t())),kron(id2, Ry6@sm@Ry6.t())) + \
                                   kron(kron(kron(id2,id2),kron(id2,Ry4@sm@Ry4.t())),kron(id2, Ry6@sp@Ry6.t())) + \
                                 2*kron(kron(kron(id2,id2),kron(id2,Ry4@sz@Ry4.t())),kron(id2, Ry6@sz@Ry6.t())) )   # 46 ok
                
                H2 = H2 + args.Jd*(kron(kron(kron(id2,Ry2@sp@Ry2.t()),kron(id2,id2)),kron(id2, Ry6@sm@Ry6.t())) + \
                                   kron(kron(kron(id2,Ry2@sm@Ry2.t()),kron(id2,id2)),kron(id2, Ry6@sp@Ry6.t())) + \
                                 2*kron(kron(kron(id2,Ry2@sz@Ry2.t()),kron(id2,id2)),kron(id2, Ry6@sz@Ry6.t())))    # 26 ok
                
                H2 = H2 + args.Jh*(kron(kron(kron(id2,Ry2@sp@Ry2.t()),kron(id2, Ry4@sm@Ry4.t())),kron(id2,id2)) + \
                                   kron(kron(kron(id2,Ry2@sm@Ry2.t()),kron(id2, Ry4@sp@Ry4.t())),kron(id2,id2)) + \
                                 2*kron(kron(kron(id2,Ry2@sz@Ry2.t()),kron(id2, Ry4@sz@Ry4.t())),kron(id2,id2)))  # 24 ok
                
                H2 = H2 + args.Jh*(kron(kron(kron(id2,id2),kron(Ry3@sp@Ry3.t(),id2)),kron(id2, Ry6@sm@Ry6.t())) + \
                                   kron(kron(kron(id2,id2),kron(Ry3@sm@Ry3.t(),id2)),kron(id2, Ry6@sp@Ry6.t())) + \
                                 2*kron(kron(kron(id2,id2),kron(Ry3@sz@Ry3.t(),id2)),kron(id2, Ry6@sz@Ry6.t())))  # 36 ok
            
                return H2
    
            # #      4 
            # #    / B\  
            # #   6 -- 5
            # #   | /  |
            # #   1 -- 2
            # #    \ A/ 
            # #      3
            # #
            # # Jt 12 56 // Jd 15 // Jh 16 25
            # # BA  # WORK seems to work for a 1x1 unit cell with flipped B.

            def Hz(Ry1, Ry2, Ry3, Ry4, Ry5, Ry6):

                H3 = args.Jt*(kron(kron(kron(Ry1@sp@Ry1.t(),Ry2@sm@Ry2.t()),kron(id2, id2)),kron(id2,id2)) + \
                              kron(kron(kron(Ry1@sm@Ry1.t(),Ry2@sp@Ry2.t()),kron(id2, id2)),kron(id2,id2)) + \
                            2*kron(kron(kron(Ry1@sz@Ry1.t(),Ry2@sz@Ry2.t()),kron(id2, id2)),kron(id2,id2)) )     # 12
                
                H3 = H3 + args.Jt*(kron(kron(kron(id2,id2),kron(id2,id2)),kron(Ry5@sp@Ry5.t(), Ry6@sm@Ry6.t())) + \
                                   kron(kron(kron(id2,id2),kron(id2,id2)),kron(Ry5@sm@Ry5.t(), Ry6@sp@Ry6.t())) + \
                                 2*kron(kron(kron(id2,id2),kron(id2,id2)),kron(Ry5@sz@Ry5.t(), Ry6@sz@Ry6.t())) )  # 56
                
                H3 = H3 + args.Jd*(kron(kron(kron(Ry1@sp@Ry1.t(),id2),kron(id2, id2)),kron(Ry5@sm@Ry5.t(),id2)) + \
                                   kron(kron(kron(Ry1@sm@Ry1.t(),id2),kron(id2, id2)),kron(Ry5@sp@Ry5.t(),id2)) + \
                                 2*kron(kron(kron(Ry1@sz@Ry1.t(),id2),kron(id2, id2)),kron(Ry5@sz@Ry5.t(),id2)))  # 15
                
                H3 = H3 + args.Jh*(kron(kron(kron(Ry1@sp@Ry1.t(),id2),kron(id2, id2)),kron(id2, Ry6@sm@Ry6.t())) + \
                                   kron(kron(kron(Ry1@sm@Ry1.t(),id2),kron(id2, id2)),kron(id2, Ry6@sp@Ry6.t())) + \
                                 2*kron(kron(kron(Ry1@sz@Ry1.t(),id2),kron(id2, id2)),kron(id2, Ry6@sz@Ry6.t())))  # 16
                
                H3 = H3 + args.Jh*(kron(kron(kron(id2,Ry2@sp@Ry2.t()),kron(id2,id2)),kron( Ry5@sm@Ry5.t(),id2)) + \
                                   kron(kron(kron(id2,Ry2@sm@Ry2.t()),kron(id2,id2)),kron( Ry5@sp@Ry5.t(),id2)) + \
                                 2*kron(kron(kron(id2,Ry2@sz@Ry2.t()),kron(id2,id2)),kron( Ry5@sz@Ry5.t(),id2)))  # 25

                return H3


            # Inside the Need phase.
            print(Ry(np.pi))
            print(Rypi)
            H1 = Hz(id2,id2,id2,Ry(np.pi),Ry(np.pi),Ry(np.pi)) 
            H2 = Hy(id2,id2,id2,Ry(np.pi),Ry(np.pi),Ry(np.pi)) 
            H3 = Hx(id2,id2,id2,Ry(np.pi),Ry(np.pi),Ry(np.pi)) 

            H = [H1/6]
            # H2 = Hy(id2,id2,id2,Rypi,Rypi,Rypi)
            # H1 = Hx(id2,id2,id2,Rypi,Rypi,Rypi)
            # H = [H3]

            # Ruby lattice (120-canted order)
            # pi = np.pi
            # # H1 = Hz(Ry(-1/6*pi), Ry(-3/2*pi), Ry(-5/6*pi),Ry(-0/1*pi),Ry(-2/3*pi), Ry(-4/3*pi))
            # # H2 = Hz(Ry(-3/2*pi), Ry(-5/6*pi), Ry(-1/6*pi),Ry(-4/3*pi),Ry(-0/1*pi), Ry(-2/3*pi))
            # # H3 = Hz(Ry(-5/6*pi), Ry(-1/6*pi), Ry(-3/2*pi),Ry(-2/3*pi),Ry(-4/3*pi), Ry(-0/1*pi))
            
            # H1 = Hz(Ry(-0/1*pi), Ry(-2/3*pi), Ry(-4/3*pi),Ry(-1/6*pi),Ry(-3/2*pi), Ry(-5/6*pi))
            # H2 = Hz(Ry(-4/3*pi), Ry(-0/6*pi), Ry(-2/3*pi),Ry(-3/2*pi),Ry(-5/6*pi), Ry(-1/6*pi))
            # H3 = Hz(Ry(-2/3*pi), Ry(-4/3*pi), Ry(-0/2*pi),Ry(-5/6*pi),Ry(-1/6*pi), Ry(-3/2*pi))
            
            # # H4 = Hx(Ry(-5/6*pi),Ry(-1/6*pi),Ry(-3/2*pi),Ry(-0/1*pi),Ry(-2/3*pi),Ry(-4/3*pi))
            # # H5 = Hx(Ry(-1/6*pi),Ry(-3/2*pi),Ry(-5/6*pi),Ry(-4/3*pi),Ry(-0/1*pi),Ry(-2/3*pi))
            # # H6 = Hx(Ry(-3/2*pi),Ry(-5/6*pi),Ry(-1/6*pi),Ry(-2/3*pi),Ry(-4/3*pi),Ry(-0/1*pi))
            
            # # H7 = Hy(Ry(-3/2*pi),Ry(-5/6*pi),Ry(-1/6*pi),Ry(-0/1*pi),Ry(-2/3*pi),Ry(-4/3*pi))
            # # H8 = Hy(Ry(-5/6*pi),Ry(-1/6*pi),Ry(-3/2*pi),Ry(-4/3*pi),Ry(-0/1*pi),Ry(-2/3*pi))
            # # H9 = Hy(Ry(-1/6*pi),Ry(-3/2*pi),Ry(-5/6*pi),Ry(-2/3*pi),Ry(-4/3*pi),Ry(-0/1*pi))
            
            # # H = [H1,H2,H3,H4,H5,H6,H7,H8,H9]
            # # H = [H1,H2,H3]
            # H1 = Hz(id2,id2,id2,id2,id2,id2)

            Mpx = kron(kron(kron(id2,sx),kron(id2,id2)),kron(id2,id2))
            Mpy = kron(kron(kron(id2,sy),kron(id2,id2)),kron(id2,id2))
            Mpz = kron(kron(kron(id2,sz),kron(id2,id2)),kron(id2,id2))

    elif args.model == 'SO(4)':


            sx = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)*0.5
            sy = torch.tensor([[0, -1], [1, 0]], dtype=dtype, device=device)*0.5
            sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
            sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
            sz = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)*0.5
            id2 = torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device)
            
            def Ry(theta):
                return np.cos(theta/2)*id2 + np.sin(theta/2)*sy*2


            theta = np.pi

            tp = sp
            tm = sm
            tz = sz


            Mpz = kron(kron(sz,id2),kron(id2,id2))
            Mpx = kron(kron(sp,id2),kron(id2,id2))
            Mpy = kron(kron(sm,id2),kron(id2,id2))

            hy =        kron(kron(sp,id2),kron(Ry(theta)@sm@Ry(theta).t(),id2))/2 
            hy = hy +   kron(kron(sm,id2),kron(Ry(theta)@sp@Ry(theta).t(),id2))/2 
            hy = hy +   kron(kron(sz,id2),kron(Ry(theta)@sz@Ry(theta).t(),id2))

            hy = hy +   kron(kron(id2,tp),kron(id2,Ry(theta)@tm@Ry(theta).t()))/2 
            hy = hy +   kron(kron(id2,tm),kron(id2,Ry(theta)@tp@Ry(theta).t()))/2
            hy = hy +   kron(kron(id2,tz),kron(id2,Ry(theta)@tz@Ry(theta).t()))

            hy = hy + 4*(kron(kron(sp,tp),kron(Ry(theta)@sm@Ry(theta).t(),Ry(theta)@tm@Ry(theta).t()))/4 + \
                         kron(kron(sp,tm),kron(Ry(theta)@sm@Ry(theta).t(),Ry(theta)@tp@Ry(theta).t()))/4 + \
                         kron(kron(sp,tz),kron(Ry(theta)@sm@Ry(theta).t(),Ry(theta)@tz@Ry(theta).t()))/2)
            
            hy = hy + 4*(kron(kron(sm,tp),kron(Ry(theta)@sp@Ry(theta).t(),Ry(theta)@tm@Ry(theta).t()))/4 + \
                         kron(kron(sm,tm),kron(Ry(theta)@sp@Ry(theta).t(),Ry(theta)@tp@Ry(theta).t()))/4 + \
                         kron(kron(sm,tz),kron(Ry(theta)@sp@Ry(theta).t(),Ry(theta)@sz@Ry(theta).t()))/2)
            
            hy = hy + 4*(kron(kron(sz,tp),kron(Ry(theta)@sz@Ry(theta).t(),Ry(theta)@tm@Ry(theta).t()))/2 + \
                         kron(kron(sz,tm),kron(Ry(theta)@sz@Ry(theta).t(),Ry(theta)@tp@Ry(theta).t()))/2 + \
                         kron(kron(sz,tz),kron(Ry(theta)@sz@Ry(theta).t(),Ry(theta)@tz@Ry(theta).t())))

            hy = hy + 1/4*kron(kron(id2,id2),kron(id2,id2))

            H = [hy]

    else:
        print ('please, choose a valid model')
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

    with io.open(key + '.log', 'a', buffering=1, newline='\n') as logfile:

        En = 4
        Etmp = 5
        epoch = 0
        while epoch < args.Nepochs:
            epoch = epoch + 1

            # Train step and get loss and magnetization values
            loss, Mx, My, Mz = train_step(H, Mpx, Mpy, Mpz, args, dtype)
           
            # Save checkpoint periodically
            if (epoch % args.save_period == 0):
                save_checkpoint(key + '/peps.tensor'.format(epoch), model, optimizer)

            with torch.no_grad():
                Etmp = En
                En, Mx, My, Mz = model.forward(H, Mpx, Mpy, Mpz, args.chi if args.chi_obs is None else args.chi_obs, dtype)
                Mg = torch.sqrt(Mx**2 + My**2 + Mz**2)
                message = ('{} ' + 5 * '{:.8f} ').format(epoch, En*1.5, Mx, My, Mz, Mg)
                print('epoch, En, Mx, My, Mz, Mg', message)
                logfile.write(message + u'\n')


