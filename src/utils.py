import re
import torch
import sys 

def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2



def save_checkpoint(checkpoint_path, model, optimizer):
    
    """
        ARGS: 
            - checkpoint_path
            - model
            - optimiser

        Saves the model as a dictionary, with the buffers and nn.parameters with key state_dict, and 
        optimiser.
    """
    state = {'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}
    
    torch.save(state, checkpoint_path)




def load_checkpoint(checkpoint_path, args, model):

    d, D = args.d, args.D
    dtype, device = model.A1.dtype, model.A1.device

    print('load old model from %s ' % checkpoint_path )
    state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage, weights_only=False)
    match =  re.search('_D([0-9]*)', checkpoint_path)

    if match:

        print( 'Dold = ', match.group(1))
        Dold = int(re.search('_D([0-9]*)', checkpoint_path).group(1))

        Aold = state['state_dict']['final_A1']
        Bold = state['state_dict']['final_A2']

        B1 = 1E-2*torch.rand( d, D, D, D, dtype=dtype, device=device)
        B1[:, :Dold, :Dold, :Dold] = Aold

        B2 = 1E-2*torch.rand( D, D, D, dtype=dtype, device=device)
        B2[:Dold, :Dold, :Dold] = Bold
       
        state_dict = model.state_dict()
        state_dict['A1'] = B1
        state_dict['A2'] = B2
        state_dict['final_A1'] = B1
        state_dict['final_A2'] = B2

        model.load_state_dict(state_dict)

    else:
        model.load_state_dict(state['state_dict'])

