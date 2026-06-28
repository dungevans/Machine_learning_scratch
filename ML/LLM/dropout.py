import numpy as np 
import torch 
import torch.nn as nn 

def dropout(x, p=0.5, rng=None)-> np.ndarray :
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.ndarray (x)
    dim = x.shape 
    rows =dim[0]
    cols = dim[1]
    result = np.random.randn ( dim ) 
    
    for i in range ( rows ) :
        np.random.seed ( 123 )

        result[i, : ] =  pos =  np.linspace ( 0,1 , cols )

    return result 
