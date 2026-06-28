import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    position = np.arange ( 0, seq_length )
    cols = np.arange ( 0, d_model )
    W_PE = np.zeros((seq_length, d_model ))
    
    cols = cols//2 
    common = np.exp(np.log(10000)*2*cols/d_model )
    for col in cols : 
        if col % 2 == 0 : 
            W_PE [:, col] = np.sin ( position/common)
        else :
            W_PE [:, col] = np.cos ( position/common)

    return W_PE 
#test 
w_pe = positional_encoding(seq_length = 4, d_model = 6)  
print  ( w_pe)
        



