import numpy as np 
import torch 
import torch.nn as nn 
def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Single RNN cell forward pass.
    """
    W_xh_tranpose = W_xh.transpose()
    W_hh_tranpose = W_hh.transpose()
    ht = np.tanh ( x_t@W_xh_tranpose + h_prev@W_hh_tranpose+b_h )
    return ht 

x_t = np.array([1.0, 0.5], dtype = float)
h_prev =np.array ([0.0, 0.0], dtype = float )
W_xh =np.array([[0.1, 0.2], [0.3, 0.4]], dtype = float ) 
W_hh = np.array([[0.5, 0.1], [0.2, 0.3]], dtype = float ) 
b_h = np.array([0.0, 0.0], dtype = float )  
hidden_states = rnn_cell ( x_t, h_prev,W_xh, W_hh, b_h )
print ( hidden_states)