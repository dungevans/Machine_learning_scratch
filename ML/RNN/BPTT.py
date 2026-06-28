import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    # YOUR CODE HERE
    dtanh = (1-h_t**2)*dh_next 
    dW_hh = dtanh.T  @ h_prev
    dh_prev = dtanh @ W_hh
    return (dh_prev, dW_hh)
dh_next = np.array([[1.0, 0.0]])
h_t = np.array([[0.5, -0.3]])
h_prev = np.array([[0.2, 0.4]])
x_t = np.array([[1.0, 0.5]])
W_hh = np.array([[0.5, 0.1], [0.2, 0.3]])
result = bptt_single_step ( dh_next, h_t, h_prev, x_t, W_hh)
print (result) 