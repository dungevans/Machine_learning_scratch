import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    """
    Initialize the hidden state for an RNN.
    """
    # YOUR CODE HERE
    hidden_states = np.zeros ((batch_size, hidden_dim), dtype = float )
    return hidden_states
a = init_hidden ( 2 ,2 ) 
print ( a )
print ( a.shape ) 