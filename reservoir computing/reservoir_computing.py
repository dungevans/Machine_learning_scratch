import torch 
import torch.nn as nn 
import numpy as np 
class Reservoir_computing : 
    def __init__ ( self, input_dim : int , num_input : int ,output_dim : int , learning_rate : float, reservoir_dim : int , spectral_radius : int    ) : 
        """
        the dim of reservoir should be around 1/4 to 1/2 num__input 
        """
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.learning_rate = learning_rate 
        self.reservoir_dim = reservoir_dim 
        self.spectral_radius = spectral_radius
        self.num_input = num_input 
        # khoi tao trong so reservoir
        def create_sparse_matrix(dim, density=0.05):
    # Tạo ma trận toàn 0
            W = np.zeros((dim, dim))
            # Chỉ lấp đầy 5% số phần tử
            num_non_zero = int(dim * dim * density)
            for _ in range(num_non_zero):
                i, j = np.random.randint(0, dim, 2)
                W[i, j] = np.random.uniform(-0.5, 0.5)
            return W
        
        #bo trong so cua reservoir nen co phan phoi xung quanh 0 
    
        self.Wrc = create_sparse_matrix ( reservoir_dim) 
        def scaler ( Weight ) -> torch.Tensor :
            if Weight.shape[0] == Weight.shape[1] : 
                actual_spectral_radius = np.max ( np.abs(np.linalg.eigvals(Weight)) ) 
                Weight = Weight * (self.spectral_radius/actual_spectral_radius)
            else : 
                # expect_singular = 0.2 
                # actual_singular = np.max ( np.abs(np.linalg.svd(Weight)) ) 
                # Weight = Weight * (expect_singular/actual_singular)
                Weight = Weight * 0.2

            if  isinstance (Weight, torch.Tensor) : 
                return Weight
            else : 
                return torch.tensor ( Weight, dtype = torch.float32)  
        self.Wrc = scaler ( self.Wrc )
        self.W_in  = np.random.uniform ( -0.5, 0.5, size = (reservoir_dim,input_dim )  )
        self.W_in  = scaler ( self.W_in)
        self.W_out =nn.Linear(reservoir_dim + input_dim, output_dim, bias = False)


        
        self.cache = {}
    def forward ( self , data ) : 
        T = data.shape[0]
        current_state = torch.zeros( self.reservoir_dim, dtype = torch.float32)
        states_history = []
        for t in range ( T) : 
            u_t = data[t] #shape : (input_dim,)
            input_part = torch.matmul(self.W_in, u_t)
            
            # Wrc @ x(t-1) -> shape: (reservoir_dim,)
            reservoir_part = torch.matmul(self.Wrc, current_state)
            
            
            # x(t) = (1 - alpha) * x(t-1) + alpha * tanh(W_in*u(t) + Wrc*x(t-1))
            new_state = (1 - self.learning_rate) * current_state + self.learning_rate * torch.tanh(input_part + reservoir_part)
            

            current_state = new_state
            states_history.append(current_state)
            
        
        X_states = torch.stack(states_history)
        X_augmented = torch.cat([X_states, data], dim=1)
        

        # Y_pred shape: (T, output_dim)
        Y_pred = self.W_out(X_augmented)
        
        return Y_pred 
    
    
    
    
    
    
    
    
    
        
                
