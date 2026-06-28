import torch  
import torch.nn as nn
import numpy as np 
import math 
class Activate : 
    def __init__ (self, function_name : str  ) :

        self.name = function_name 
        self.last_input = None 

    def forward ( self, x ) :
        if self.name == 'sigmoid' : 
            output = torch.sigmoid ( x )
        if self.name == 'reLu' : 
            output = torch.relu(x)
        
        if self.name == 'tanh' : 
            output = torch.tanh  (x)
        if self.name == 'softmax' : 
            output = torch.softmax(x)
        return output
        


class MLP : 
    def __init__ ( self, dim_in: int , dim_out : int , lr : int ) : 
        
        torch.manual_seed ( 42 )
        self.lr = lr 
        self.dim_in = dim_in 
        self.dim_out = dim_out
        self.W = torch.randn ( self.dim_in , self.dim_out )*0.01
        self.B = torch.randn (self.dim_out)*0.01
        self.act = Activate(function_name='reLu') 
        self.cache = { }


    def forward ( self, x) : 
        output = torch.matmul(x, self.W) + self.B 
        output_act = self.act.forward(output)
        self.cache['data'] = x 
        self.cache['output_prev_layer'] = output 
        return output_act


        
    

    def backward ( self, grad_output , lr  ) :
        """
        grad_output: gradient từ lớp phía sau truyền tới (dL/dA)
        """

        d_act = (self.cache['output_prev_layer'] > 0).float() 
        dz = grad_output * d_act
        
        
        dW = torch.matmul(self.cache['data'].t(), dz)
        dB = torch.sum(dz, dim=0)
        
        
        dx = torch.matmul(dz, self.W.t())
        
        
        with torch.no_grad():
            self.W -= lr * dW
            self.B -= lr * dB
            
        return dx





