from torch import empty
import math


# Parent
class Module ( object ):
    
    def forward (self , input_ ):
        return input_
    def backward (self, grad):
        
        #Call backward() for previous module
        if self.prev_module is not None:
            prev_grads = self.prev_module.backward(grad)
            
    def param ( self ):
        return []


# Loss function
class LossMSE(Module):
    
    def __init__(self, prev_module = None):
        self.prev_module =  prev_module
    
    def set_truth(self,y_true):
        self.y_true = y_true
    
    def forward (self , input_ ):
        assert input_.shape[0] == self.y_true.shape[0], "Batch size must match!"
        assert input_.shape[1] == self.y_true.shape[1], "Input and output size must match!"
        self.curr_input = input_
        return (self.y_true-input_).square().mean(1,True)  #Average per input not accross batches!
    
    def backward (self):
        #Calculate gradient

        grad = -2 *(self.y_true-self.curr_input) / (self.curr_input.shape[1])   #Divide by number of output samples not batch size     

        #Call backward() for previous module
        if self.prev_module is not None:
            prev_grads = self.prev_module.backward(grad)
    
    def param ( self ):
        return []


# Activation functions
class ReLU(Module):
    
    def __init__(self, prev_module = None):
        self.prev_module =  prev_module
        self.curr_grad = 0 #Temporary

    def forward (self , input_ ):
        self.curr_grad = (input_ > 0)
        return input_ * self.curr_grad
        
    def backward (self , gradwrtoutput):
        #Calculate gradient
        grad = self.curr_grad * gradwrtoutput
        
        #Call backward() for previous module
        if self.prev_module is not None:
            prev_grads = self.prev_module.backward(grad)
    
    def param ( self ):
        return []


class Tanh(Module):
    
    def __init__(self, prev_module = None):
        self.prev_module =  prev_module
        self.curr_grad = 0 #Temporary

    def forward (self , input_ ):
        self.curr_grad = input_.tanh()
        return self.curr_grad
        
    def backward (self , gradwrtoutput):
        #Calculate gradient        
        grad = self.curr_grad.tanh().pow(2).multiply(-1).add(1) * gradwrtoutput
        
        #Call backward() for previous module
        if self.prev_module is not None:
            prev_grads = self.prev_module.backward(grad)
    
    def param ( self ):
        return []
    

class Sigmoid(Module):    
    def __init__(self, prev_module = None):
        self.prev_module =  prev_module
        self.curr_grad = 0 #Temporary

    def forward (self , input_ ):
        self.curr_grad = input_.sigmoid()
        return self.curr_grad
        
    def backward (self , gradwrtoutput):
        #Calculate gradient
        sig = self.curr_grad.sigmoid()
        grad = sig * sig.multiply(-1).add(1) * gradwrtoutput
        
        #Call backward() for previous module
        if self.prev_module is not None:
            prev_grads = self.prev_module.backward(grad)
    
    def param ( self ):
        return []


# Linear/fully connected layer
class FCC(Module):
    
    def __init__(self, input_size, output_size, prev_module = None, lr=1e-1, N = None,init_weights = None):
        self.input_size = input_size
        self.output_size = output_size
        self.prev_module =  prev_module

        # Uniform initialization
        # self.weights = empty(input_size, output_size).normal_(0, math.sqrt(2/(input_size + output_size)))
        self.weights = empty(input_size, output_size).uniform_(-1* math.sqrt(1/input_size),math.sqrt(1/input_size) )
        self.bias = empty(1,output_size).uniform_(-1* math.sqrt(1/input_size),math.sqrt(1/input_size) )
        self.initial_weights = self.weights
        self.curr_input = 0
        self.lr = lr
        self.batch_size = 1

    def forward (self , input_ ):
        assert input_.shape[1] == self.input_size, "Input size must match!" 
        out = input_ @ (self.weights) 
        out += self.bias
        assert out.shape[1] == self.output_size, "Output size must match!" 
        assert out.shape[0] == input_.shape[0], "Batch size is not consistent!"
        self.curr_input = input_
        self.batch_size = input_.shape[0]
        return out
        
    def backward (self , gradwrtoutput):
        #Calculate gradient
        grad = gradwrtoutput @ (self.weights.T)
        
        #update weights
        self.update(gradwrtoutput, self.lr)  #This is the correct version
        #self.update(grad, self.lr)
        
        #Call backward() for previous module
        if self.prev_module is not None:
            prev_grads = self.prev_module.backward(grad)
    
    def update(self, gradwrtoutput, learning_rate):
        self.weights -= learning_rate * ( self.curr_input.T @ gradwrtoutput ) / self.batch_size
        self.bias -= learning_rate * gradwrtoutput.mean(0,True)
        
    def param ( self ):
        return [self.weights, self.bias]
    
    def initials(self):
        return self.initial_weights


# Sequential builder
class Sequential(object):
    def __init__(self, layer_list, arguments, loss='MSE', lr = 2 * 1e-2):
        self.layers = []
        last_layer = None        
        for idx ,layer_name in enumerate(layer_list):
            if(layer_name == 'FCC'):
                assert arguments[idx] != [], "FCC requires a tuple as input!"
                curr_layer =FCC(arguments[idx][0], arguments[idx][1], last_layer, lr=lr)
                self.inits = curr_layer.initials()
                self.layers.append(curr_layer)
                last_layer = curr_layer
            elif(layer_name == 'ReLU'):
                assert arguments[idx] == [], "Relu requires no input!"
                curr_layer = ReLU(last_layer)
                self.layers.append(curr_layer)
                last_layer = curr_layer
            elif(layer_name == 'Tanh'):
                assert arguments[idx] == [], "Tanh requires no input!"
                
                curr_layer = Tanh(last_layer)
                self.layers.append(curr_layer)
                last_layer = curr_layer
                
            elif(layer_name == 'Sigmoid'):
                assert arguments[idx] == [], "Sigmoid requires no input!"

                curr_layer = Sigmoid(last_layer)
                self.layers.append(curr_layer)
                last_layer = curr_layer
                
            else:
                raise Exception("No Module matches the input")

        if loss == 'MSE':
            curr_layer = LossMSE(last_layer)
            self.layers.append(curr_layer)
        else:
            raise Exception("No Loss matches the input")
                
    def train(self,input_, g_truth):
        out = input_
        self.layers[-1].set_truth(g_truth)
        for layer in self.layers[:-1]:
            out = layer.forward(out)
        loss = self.layers[-1].forward(out)
        self.layers[-1].backward()  
        return out,loss.mean()
        
    def eval(self,input_):
        out = input_
        for layer in self.layers[:-1]:
            out = layer.forward(out)
        return out
    
    def get_inits(self):
        return self.inits