from torch import empty
import math
# Loss function
class LossMSE(object):
    
    def __init__(self, prev_module = None):
        self.prev_module =  prev_module
    
    def set_truth(self,y_true):
        if (len(y_true.shape) >2): 
            y_true = y_true.squeeze()
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
class ReLU(object):
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


class Tanh(object):
    
    def __init__(self, prev_module = None):
        self.prev_module =  prev_module
        self.curr_grad = 0 #Temporary

    def forward (self , input_):
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
    

class Sigmoid(object):    
    def __init__(self, prev_module = None):
        self.prev_module =  prev_module
        self.curr_grad = 0 #Temporary

    def forward (self , input_):
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
    
    
class Dropout(object):    
    def __init__(self, prev_module = None, chance = 0.5):
        self.prev_module =  prev_module
        self.chance = chance

    def forward (self , input_ ,training = True):
        if (not training):
            return input_        
        self.curr_grad = empty(input_.shape).uniform_(0,1) > self.chance
        return input_ * self.curr_grad
        
    def backward (self , gradwrtoutput):
        #Calculate gradient
        grad = self.curr_grad * gradwrtoutput
        
        #Call backward() for previous module
        if self.prev_module is not None:
            prev_grads = self.prev_module.backward(grad)
    
    def param ( self ):
        return []
    
class BatchNorm(object):    
    def __init__(self, input_size, running_mean_momentum = 0.1,prev_module = None, lr=1e-1,momentum = 0):
        self.prev_module =  prev_module
        self.scale_weight = empty(input_size).fill_(1)
        self.translation_weight = empty(input_size).fill_(0)
        self.running_batch_mean = -1 # -1 signals they are not set yet
        self.running_batch_var = -1  # -1 signals they are not set yet
        self.running_started = False
        self.rmm = running_mean_momentum  #Used for calculating the running statistics not parameter updating
        self.lr = lr  #Used for parameter updating
        self.momentum = momentum #Used for parameter updating
        self.input_size = input_size
        self.scale_increment = empty(self.scale_weight.shape).fill_(0)  #Must be initialized to 0
        self.translation_increment = empty(self.translation_weight.shape).fill_(0) #Must be initialized to 0
        
    def forward (self , input_ ,training = True):
        assert input_.shape[1] == self.input_size, "Input size must match!" 
        if (not training):
            curr_mean = self.running_batch_mean
            curr_var = self.running_batch_var
        else:
            #Normalize each input dimension in the batch
            curr_mean = input_.mean(dim = 0)
            curr_var = input_.var(dim = 0)
            
        normalized_input = (input_ - curr_mean.expand(input_.shape)) / (curr_var + 1e-5).sqrt().expand(input_.shape)
        #Update running statistics
        if(not self.running_started):
            self.running_batch_mean = curr_mean
            self.running_batch_var = curr_var
            self.running_started = True
        elif(training):
            self.running_batch_mean = self.rmm *curr_mean + (1-self.rmm) * self.running_batch_mean
            self.running_batch_var =  self.rmm * curr_var + (1-self.rmm) * self.running_batch_var

        #Scale them back          
        scaled_input = normalized_input * self.scale_weight.expand(input_.shape) + self.translation_weight.expand(input_.shape) 

        #Remember for the backwards pass                
        self.last_var = curr_var
        self.normalized_input = normalized_input
        return scaled_input
    
    def update(self, gradwrtoutput):
        
        self.scale_increment = self.momentum * self.scale_increment + self.lr * ( self.normalized_input * gradwrtoutput).mean(0)
        self.translation_increment = self.momentum * self.translation_increment - self.lr * gradwrtoutput.mean(0)
        
        self.scale_weight -=  self.scale_increment
        self.translation_weight -=  self.translation_increment
        
    def backward (self , gradwrtoutput):
        #Calculate gradient
        grad = gradwrtoutput * (self.scale_weight / (self.last_var + 1e-5).sqrt()).expand(gradwrtoutput.shape)
        
        #update weights
        self.update(gradwrtoutput)  #This is the correct version
        
        #Call backward() for previous module
        if self.prev_module is not None:
            prev_grads = self.prev_module.backward(grad)
    
    def param ( self ):
        return [self.scale_weight,self.translation_weight,self.running_batch_mean,self.running_batch_var]


# Linear/fully connected layer
class FCC(object):
    
    def __init__(self, input_size, output_size, prev_module = None, lr=1e-1,momentum = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.prev_module =  prev_module
        self.momentum = momentum
        self.lr = lr
        self.batch_size = 1
        
        # Uniform initialization
        self.weights = empty(input_size, output_size).uniform_(-1* math.sqrt(1/input_size),math.sqrt(1/input_size) )
        self.bias = empty(1,output_size).uniform_(-1* math.sqrt(1/input_size),math.sqrt(1/input_size) )
        self.initial_weights = self.weights
        
        self.weight_increment = empty(self.weights.shape).fill_(0) #Must be initialized to 0
        self.bias_increment = empty(self.bias.shape).fill_(0) #Must be initialized to 0

    def forward (self , input_):
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
        self.update(gradwrtoutput)  #This is the correct version
        
        #Call backward() for previous module
        if self.prev_module is not None:
            prev_grads = self.prev_module.backward(grad)
    
    def update(self, gradwrtoutput):
        self.weight_increment =  self.momentum * self.weight_increment + self.lr * ( self.curr_input.T @ gradwrtoutput ) / self.batch_size
        self.bias_increment = self.momentum * self.bias_increment + self.lr * gradwrtoutput.mean(0,True)
        
        self.weights -=  self.weight_increment
        self.bias -= self.bias_increment
        
    def param ( self ):
        return [self.weights, self.bias]
    
    def initials(self):
        return self.initial_weights


# Sequential builder
class Sequential(object):
    def __init__(self, layer_list, arguments, loss='MSE', lr = 2 * 1e-2,momentum = 0):
        self.layers = []
        last_layer = None    
        self.special_layers = [] #Layers where training and eval have different forward fucntionality such as Dropout and BatchNorm
        for idx ,layer_name in enumerate(layer_list):
            if(layer_name == 'FCC'):
                assert arguments[idx] != [], "FCC requires a tuple as input!"
                curr_layer =FCC(arguments[idx][0], arguments[idx][1], last_layer, lr=lr ,momentum = momentum)
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
                
            elif(layer_name == 'Dropout'):
                assert arguments[idx] != [], "Dropout requires a probability as input!"

                curr_layer = Dropout(last_layer,arguments[idx][0])
                self.layers.append(curr_layer)
                last_layer = curr_layer
                self.special_layers.append(idx)
            elif(layer_name == 'BatchNorm'):
                assert arguments[idx] != [], "BatchNorm requires a tuple: input size and running mean momentum as inputs!"
                assert len(arguments[idx]) == 2, "BatchNorm requires a tuple: input size and running mean momentum as inputs!"
                curr_layer = BatchNorm(arguments[idx][0], arguments[idx][1], last_layer, lr=lr, momentum = momentum)
                self.layers.append(curr_layer)
                last_layer = curr_layer
                self.special_layers.append(idx)
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
        for idx,layer in enumerate(self.layers[:-1]):
            if (idx in self.special_layers):
                out = layer.forward(out,False)
            else:
                out = layer.forward(out)
        return out
    
    def get_inits(self):
        return self.inits

    
def modelTester(model, test_input,test_target,test_batch):    
    acc = 0
    count = 0
    test_size = test_input.size(0)
    minibatch = test_batch
    for i in range(0, test_size, minibatch):
        truth = test_target.narrow(0, i, minibatch)
        inp = test_input.narrow(0, i, minibatch)
        out = model.eval(inp)
        acc += accuracy_count(out, truth)

    return acc/test_size

def modelTrainer(model, num_epoch = 100,train_input =None ,train_target =None,val_input = None,val_target = None, train_batch = 20, valid_batch = 0):
    
    loss_track = []
    val_acc_track = []
    train_acc_track = []
    for epoch in range(num_epoch):
        minibatch = train_batch
        for i in range(0, train_input.size(0), minibatch):
            out,loss = model.train(train_input.narrow(0, i, minibatch), train_target.narrow(0, i, minibatch).unsqueeze(1))
        loss_track.append(loss.item())
        train_acc_track.append(modelTester(model, train_input,train_target,train_batch))        
        
        if (val_input is not None):                    
            val_acc_track.append(modelTester(model, val_input,val_target,valid_batch))
    return model, loss_track, [train_acc_track, val_acc_track]


def accuracy_count(pred,true):
    pred = pred.squeeze()
    if(len(pred.shape) > 1):        
        pred = (pred[:,0] > pred[:,1]).long().squeeze()
        true = true[:,0].long().squeeze()
        return (pred.size(0) - (pred-true).abs().sum()).item()
    else:
        pred = (pred > 0.5).long().view(-1)
        true = true.long().view(-1)
        return (pred.size(0) - (pred-true).abs().sum()).item()