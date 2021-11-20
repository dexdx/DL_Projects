import dlc_practical_prologue
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

'''
Currently includes:
    - FCC 
    - Siamese

Both with the option to share weights and to use and auxiliary function
'''

class FCC(nn.Module):
    # Fully connected components
    def __init__(self, aux=False, share=False):
        super(FCC, self).__init__()
        
        # Parameters that will define weight sharing 
        # And the use of auxiliary losses
        self.aux = aux
        self.share = share
        
        # Define architecture for first part of the network 
        self.LeNet1 = nn.Sequential(
            nn.Linear(196, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()            
        )
        
        # Copy of LeNet1, to be used of there is no weight sharing
        self.LeNet1_v2 = nn.Sequential(
            nn.Linear(196, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()            
        )
        
        # Second part of the network, that will combine the outputs
        # of the two images
        self.LeNet2 = nn.Sequential(
            nn.Linear(64, 2),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
        
        # Auxiliary digit classification
        self.Aux = nn.Sequential(
            nn.Linear(32, 16),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 10),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(10),
            nn.ReLU()
        )
        
        # To be used if no weight sharing
        self.Aux_v2 = nn.Sequential(
            nn.Linear(32, 16),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 10),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(10),
            nn.ReLU()
        )
    
    def forward(self, x1, x2):  
        # Flatten both images
        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)
        
        # Forward pass of first image through first network
        x1 = self.LeNet1(x1)
        # Forward pass of second image (to decide whether through)
        # first or second network
        if self.share:
            x2 = self.LeNet1(x2)
        else:
            x2 = self.LeNet1_v2(x2)
        # Concatenate outputs
        x = torch.cat((x1.view(x1.size()[0], -1), x2.view(x1.size()[0], -1)), dim = 1)
        # Second part of network
        x = self.LeNet2(x)
        
        # If necessary caclulate outputs from auxiliary networks
        if(self.aux):
            if self.share:
                x1 = self.Aux(x1)
                x2 = self.Aux(x2)
            else:
                x1 = self.Aux(x1)
                x2 = self.Aux_v2(x2)
            
        return x1,x2, x
            
    # Functions to manually set the use of weight sharing and auxiliary networks
    def set_auxillary(self):
        self.aux = True
        
    def set_sharing(self):
        self.share = True
    

class Siamese(nn.Module):
    # Convolutional, 'Siamese' network
    def __init__(self, aux=False, share=False):
        super(Siamese, self).__init__()
        
        # Parameters to set the use of auxiliary losses and weight sharing 
        # during instantiation
        self.aux = aux
        self.share = share

        # Fully connected section of the network
        self.LeNet1_x1 = nn.Sequential(
            nn.Conv2d(1,16,3),  # 16x12x12 (input is 1x14x14)
            nn.MaxPool2d(2),    # 16x6x6
            nn.Dropout(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3), # 32x4x4
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3), # 64x2x2 (256)
            nn.BatchNorm2d(64),
            nn.ReLU()        
        )
        
        # Copy of Fully connected section of the network
        self.LeNet2_x1 = nn.Sequential(
            nn.Linear(256,128),  # 1x64
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(128,64),   # 1x32
            nn.Dropout(),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(64,32),   # 1x32
            nn.Dropout(),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )        
        
        # Convolutional section of the network
        self.LeNet1_x2 = nn.Sequential(
            nn.Conv2d(1,16,3),  # 16x12x12 (input is 1x14x14)
            nn.MaxPool2d(2),    # 16x6x6
            nn.Dropout(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3), # 32x5x5
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,3), # 32x5x5
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Copy of Convolutional section of the network
        self.LeNet2_x2 = nn.Sequential(
            nn.Linear(256,128),  # 1x64
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(128,64),   # 1x32
            nn.Dropout(),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(64,32),   # 1x32
            nn.Dropout(),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        
        # Auxiliary linear net for classification
        self.AuxLayer_x1 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(16,10)     # 1x10
        )
        # Auxiliary linear net for classification
        self.AuxLayer_x2 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(16,10)     # 1x10
        )
        
        # Final section of the layer, after getting difference
        self.LeNet3 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(16,2)     # 1x2
        )
        
    # Forward pass of first image
    def forward_x2(self, x):
        x = self.LeNet1_x2(x)
        x = x.view(-1,1,256)
        x = self.LeNet2_x2(x)
        
        return x  
        
    # Forward pass of second image (to be used if no weightsharing)
    def forward_x1(self, x):
        x = self.LeNet1_x1(x)
        x = x.view(-1,1,256)
        x = self.LeNet2_x1(x)
        return x
    
    # Overall forward pass
    def forward(self, x1, x2):
        if self.share:
            x1 = self.forward_x1(x1)
            x2 = self.forward_x1(x2)
        else:
            x1 = self.forward_x1(x1)
            x2 = self.forward_x2(x2)
        x3 = x2 - x1
        x3 = self.LeNet3(x3)
        if(self.aux):
            if self.share:
                x1 = self.AuxLayer_x1(x1)
                x2 = self.AuxLayer_x1(x2)
            else:
                x1 = self.AuxLayer_x1(x1)
                x2 = self.AuxLayer_x2(x2)
        return x1,x2,x3
    
    # Functions to set auxialiary loss and weight sharing manuallt
    def set_auxillary(self):
        self.aux = True
        
    def set_sharing(self):
        self.share = True