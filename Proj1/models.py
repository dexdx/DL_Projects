import dlc_practical_prologue
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

'''
Currently includes:
    - FCC 
    - FCC_aux
    - Siamese
    - Siamese no weight sharing 
    - Siamese aux
'''

class FCC(nn.Module):

    def __init__(self):
        super(FCC, self).__init__()
        
        self.aux = False
        
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
        self.LeNet2 = nn.Sequential(
            nn.Linear(64, 2),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
        
        self.Aux = nn.Sequential(
            nn.Linear(32, 16),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 10),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(10),
            nn.ReLU()
        )
    
    def forward(self, x1, x2):  
        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)
        
        x1 = self.LeNet1(x1)
        x2 = self.LeNet1(x2)
        x = torch.cat((x1.view(x1.size()[0], -1), x2.view(x1.size()[0], -1)), dim = 1)
        x = self.LeNet2(x)
        if(self.aux):            
            x1 = self.Aux(x1)
            x2 = self.Aux(x2)
            
        return x1,x2, x
        
    
    
    def setAuxillary(self):
        self.aux = True

class Siamese_no_sharing(nn.Module):

    def __init__(self):
        super(Siamese_no_sharing, self).__init__()
        
        self.aux = False
        self.LeNet1_x1 = nn.Sequential(
            nn.Conv2d(1,16,5),  # 16x10x10 (input is 1x14x14)
            nn.MaxPool2d(2),    # 16x5x5
            nn.Dropout(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,2), # 32x4x4
            nn.MaxPool2d(2),    # 32x2x2 (-> 1x128 before LeNet2)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.LeNet2_x1 = nn.Sequential(
            nn.Linear(128,64),  # 1x64
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(64,32),   # 1x32
            nn.Dropout(),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )        
        
        self.LeNet1_x2 = nn.Sequential(
            nn.Conv2d(1,16,5),  # 16x10x10 (input is 1x14x14)
            nn.MaxPool2d(2),    # 16x5x5
            nn.Dropout(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,2), # 32x4x4
            nn.MaxPool2d(2),    # 32x2x2 (-> 1x128 before LeNet2)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.LeNet2_x2 = nn.Sequential(
            nn.Linear(128,64),  # 1x64
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(64,32),   # 1x32
            nn.Dropout(),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.AuxLayer_x1 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(16,10)     # 1x10
        )
        self.AuxLayer_x2 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(16,10)     # 1x10
        )
        
        self.LeNet3 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(16,2)     # 1x2
        )
        
    def forward_x2(self, x):
        x = self.LeNet1_x2(x)
        x = x.view(-1,1,128)
        x = self.LeNet2_x2(x)
        return x  
        
        
    def forward_x1(self, x):
        x = self.LeNet1_x1(x)
        x = x.view(-1,1,128)
        x = self.LeNet2_x1(x)
        return x
    
    def forward(self, x1, x2):
        x1 = self.forward_x1(x1)
        x2 = self.forward_x2(x2)
        x3 = x2 - x1
        x3 = self.LeNet3(x3)
        if(self.aux):
            x1 = self.AuxLayer_x1(x1)
            x2 = self.AuxLayer_x2(x2)
        return x1,x2,x3
    
    def setAuxillary(self):
        self.aux = True
    
    
# Weight-sharing "Siamese" LeNet
class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.aux = False
        self.LeNet1 = nn.Sequential(
            nn.Conv2d(1,16,5),  # 16x10x10 (input is 1x14x14)
            nn.MaxPool2d(2),    # 16x5x5
            nn.Dropout(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,2), # 32x4x4
            nn.MaxPool2d(2),    # 32x2x2 (-> 1x128 before LeNet2)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.LeNet2 = nn.Sequential(
            nn.Linear(128,64),  # 1x64
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(64,32),   # 1x32
            nn.Dropout(),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.LeNet3 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(16,2)     # 1x2
        )
        self.AuxLayer = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.BatchNorm1d(1),
            nn.ReLU(),
            nn.Linear(16,10)     # 1x10
        )
        
    def forward_bro(self, x):
        x = self.LeNet1(x)
        x = x.view(-1,1,128)
        x = self.LeNet2(x)
        return x
    
    def forward(self, x1, x2):
        x1 = self.forward_bro(x1)
        x2 = self.forward_bro(x2)
        x3 = x2 - x1
        if(self.aux):
            x1 = self.AuxLayer(x1)
            x2 = self.AuxLayer(x2)
        x3 = self.LeNet3(x3)
        return x1,x2,x3
    
    def setAuxillary(self):
        self.aux = True