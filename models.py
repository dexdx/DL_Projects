import dlc_practical_prologue
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

'''
Currently includes:
    - FCC 
    - Siamese
    - Siamese no weight sharing 
'''

class FCC(nn.Module):

    def __init__(self):
        super(FCC, self).__init__()
        
#         self.LeNet = nn.Sequential(
#             nn.Linear(392,1024),  # 16x12x12 (input is 1x14x14)
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024,512),  # 16x12x12 (input is 1x14x14)
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 128),  # 16x12x12 (input is 1x14x14)
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 2),  # 16x12x12 (input is 1x14x14)
#             nn.BatchNorm1d(2),
#             nn.ReLU(),
#         )
        
        self.LeNet = nn.Sequential(
            nn.Linear(392, 2056),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(2056),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2056,1024),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 64),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 2),  # 16x12x12 (input is 1x14x14)
            nn.BatchNorm1d(2),
            nn.ReLU(),
        )
    
    def forward(self, x1, x2):
#         print(x1.view(-1).size())
        x = torch.cat((x1.view(x1.size()[0], -1), x2.view(x1.size()[0], -1)), dim = 1)
        return self.LeNet(x)
    
class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        
#         self.LeNet1 = nn.Sequential(
#             nn.Conv2d(1,16,5),  # 16x10x10 (input is 1x14x14)
#             nn.MaxPool2d(2),    # 16x5x5
#             nn.BatchNorm2d(16)
#             nn.ReLU(),
#             nn.Conv2d(16,32,2), # 32x4x4
#             nn.MaxPool2d(2),    # 32x2x2 (-> 1x128 before LeNet2)
#             nn.BatchNorm2d(32)
#             nn.ReLU()
#         )
        self.LeNet1 = nn.Sequential(
            nn.Conv2d(1,8,3),  # 16x12x12 (input is 1x14x14)
            nn.MaxPool2d(2),    # 16x6x6
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8,16,3), # 32x4x4
#             nn.MaxPool2d(2),    # 32x2x2 (-> 1x128 before LeNet2)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3), # 64x2x2
#             nn.MaxPool2d(2),    # 32x2x2 (-> 1x128 before LeNet2)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.LeNet2 = nn.Sequential(
            nn.Linear(128,64),  # 1x64
            nn.ReLU(),
            nn.Linear(64,32),   # 1x32
            nn.ReLU()
        )
        self.LeNet3 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.Sigmoid(),
            nn.Linear(16,2)     # 1x2
        )
        
        
class Siamese_no_sharing(nn.Module):

    def __init__(self):
        super(Siamese_no_sharing, self).__init__()
        
        self.LeNet1_x1 = nn.Sequential(
            nn.Conv2d(1,16,5),  # 16x10x10 (input is 1x14x14)
            nn.MaxPool2d(2),    # 16x5x5
            nn.ReLU(),
            nn.Conv2d(16,32,2), # 32x4x4
            nn.MaxPool2d(2),    # 32x2x2 (-> 1x128 before LeNet2)
            nn.ReLU()
        )
        self.LeNet2_x1 = nn.Sequential(
            nn.Linear(128,64),  # 1x64
            nn.ReLU(),
            nn.Linear(64,32),   # 1x32
            nn.ReLU()
        )        
        
        self.LeNet1_x2 = nn.Sequential(
            nn.Conv2d(1,16,5),  # 16x10x10 (input is 1x14x14)
            nn.MaxPool2d(2),    # 16x5x5
            nn.ReLU(),
            nn.Conv2d(16,32,2), # 32x4x4
            nn.MaxPool2d(2),    # 32x2x2 (-> 1x128 before LeNet2)
            nn.ReLU()
        )
        self.LeNet2_x2 = nn.Sequential(
            nn.Linear(128,64),  # 1x64
            nn.ReLU(),
            nn.Linear(64,32),   # 1x32
            nn.ReLU()
        )
        
        self.LeNet3 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.Sigmoid(),
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
        return x3
    
    
class Siamese_ws_auxilary(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        
        self.LeNet1 = nn.Sequential(
            nn.Conv2d(1,16,5),  # 16x10x10 (input is 1x14x14)
            nn.MaxPool2d(2),    # 16x5x5
            nn.ReLU(),
            nn.Conv2d(16,32,2), # 32x4x4
            nn.MaxPool2d(2),    # 32x2x2 (-> 1x128 before LeNet2)
            nn.ReLU()
        )
        self.LeNet2 = nn.Sequential(
            nn.Linear(128,64),  # 1x64
            nn.ReLU(),
            nn.Linear(64,32),   # 1x32
            nn.ReLU()
        )
        self.LeNet3 = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.Sigmoid(),
            nn.Linear(16,2)     # 1x2
        )
        self.AuxLayer = nn.Sequential(
            nn.Linear(32,16),   # 1x16
            nn.Sigmoid(),
            nn.Linear(16,10)     # 1x2
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
        x1 = self.AuxLayer(x1)
        x2 = self.AuxLayer(x2)
        x3 = self.LeNet3(x3)
        return x1,x2,x3