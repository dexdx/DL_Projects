import dlc_practical_prologue
import torch
from torch import nn
from torch import optim
import torchvision
from torch.nn import functional as F
from tqdm import trange


class Proj1():
    def train_model(model, train_input, train_target, batch_size, nb_epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = 1e-1)

        for e in range(nb_epochs):
            acc_loss = 0
            for b in range(0, train_input.size(0), batch_size):
                imgs = train_input.narrow(0, b, batch_size)
                imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
                imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
                output = model(imgs1, imgs2).view(batch_size, -1)
                loss = criterion(output, train_target.narrow(0, b, batch_size))
                acc_loss += loss.item()
                model.zero_grad()
                loss.backward()
                optimizer.step()
            #print(e, acc_loss)

    def train_model_aux(model, train_input, train_target, train_classes, batch_size, nb_epochs):
        criterion = nn.CrossEntropyLoss()
        aux_criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr = 1e-1, weight_decay = 0.0002)
    #     optimizer = optim.SGD(model.parameters(), lr = 1e-1, weight_decay = 0.0001)

        for e in range(nb_epochs):
            acc_loss = 0
            for b in range(0, train_input.size(0), batch_size):
                imgs = train_input.narrow(0, b, batch_size)
                imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
                imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
                x1_pred, x2_pred , output = model(imgs1, imgs2)
                x1_pred = x1_pred.view(batch_size, -1)
                x2_pred = x2_pred.view(batch_size, -1)
                output = output.view(batch_size, -1)
                loss = criterion(output, train_target.narrow(0, b, batch_size))
                loss_aux1 = aux_criterion(x1_pred,train_classes.narrow(0, b, batch_size)[:,0])
                loss_aux1 += aux_criterion(x2_pred,train_classes.narrow(0, b, batch_size)[:,1])
                loss = loss + loss_aux1
                acc_loss += loss.item()
                model.zero_grad()
                loss.backward()
                optimizer.step()
            print(e, acc_loss)
            
    def compute_nb_errors(model, input_data, target_data, batch_size):
        nb_errors = 0
    
        for b in range(0, input_data.size(0), batch_size):
                imgs = input_data.narrow(0, b, batch_size)
                target = target_data.narrow(0, b, batch_size)
                imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
                imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
                output = model(imgs1, imgs2)
                output = output.view(batch_size, -1)
                pred = output.max(1)[1]
                nb_errors += (pred-target).abs().sum().item()

        return nb_errors
    
    def compute_nb_errors_aux(model, input_data, target_data, batch_size):
        nb_errors = 0

        for b in range(0, input_data.size(0), batch_size):
                imgs = input_data.narrow(0, b, batch_size)
                target = target_data.narrow(0, b, batch_size)
                imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
                imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
                _,_, output = model(imgs1, imgs2)
                output = output.view(batch_size, -1)
                pred = output.max(1)[1]
                nb_errors += (pred-target).abs().sum().item()

        return nb_errors
    
    