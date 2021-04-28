import dlc_practical_prologue
import torch
from torch import nn
from torch import optim
import torchvision
from torch.nn import functional as F
from tqdm import trange
import matplotlib.pyplot as plt
            
def compute_nb_errors(model, input_data, target_data, batch_size):
    nb_errors = 0
    model.eval()
    for b in range(0, input_data.size(0), batch_size):
        imgs = input_data.narrow(0, b, batch_size)
        target = target_data.narrow(0, b, batch_size)
        imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
        imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
        try:
            output = model(imgs1, imgs2).view(batch_size, -1)
        except:
            _,_,output = model(imgs1, imgs2)
            output = output.view(batch_size, -1)
        pred = output.max(1)[1]
        nb_errors += (pred-target).abs().sum().item()

    return nb_errors
    
def train_model(model, train_input, train_target, batch_size, nb_epochs, train_classes = None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    aux_criterion = nn.CrossEntropyLoss()

    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, train_input.size(0), batch_size):
            imgs = train_input.narrow(0, b, batch_size)
            imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
            imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
            if train_classes==None:
                _, _, output = model(imgs1, imgs2)
                output = output.view(batch_size, -1)
                loss = criterion(output, train_target.narrow(0, b, batch_size))
            else:
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
        
def train_model_track_errors(model, train_input, train_target, batch_size, nb_epochs, test_input, test_target, train_classes = None, test_classes = None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    aux_criterion = nn.CrossEntropyLoss()
    
    
    train_loss = []
    test_loss = [] 
    train_acc = []
    test_acc = []
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, train_input.size(0), batch_size):
            imgs = train_input.narrow(0, b, batch_size)
            imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
            imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
            if train_classes==None:
                _, _, output = model(imgs1, imgs2)
                output = output.view(batch_size, -1)
                loss = criterion(output, train_target.narrow(0, b, batch_size))
            else:
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
            
        train_loss.append(acc_loss)
        train_acc.append(1-compute_nb_errors(model, train_input, train_target, batch_size)/train_input.size(0))
        test_acc.append(1-compute_nb_errors(model, test_input, test_target, batch_size)/test_input.size(0))
        test_loss_epoch = 0
        for b in range(0, test_input.size(0), batch_size):
            imgs = test_input.narrow(0, b, batch_size)
            target = test_target.narrow(0, b, batch_size)
            imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
            imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
            _, _, output = model(imgs1, imgs2)
            output = output.view(batch_size, -1)
            test_loss_epoch += criterion(output, test_target.narrow(0, b, batch_size))
        test_loss.append(test_loss_epoch.item())
    
    return train_loss, test_loss, train_acc, test_acc

def plot(train_loss, test_loss, train_acc, test_acc):
    # Parameters are the exact output from train_model_track_errors
    plt.figure(figsize = (9, 6))
    
    ax_loss = plt.gca()
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    p1 = ax_loss.plot(train_loss, 'r', label = 'Train Loss')
    p2 = ax_loss.plot(test_loss,  'b', label = 'Test Loss')
    
    ax_acc = ax_loss.twinx()  # instantiate a second axes that shares the same x-axis

    ax_acc.set_ylabel('Accuracy')  # we already handled the x-label with ax1
    p3 = ax_acc.plot(test_acc, 'g', label = 'Test Accuracy')
    p4 = ax_acc.plot(train_acc, 'k', label = 'Train Accuracy')
    legends = [l.get_label() for l in p1+p2+p3+p4]
    plt.grid()
    ax_acc.legend(p1+p2+p3+p4, legends)
    plt.show()