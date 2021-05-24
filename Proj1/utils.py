# Import PyTorch and libraries required for the project
import dlc_practical_prologue
import torch
from torch import nn
from torch import optim
import torchvision
from torch.nn import functional as F
from tqdm import trange
import matplotlib.pyplot as plt
            
# Function to count the number of errors in a prediction
def compute_nb_errors(model, input_data, target_data, batch_size):
    '''
    Function to Count the number of errors in a testing set.
    '''
    nb_errors = 0
    # Set the model in evaluation mode
    model.eval()
    # Iterate every batch of the testing data
    for b in range(0, input_data.size(0), batch_size):
        # Retrieve batch of images and targets
        imgs = input_data.narrow(0, b, batch_size)
        target = target_data.narrow(0, b, batch_size)
        # Prepare for model
        imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
        imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
        # Get output of the model and predict classes based on it
        _,_,output = model(imgs1, imgs2)
        output = output.view(batch_size, -1)
        pred = output.max(1)[1]
        # Count number of errors by counting the differences between the targets and the predictions
        nb_errors += (pred-target).abs().sum().item()
    # Set the model back into trainin mode
    model.train()
    return nb_errors
    
def train_model(model, train_input, train_target, batch_size, nb_epochs, train_classes = None):
    '''
    Function to train the model.
    '''
    # Set the model in training mode
    model.train()
    # Define loss functions, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-2)
    aux_criterion = nn.CrossEntropyLoss()

    # Iterate over the number of epochs
    for e in range(nb_epochs):
        # Set loss to zero
        acc_loss = 0
        # Iterate minibatches
        for b in range(0, train_input.size(0), batch_size):
            # Get minibatch and prepare for input
            imgs = train_input.narrow(0, b, batch_size)
            imgs1 = imgs[:,0].view(batch_size, 1, 14, 14)
            imgs2 = imgs[:,1].view(batch_size, 1, 14, 14)
            # If conditional to check whether we are training with or without auxiliary loss
            if train_classes==None:
                # Perform forward pass and calculate loss
                _, _, output = model(imgs1, imgs2)
                output = output.view(batch_size, -1)
                loss = criterion(output, train_target.narrow(0, b, batch_size))
            else:
                # Perform forward pass and calculate both auxiliary and main loss
                x1_pred, x2_pred , output = model(imgs1, imgs2)
                x1_pred = x1_pred.view(batch_size, -1)
                x2_pred = x2_pred.view(batch_size, -1)
                output = output.view(batch_size, -1)
                loss = criterion(output, train_target.narrow(0, b, batch_size))
                loss_aux1 = aux_criterion(x1_pred,train_classes.narrow(0, b, batch_size)[:,0])
                loss_aux1 += aux_criterion(x2_pred,train_classes.narrow(0, b, batch_size)[:,1])
                # 'Final' loss is composed from the two of them 
                loss = loss + loss_aux1
            
            acc_loss += loss.item()
#             model.zero_grad()
            loss.backward()
            optimizer.step()
#             loss.backward()
#             optimizer.step()
            optimizer.zero_grad()
        
def train_model_track_errors(model, train_input, train_target, batch_size, nb_epochs, test_input, test_target, train_classes = None, test_classes = None):
    '''
    Function to train the model and test at every epoch. Used to keep track of the evolution of the loss and accuracy.
    '''
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

def plot(train_loss, test_loss, train_acc, test_acc, title=''):
    '''
    Function to plot the evolution of the loss and the accuracy of the train and testing sets. 
    The input parameters should be the output parameters of `train_model_track_errors`, plus the 
    title of the plot as optional parameter
    '''
    # Initialize figure
    plt.figure(figsize = (9, 6))
    
    # Plot loss
    ax_loss = plt.gca()
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title(title)
    p1 = ax_loss.plot(train_loss, 'r', label = 'Train Loss')
    p2 = ax_loss.plot(test_loss,  'b', label = 'Test Loss')
    
    # Get twin axis and plot accuracy
    ax_acc = ax_loss.twinx()  # instantiate a second axes that shares the same x-axis

    ax_acc.set_ylabel('Accuracy')  # we already handled the x-label with ax1
    p3 = ax_acc.plot(test_acc, 'g', label = 'Test Accuracy')
    p4 = ax_acc.plot(train_acc, 'k', label = 'Train Accuracy')
    legends = [l.get_label() for l in p1+p2+p3+p4]
    plt.grid()
    ax_acc.legend(p1+p2+p3+p4, legends)
    plt.show()
    

def count_parameters(model):
    '''
    Auxiliary function to count the number of parameters in a model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)