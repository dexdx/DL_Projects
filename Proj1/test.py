# Import PyTorch required libraries
import torch
from torch import nn
from torch import optim
import torchvision
from torch.nn import functional as F

# Python standard libraries
from tqdm import trange
import matplotlib.pyplot as plt
import sys

# Modules for the project
from models_v2 import FCC, Siamese
import dlc_practical_prologue
import utils

# Change this parameter to plot loss and accuracy during one round of training
plot = False

# Load and group data
train_input, train_target, train_classes, test_input, test_target, test_classes = dlc_practical_prologue.generate_pair_sets(1000)
train_data = (train_input, train_target, train_classes)
test_data = (test_input, test_target, test_classes)

# Instantiate models and strings to name them
models = (FCC(), FCC(share=True), FCC(aux=True), FCC(share=True, aux=True), 
          Siamese(), Siamese(share=True), Siamese(aux=True), Siamese(share=True, aux=True))
model_names = ('FCC' , 'FCC with shared weights', 'FCC with auxiliary loss', 'FCC with auxiliary loss and shared weights', 'Siamese without wwight sharing',
                'Siamese with shared weights', 'Siamese with auxiliary loss', 'Siamese with auxiliary loss and shared weights')

# Declare function that will evaluate all the models
def evaluate_all(models, model_names, train_data, test_data, batch_size=100, rounds=10, epochs=25):
    # Unpack data
    train_input, train_target, train_classes = train_data
    test_input, test_target, test_classes = test_data
    # Initialize err rates arrays
    train_error_rates = torch.empty(rounds)
    test_error_rates = torch.empty(rounds)
    
    # Iterate models
    for model, name in zip(models, model_names):
        print(f'Training model {name}:')
        # Iterate over r to get average
        for r in trange(rounds):
            # Randomly reset all the model parameters 
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
                    
            # Call train_model with the correct parameters (no train_classes )
            if 'Aux' in name:
                utils.train_model(model, train_input, train_target, batch_size, epochs, train_classes)
            else:
                utils.train_model(model, train_input, train_target, batch_size, epochs, train_classes=None)
            # Get error on train and test set
            train_error_rates[r] = utils.compute_nb_errors(model, train_input, train_target, batch_size)/train_input.size(0)
            test_error_rates[r] = utils.compute_nb_errors(model, test_input, test_target, batch_size)/test_input.size(0)
            
        print(f'For the model {name}, the train average error rate is {100*train_error_rates.mean():.2f}%  (with standard deviation of \
{100*train_error_rates.std():.2f}) and the test average error rate {100*test_error_rates.mean():.2f}% (with standard deviation of {100*test_error_rates.std():.2f})\n')
#         print('For the model {}, the train average error rate is {:.3}% and the test average error rate is {:.3}%.\n'
#               .format(name, 100*train_error_rates.mean(), 100*test_error_rates.mean()))
        
# Call function
evaluate_all(models, model_names, train_data, test_data, batch_size=100, rounds=10, epochs=25)

# If plotting required:
if plot:
    
    # Reinstantiate models
    models = (FCC(), FCC(share=True), FCC(aux=True), FCC(share=True, aux=True), 
              Siamese(), Siamese(share=True), Siamese(aux=True), Siamese(share=True, aux=True))
    model_names = ('FCC' , 'FCC with shared weights', 'FCC with auxiliary loss', 'FCC with auxiliary loss and shared weights',
               'Siamese without wwight sharing', 'Siamese with shared weights', 'Siamese with auxiliary loss', 'Siamese with auxiliary loss and shared weights')

    # Iterate models train and plot
    for i in range(len(models)):
        losses = list(utils.train_model_track_errors(models[i], train_input, train_target, 100, 25, test_input, test_target, train_classes)) + list(model_names[i])
        utils.plot(*losses)