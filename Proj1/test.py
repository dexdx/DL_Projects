# from models import FCC, Siamese, Siamese_no_sharing
from models_v2 import FCC, Siamese
import dlc_practical_prologue
import torch
from torch import nn
from torch import optim
import torchvision
from torch.nn import functional as F
from tqdm import trange
import matplotlib.pyplot as plt
import models
# from utils import Proj1
import utils
import sys

plot = False
if len(sys.argv)<2:
    print("Note:\nUsing:\t`python test.py plot`\t, the program will show a plot of loss and accuracy for each model.")
else:
    plot = True

# Load data
train_input, train_target, train_classes, test_input, test_target, test_classes = dlc_practical_prologue.generate_pair_sets(1000)

train_data = (train_input, train_target, train_classes)
test_data = (test_input, test_target, test_classes)

# Instantiate models and strings to name them
models = (FCC(), FCC(share=True), FCC(aux=True), FCC(share=True, aux=True), Siamese(), Siamese(share=True), Siamese(aux=True), Siamese(share=True, aux=True))
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
            # Call train_model with the correct parameters
            if 'Aux' in name:
                utils.train_model(model, train_input, train_target, batch_size, epochs, train_classes)
            else:
                utils.train_model(model, train_input, train_target, batch_size, epochs, train_classes=None)
            # Get error on train and test set
            train_error_rates[r] = utils.compute_nb_errors(model, train_input, train_target, batch_size)/train_input.size(0)
            test_error_rates[r] = utils.compute_nb_errors(model, test_input, test_target, batch_size)/test_input.size(0)
            
        #print(f'For the model {name}, the train average error rate is {train_error_rates.mean()} and the test average error rate {test_error_rates.mean()}\n')
        print('For the model {}, the train average error rate is {:.3}% and the test average error rate is {:.3}%.\n'
              .format(name, 100*train_error_rates.mean(), 100*test_error_rates.mean()))
        
# Call function
evaluate_all(models, model_names, train_data, test_data, batch_size=100, rounds=10, epochs=25)

# If plotting required:
if plot:
    models = (FCC(), FCC(share=True), FCC(aux=True), FCC(share=True, aux=True), Siamese(), Siamese(share=True), Siamese(aux=True), Siamese(share=True, aux=True))

    model_names = ('FCC' , 'FCC with shared weights', 'FCC with auxiliary loss', 'FCC with auxiliary loss and shared weights',
               'Siamese without wwight sharing', 'Siamese with shared weights', 'Siamese with auxiliary loss', 'Siamese with auxiliary loss and shared weights')


    for model in models:
        losses = utils.train_model_track_errors(model, train_input, train_target, 100, 25, test_input, test_target, train_classes)
        utils.plot(*losses)