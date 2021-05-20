from torch import empty
import math
from NN_Modules import LossMSE, ReLU, Tanh, Sigmoid, FCC, Sequential, modelTester, modelTrainer

#Generates the ground truth for the toy problem given to us.
def in_circle(dataset):
    dataset = dataset.add(-0.5) #Center the points
    target = empty(dataset.shape[0], 2)
    target[:,0] = dataset.pow(2).sum(1) <= 1 / (2 * math.pi)
    target[:,1] =  (target[:,0] != 1)
    return target

def main():
    train_size = 1000
    test_size = 1000
    train_input = empty(train_size, 2).uniform_(0,1)
    train_target = in_circle(train_input)
    test_input = empty(test_size, 2).uniform_(0,1)
    test_target = in_circle(test_input)

    #Create them model using the Sequential class. This model has 2 input neurons, 3 hidden layers of 25 neurons and 2 output neurons.
    # First and third layers use Tanh as their activation function. Second layer uses ReLU.
    # Our loss function is MSE and we 7e-2 as our learning rate nad 0.15 as our momentum rate.
    model = Sequential(["FCC","Tanh", "FCC","ReLU","FCC","Tanh", "FCC"],[[2,25],[], [25,25],[], [25,25],[], [25,2]],"MSE",lr = 7e-2, momentum = 0.15)
    #Train the model for 1000 epochs. We also use a batch size of 20.
    trained_model ,loss, train_acc  = modelTrainer(model  , num_epoch = 1000, train_input = train_input,train_target = train_target, train_batch = 20)
    #Measure models accuracy using the  test set.
    test_acc = modelTester(trained_model, test_input,test_target,10)
    print("Final Test accuracy: ", test_acc)
    print("Final Train accuracy: ", train_acc[0][-1])
    


if __name__ == "__main__":
    main()
    
    
