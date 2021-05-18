from torch import empty
import math
from NN_Modules import LossMSE, ReLU, Tanh, Sigmoid, FCC, Sequential, modelTester, modelTrainer

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


    model = Sequential(["FCC","Tanh", "FCC","ReLU","FCC","Tanh", "FCC"],[[2,25],[], [25,25],[], [25,25],[], [25,2]],"MSE",momentum = 0.1)
    trained_model ,loss, train_acc  = modelTrainer(model = model , num_epoch = 300, train_input = train_input,train_target = train_target, train_batch = 20)
    test_acc = modelTester(trained_model, test_input,test_target,10)
    print("Final Test accuracy: ", test_acc)
    print("Final Train accuracy: ", train_acc[0][-1])

if __name__ == "__main__":
    main()