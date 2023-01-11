import numpy as np
X=np.array(([2,9],[1,5],[3,6]))
y=np.array(([92],[86],[89]))
y=y/100
#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
#Derivate of Sigmoid function
def derivatives_sigmoid(x):
    return x*(1-x)
#Variable Initialization
epoch=10000
lr=0.1
inputlayer_neurons=2
hiddenlayer_neurons=3
outputlayer_neurons=1
#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bias_hidden=np.random.uniform(size=(1,hiddenlayer_neurons))
weight_hidden=np.random.uniform(size=(hiddenlayer_neurons,outputlayer_neurons))
bias_output=np.random.uniform(size=(1,outputlayer_neurons))
for i in range(epoch):
    #Forward Propagation
    hinp1=np.dot(X,wh)
    hinp=hinp1+bias_hidden
    hlayer_activation=sigmoid(hinp)
    
    outinp1=np.dot(hlayer_activation,weight_hidden)
    outinp=outinp1+bias_output
    output=sigmoid(outinp)
    
    #Backpropagation
    EO=y-output
    outgrad=derivatives_sigmoid(output)
    d_output=EO*outgrad
    
    EH=d_output.dot(weight_hidden.T)
    hiddengrad=derivatives_sigmoid(hlayer_activation)
    d_hiddenlayer=EH*hiddengrad
    
    #update the weights
    weight_hidden+=hlayer_activation.T.dot(d_output) * lr
    bias_hidden+=np.sum(d_hiddenlayer,axis=0,keepdims=True) * lr
    
    wh+=X.T.dot(d_hiddenlayer) * lr
    bias_output=np.sum(d_output,axis=0,keepdims=True) * lr
    
print("Input: \n",str(X))
print("Actual Output: \n",str(y))
print("Predicted Output: \n",output)