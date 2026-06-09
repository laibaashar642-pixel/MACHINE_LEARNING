#Human Brain mai jistrah nuerons hoty hai jo baar baar aik cheez ko dekh kr uskay pattern bnaty hai isi trah ai mai bhi nuetral network as a neuron kaam krty hai jo help krty hai pattern bnanay mai
import numpy as np
#Sigmoid Activation Function
def sigmoid(x):
    return 1/(1+np.exp(-x))
#derivative of sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
#Training Data
X=np.array([[0,1],[1,1],[1,0],[1,1]])#X Inputs
Y=np.array([[1],[0],[1],[0]])#XOR Output
#Initialize Weights
np.random.seed(42)
weights_input_hidden=np.random.rand(2,2)
weights_hidden_output=np.random.rand(2,1)
#Training the loop
for epoch in range(1000):
    #Forward Propagation
    hidden_input=np.dot(X,weights_input_hidden)
    hidden_output=sigmoid(hidden_input)
    final_input=np.dot(hidden_output,weights_hidden_output)
    final_output=sigmoid(final_input)
#Error 
error=Y-final_output
#backward Propagation
d_output=error*sigmoid_derivative(final_output)
d_hidden=d_output.dot(weights_hidden_output.T)*sigmoid_derivative(hidden_output)
#Update Weights
weights_hidden_output+=hidden_output.T.dot(d_output)
weights_input_hidden+=X.T.dot(d_hidden)
print("Final Output after training:",final_output)
#m.omar@aurixtech.com