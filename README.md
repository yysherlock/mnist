#MNIST Project 
## NN package
* nn.py: Mini batch gradient descent with Least Square Error as cost function, and no vectorization.
This version achieves:
[]  **92.45%**(learning rate = 3.0, w/o weight decay)
[]  (learning rate = 1.0, with weight decay) 
on test set .
Several detailed things (i.e. hyper-parameters) have to be taken into consideriation, 1) how to properly deal with the parameters/weights initialization, 2) learning rate, 3) weight\_decay, 4) when to converage (tolerance), 5) batch\_size.
