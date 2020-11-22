"""
This problem is modified from a problem in Stanford CS 231n assignment 1. 
In this problem, we implement the neural network with tensorflow instead of numpy
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

##################################################################################################################
# Function suggestions
# np.random.random_sample(), tf.Variable(), tf.matmul(), tf.nn.relu(), tf.tanh(), tf.sigmoid(), 
# tf.nn.softmax(), tf.reduce_sum(), tf.square(), tf.keras.optimizers.SGD(), tf.keras.optimizers.Adam()
# tf.keras.losses.MSE(), tf.keras.losses.CategoricalCrossentropy(), tf.keras.Model.compile(), tf.keras.Model.fit()
##################################################################################################################


class DenseLayer(tf.keras.layers.Layer):
    """
    Implement a dense layer 
    """

    def __init__(self, input_dim, output_dim, activation, reg_weight, param_init=None):

        """
        Initialize weights of the DenseLayer. In Tensorflow's implementation, the weight 
        matrix and bias term are initialized in the `build` function. Here let's use the simple form. 

        https://www.tensorflow.org/guide/keras/custom_layers_and_models

        args:
            input_dim: integer, the dimension of the input layer
            output_dim: integer, the dimension of the output layer
            activation: string, can be 'linear', 'relu', 'tanh', 'sigmoid', or 'softmax'. 
                        It specifies the activation function of the layer
            reg_weight: the regularization weight/strength, the lambda value in a regularization 
                        term 0.5 * \lambda * ||W||_2^2
                        
            param_init: `dict('W'=W, 'b'=b)`. Here W and b should be `np.array((input_dim, output_dim))` 
                        and `np.array((1, output_dim))`. The weight matrix and the bias vector are 
                        initialized by `W` and `b`. 
                        NOTE: `param_init` is used to check the correctness of your function. For you 
                        own usage, `param_init` can be `None`, and the parameters are initialized 
                        within this function. But when `param_init` is not None, you should 
                        `param_init` to initialize your function. 

        """


        super(DenseLayer, self).__init__()

        # set initial values for weights and bias terms. 
        # Note: bad initializations may lead to bad performance later
        self.activation = activation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reg_weight = reg_weight
        self.param_init = param_init
        
        

        if param_init == 'autograder':
            param_init = dict(W=None, b=None)
            np.random.seed(137)
            param_init['W'] = np.random.random_sample((input_dim, output_dim)) 
            param_init['b'] = np.random.random_sample((output_dim, )) 
        else:
            param_init = dict(W=None, b=None)
            np.random.seed(100)
            param_init['W'] = np.random.random_sample((input_dim, output_dim)) 
            param_init['b'] = np.random.random_sample((output_dim, )) 
            # please do your own initialization here    
            # You can pass in your own initialization through `param_init` or do random initialization here.  
        self.w = tf.Variable(param_init['W'],dtype = 'float32', trainable = True) 
        self.b = tf.Variable(param_init['b'],dtype = 'float32', trainable = True) 
        
        # Initialize necessary tf variables with `param_init`
       

    def call(self, inputs, training=None, mask=None):
        """
        This function implement the `call` function of the class's parent `tf.keras.layers.Layer`. Please 
        consult the documentation of this function from `tf.keras.layers.Layer`.
        """
               
        # Implement the linear transformation
        linear_outputs = tf.matmul(inputs, self.w) + self.b

        # Implement the activation function
        if self.activation is 'linear':
            outputs = linear_outputs
        elif self.activation is 'relu':
            outputs = tf.nn.relu(linear_outputs)
        elif self.activation is 'tanh':
            outputs = tf.tanh(linear_outputs)
        elif self.activation is 'sigmoid':
            outputs = tf.sigmoid(linear_outputs)
        elif self.activation is 'softmax':
            outputs = tf.nn.softmax(linear_outputs)
        

        # check self.add_loss() to add the regularization term to the training objective
        rate = 0.5 * self.reg_weight 
        self.add_loss(rate * tf.reduce_sum(tf.square(self.w)))


        return outputs    


class Feedforward(tf.keras.Model):

    """
    A feedforward neural network. 
    """

    def __init__(self, input_size, depth, hidden_sizes, output_size, reg_weight, task_type, *args, **kwargs):

        """
        Initialize the model. This way of specifying the model architecture is clumsy, but let's use this straightforward
        programming interface so it is easier to see the structure of the program. Later when you program with keras 
        layers, please think about how keras layers are implemented to take care of all components.  

        NOTE: args and kwargs can be used to provide extra parameters, such as parameters for data normalization. But the 
        implementation needs to be able to run without *args or **kwargs because the autograder cannot provide these 
        arguments. 

        args:
          input_size: integer, the dimension of the input.
          depth:  integer, the depth of the neural network, or the number of connection layers. 
          hidden_sizes: list of integers. The length of the list should be one less than the depth of the neural network.
                        The first number is the number of output units of first connection layer, and so on so forth.
          output_size: integer, the number of classes. In our regression problem, please use 1. 
          reg_weight: float, The weight/strength for the regularization term.
          task_type: string, 'regression' or 'classification'. The task type. 

        """

        super(Feedforward, self).__init__()

        self.input_size = input_size
        self.depth = depth
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.reg_weight = reg_weight
        self.task_type = task_type
        
                
        # Add a contition to make the program robust 
        if not (depth - len(hidden_sizes)) == 1:
            raise(Exception('The depth of the network is ', depth, ', but `hidden_sizes` has ', len(hidden_sizes),
                            ' numbers in it.'))
       
        
        # install a data normalization
        
        self.likelayer = []
        #self.likelayer.append(tf.keras.layers.BatchNormalization(axis = -1, momentum=1, epsilon=0.00001,trainable=True))
        # install all connection layers except the last one
        for d in range(len(hidden_sizes)):
            if d is 0:
                layer = DenseLayer(input_size, hidden_sizes[0], 'tanh', reg_weight, param_init=None)
                self.likelayer.append(layer)
            else:
                layer = DenseLayer(hidden_sizes[d-1], hidden_sizes[d], 'tanh', reg_weight, param_init=None)
                self.likelayer.append(layer)

        # decide the last layer according to the task type
        if self.task_type is 'regression':
            last_layer = DenseLayer(hidden_sizes[-1], output_size, 'tanh', reg_weight, param_init=None)
        elif self.task_type is 'classification':
            last_layer = DenseLayer(hidden_sizes[-1], output_size, 'softmax', reg_weight, param_init=None)
        
        self.likelayer.append(last_layer)
        #tanh, tanh, tanh for regression
        #relu, relu, relu
        
        



    def call(self, inputs, training=None, mask=None):
        """
        Implement the `call` function of `tf.keras.Model`. Please consult the documentation of tf.keras.Model to understand 
        this function. 
        """
        # print a message from the network function. Please don't delete this line and count how many times this message is printed
        # It seems that every time the network is evaluated, the message should be printed. If so, then the message should be printed        
        # for #batches times. However, you only see it printed once or twice, why? 
        print('I am in the network function!')

        # Now start implement this function and apply the neural network on the input 
        
        outputs = inputs 
        
        for h in range(len(self.likelayer)):
            den_layer = self.likelayer[h]
            outputs = den_layer(outputs)
                
        return outputs


def train(x_train, y_train, x_val, y_val, depth, hidden_sizes, reg_weight, num_train_epochs, task_type, *args, **kwargs):

    """
    Train the neural network defined above.

    NOTE: args and kwargs can be used to provide extra parameters, such as parameters for data normalization.

    args:
      x_train: `np.array((N, D))`, training data of N instances and D features.
      y_train: `np.array((N, C))`, training labels of N instances and C fitting targets 
      x_val: `np.array((N1, D))`, validation data of N1 instances and D features.
      y_val: `np.array((N1, C))`, validation labels of N1 instances and C fitting targets 
      depth: integer, the depth of the neural network 
      hidden_sizes: list of integers. The length of the list should be one less than the depth of the neural network.
                    The first number is the number of output units of first connection layer, and so on so forth.

      reg_weight: float, the regularization strength.
      num_train_epochs: the number of training epochs.
      task_type: string, 'regression' or 'classification', the type of the learning task.
      extra_inputs: a place h
    returns:
      model: a trained model
      history: training history from tf.keras.Model.fit()
    """

    # prepare the data to make sure the data type is correct.    
    
    
    N,D = np.shape(x_train)
    N,C = np.shape(y_train)
    N1, D = np.shape(x_val)
    N1, C = np.shape(y_val)
    
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    
    
    # initialize a model with the Feedforward class 
    
    model = Feedforward(D, depth, hidden_sizes, C, reg_weight, task_type, args)
      

    # initialize an opimizer
    # Note: we have not talked about advanced optimization algorithms such as adagrad, adam, SGD with momentum, etc., 
    # but you may want try these opimization algorithms because they are generally more effective than the basic SGD.  
    # You can find more details about these algorithms in Ch8 of the DL book or Ch11 of the D2L book 
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2) 
    
    
    # decide the loss for the learning problem
    if task_type is 'regression':
        loss_n = tf.keras.losses.MeanSquaredError()
    elif task_type is 'classification':
        loss_n = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
    
    # compile and train the model. Consider model.fit()
    # Note: model.fit() returns the training history. Please keep it and return it later
    
    model.compile(optimizer=optimizer, loss=loss_n , metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs = num_train_epochs, validation_data=(x_val, y_val))
  
    
    
    # return the model and the training history. We will print the training history
    return model, history

