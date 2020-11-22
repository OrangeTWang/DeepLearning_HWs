import tensorflow as tf
import numpy

"""
This is a short tutorial of tensorflow. After this tutorial, you should know the following concepts:
1. constant,
2. operations
3. variables 
4. gradient calculation 
5. optimizer 
"""

def regression_func(x, w, b):
    """
    The function of a linear regression model
    args: 
        x: tf.Tensor with shape (n, d) 
        w: tf.Variable with shape (d,) 
        b: tf.Variable with shape ()

    return: 
        y_hat: tf.Tensor with shape [n,]. y_hat = x * w + b (matrix multiplication)
    """
    #get the input x, w, b are tf.constant()
    
    #print(w_trans.get_shape())
    y_hat = tf.einsum('ij,j->i',x,w) + b
    #x1 = tf.matmul(x,w)
    #y_hat = tf.add(x1, b)
    #print(y_hat)
    #y_hat = tf.transpose(y_hat)
    #y_hat = tf.constant(y_hat, dtype=tf.float32,shape=[2,])

    # TODO: implement this function
    # consider these functions: `tf.matmul`, `tf.einsum`, `tf.squeeze` 

    return y_hat



def loss_func(y, y_hat):
    """
    The loss function for linear regression

    args:
        y: tf.Tensor with shape (n,) 
        y_hat: tf.Tensor with shape (n,) 

    return:
        loss: tf.Tensor with shape (). loss = (y -  y_hat)^\top (y -  y_hat) 

    """
    #y_hat = tf.constant(y_hat, dtype=tf.float32)
    #y1 = tf.constant(y, dtype=tf.float32)
    diff = tf.subtract( y , y_hat)
    loss = tf.reduce_sum(tf.square(diff))

    # TODO: implement the function. 
    # Consider these functions: `tf.square`, `tf.reduce_sum`

    return loss



def train_lr(x, y, lamb):
    
    w = tf.Variable(tf.random.uniform((1,)), dtype=tf.float32)
    b = tf.Variable(1.0, dtype=tf.float32)
  
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    for i in range(1000):
     
        with tf.GradientTape() as gt:
            gt.watch([w, b])
            y_hat = regression_func(x, w, b)
            mse = loss_func(y, y_hat)/2
            reg = (lamb/2) * tf.reduce_sum(tf.square(w))
            loss = (mse + reg)/100
        
        gradient = gt.gradient(loss, [w, b])   
        optimizer.apply_gradients(zip(gradient, [w, b]))
        if i % 100 == 1:
            print('loss becomes', loss.numpy(), 'after',i,'iteration' )
    
    """
    Train a linear regression model.

    args:
        x: tf.Tensor with shape (n, d)
        y: tf.Tensor with shape (n, )
        lamb: tf.Tensor with shape ()
    """
    
    # TODO: implement the function.
    # initialize parameters w and b


    # set an optimizer
    # please check the documentation of tf.keras.optimizers.SGD

    # loop to optimize w and b 


    return w, b

