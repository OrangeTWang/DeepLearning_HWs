"""
In this file, you should implement the forward calculation of the conventional RNN and the RNN with GRU cells. 
Please use the provided interface. The arguments are explained in the documentation of the two functions.

You also need to implement two functions that configurate GRUs in special ways.
"""

import numpy as np
from scipy.special import expit as sigmoid
import tensorflow as tf

def rnn(wt_h, wt_x, bias, init_state, input_data):
    """
    RNN forward calculation.

    args:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation. Rows corresponds 
              to dimensions of previous hidden states
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term
        init_state: shape [batch_size, hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    returns:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """

    outputs = None
    state = None
            
    batch_size, time_steps, input_size = input_data.shape
    hidden_size, hidden_size = wt_h.shape
    h_0 = init_state
    outputs = np.zeros([batch_size, time_steps, hidden_size])
    
    state =  np.tanh(bias + np.dot( init_state, wt_h ) + np.dot( input_data[:,0,:] , wt_x ) )
    outputs[:,0,:] = state
    for t in range(1,time_steps):
        state = np.tanh( bias + np.dot( state, wt_h ) + np.dot( input_data[:,t,:] , wt_x ) )
        outputs[:,t,:] = state
        
    
    ##################################################################################################
    # Please implement the basic RNN here.                                                           #
    ##################################################################################################
   
    return outputs, state


def gru(wtz_h, wtz_x, biasz, wtr_h, wtr_x, biasr, wth_h, wth_x, biash, init_state, input_data):
    """
    RNN forward calculation.

    args:
        wtz_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for z gate
        wtz_x: shape [input_size, hidden_size], weight matrix for input transformation for z gate
        biasz: shape [hidden_size], bias term for z gate
        wtr_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for r gate
        wtr_x: shape [input_size, hidden_size], weight matrix for input transformation for r gate
        biasr: shape [hidden_size], bias term for r gate
        wth_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for candicate
               hidden state calculation
        wth_x: shape [input_size, hidden_size], weight matrix for input transformation for candicate
               hidden state calculation
        biash: shape [hidden_size], bias term for candicate hidden state calculation
        init_state: shape [hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    returns:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """
    outputs = None
    state = None
    
    batch_size, time_steps, input_size = input_data.shape
    hidden_size, hidden_size = wtz_h.shape
        
    outputs = np.zeros([batch_size, time_steps, hidden_size])
    
    R = sigmoid(biasr + np.dot(init_state, wtr_h) + np.dot(input_data[:,0,:], wtr_x))
    Z = sigmoid(biasz + np.dot(init_state, wtz_h) + np.dot(input_data[:,0,:], wtz_x))
    
    cand_state = np.tanh(biash + np.dot(input_data[:,0,:], wth_x) + np.dot( np.multiply(R, init_state)  , wth_h ))
    state = np.multiply(Z,init_state) + np.multiply( (1 - Z) , cand_state)
    outputs[:,0,:] = state
    
    for t in range(1, time_steps):
        R = sigmoid(biasr + np.dot(state, wtr_h) + np.dot(input_data[:,t,:], wtr_x))
        Z = sigmoid(biasz + np.dot(state, wtz_h) + np.dot(input_data[:,t,:], wtz_x))
        cand_state = np.tanh(biash + np.dot(input_data[:,t,:], wth_x) + np.dot( np.multiply(R, state)  , wth_h ))
        state = np.multiply(Z,state) + np.multiply( (1 - Z) , cand_state)
        outputs[:,t,:] = state    
    
    
    

    ##################################################################################################
    # Please implement the GRU here.                                                                 #
    ##################################################################################################
        
    
    return outputs, state



def init_gru_with_rnn(wt_h, wt_x, bias):
    """
    This function compute parameters of a GRU such that it performs like a conventional RNN. The input are parameters 
    of an RNN, and the parameters of the GRU are returned. 

    args:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation. Rows corresponds 
              to dimensions of previous hidden states
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term

    returns:


        wtz_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for z gate
        wtz_x: shape [input_size, hidden_size], weight matrix for input transformation for z gate
        biasz: shape [hidden_size], bias term for z gate
        wtr_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for r gate
        wtr_x: shape [input_size, hidden_size], weight matrix for input transformation for r gate
        biasr: shape [hidden_size], bias term for r gate
        wth_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for candicate
               hidden state calculation
        wth_x: shape [input_size, hidden_size], weight matrix for input transformation for candicate
               hidden state calculation
        biash: shape [hidden_size], bias term for candicate hidden state calculation

    """
    input_size, hidden_size =  wt_x.shape
    
    biasz = np.ones([hidden_size]) *(-100000)
    wtz_h = np.zeros([hidden_size, hidden_size])
    wtz_x = np.zeros([input_size, hidden_size])
    
    biasr = np.ones([hidden_size]) * 100000
    wtr_h = np.zeros([hidden_size, hidden_size])
    wtr_x = np.zeros([input_size, hidden_size])
    
    wth_h = wt_h
    wth_x = wt_x
    biash = bias
    
    
    ####################################################################################################
    # Please set a set of parameters for a GRU such that it recovers an RNN with parameters passing in #
    ####################################################################################################
    
    return wtz_h, wtz_x, biasz, wtr_h, wtr_x, biasr, wth_h, wth_x, biash


def init_gru_with_long_term_memory(wt_h, wt_x, bias):
    """
    This function compute parameters of a GRU such that it maintains the initial state in the memory. The input are parameters 
    of an RNN, and the parameters of the GRU are returned. These parameters can provide shapes of weight matrices but they 
    should affect the behavior of the GRU RNN. 

    args:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation. Rows corresponds 
              to dimensions of previous hidden states
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term

    returns:


        wtz_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for z gate
        wtz_x: shape [input_size, hidden_size], weight matrix for input transformation for z gate
        biasz: shape [hidden_size], bias term for z gate
        wtr_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for r gate
        wtr_x: shape [input_size, hidden_size], weight matrix for input transformation for r gate
        biasr: shape [hidden_size], bias term for r gate
        wth_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation for candicate
               hidden state calculation
        wth_x: shape [input_size, hidden_size], weight matrix for input transformation for candicate
               hidden state calculation
        biash: shape [hidden_size], bias term for candicate hidden state calculation

    """


    ####################################################################################################
    # Please set a set of parameters for a GRU such that it keeps the initial state in its memory      #
    ####################################################################################################
    input_size, hidden_size =  wt_x.shape
    
    biasz = np.ones([hidden_size]) *(100000)
    wtz_h = np.zeros([hidden_size, hidden_size])
    wtz_x = np.zeros([input_size, hidden_size])
    
    biasr = np.ones([hidden_size]) * (-100000)
    wtr_h = np.zeros([hidden_size, hidden_size])
    wtr_x = np.zeros([input_size, hidden_size])
    
    wth_h = wt_h
    wth_x = wt_x
    biash = bias
    
    
     
    return wtz_h, wtz_x, biasz, wtr_h, wtr_x, biasr, wth_h, wth_x, biash

