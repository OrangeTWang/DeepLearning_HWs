import numpy as np
import warnings



def dropout_forward(x, drop_rate, mode):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - drop_rate: Dropout parameter. We keep each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
            if the mode is test, then just return the input.

    Outputs:
    - out: Array of the same shape as x.
    """

    mask = None
    out = None

    if mode == 'train':
        
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        out = np.maximum(0, x)
        mask = (np.random.rand(*out.shape) < drop_rate) / drop_rate # first dropout mask
        out *= mask # drop!
                
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test': 
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
               
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################


    return out



def conv_forward(x, w, b, conv_param):
    """
    An implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, H, W, C)
    - w: Filter weights of shape (HH, WW, C, F)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        
    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, H', W', F) where H' and W' are given by
      H' = (H + 2 * pad - HH + stride) // stride
      W' = (W + 2 * pad - WW + stride) // stride
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    x_pad = np.pad(x, ((0,),(pad,),(pad,),(0,)), 'constant')
    print(x_pad.shape)
    
    N, H, W, C = x.shape
    HH, WW, C, F = w.shape
    
    H_out = (H + 2 * pad - HH + stride) // stride
    W_out = (W + 2 * pad - WW + stride) // stride
    
    out_shape =  (N, H_out, W_out, F)
    out = np.zeros(out_shape)
    
    
    #第一层是
    for n in range(N):
        for f in range(F):
            h_xuhao = 0            
            for h in range(H_out):
                w_xuhao = 0
                for w_index in range(W_out):
                    #此处进行操作
                    #print('out',out[n,h,w_index,f].shape)
                    #print('x',x_pad[n, h_xuhao:(h_xuhao+HH), w_xuhao:(w_xuhao+WW),: ].shape)
                    #print('w',w[:,:,:,f].shape)
                    out[n,h,w_index,f] = np.sum( [x_pad[n, h_xuhao:(h_xuhao+HH), w_xuhao:(w_xuhao+WW),: ] * w[:,:,:,f]])
                    
                    w_xuhao += stride
                h_xuhao += stride
             
        print('out',out[n,:,:,:].shape)
    out = out + b     
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


    return out



def max_pool_forward(x, pool_param):
    """
    A implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, H, W, C)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    pool_height = pool_param['pool_height']
    pool_width =  pool_param['pool_width']
    stride =  pool_param['stride']
    
    N, H, W, C = x.shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    
    out_shape =  (N, C, H_out, W_out)
    out = np.zeros(out_shape)
    
    for n in range(N):
        for c in range(C):
            h_xuhao = 0
            for h in range(H_out):
                #print(h)
                w_xuhao = 0
                for width in range(W_out):
                    
                    out[n,c,h,width] = np.max(x[n, h_xuhao:(h_xuhao+pool_height), w_xuhao:(w_xuhao + pool_width),c])
                    
                    w_xuhao += stride
                h_xuhao += stride
                
    #print('out',out.shape)
    out = np.transpose(out, [0, 2, 3, 1])
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out


