'''
Created on 2018年12月5日

@author: coderwangson
'''
"#codeing=utf-8"

# 参考吴恩达写的
import numpy as np
def zero_padding(x,pad):
    return np.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values= 0)

#a_slice_prev是相当于一个左上角只是一个卷积核的一个卷积操作
def conv_single_step(a_slice_prev,W,b):
    s = np.multiply(a_slice_prev,W)+b
    return np.sum(s)
def tanh(z):
    r = 0
    try:
        r = (np.exp(z)-np.exp(-z))/(np.exp(z) + np.exp(-z))
    except Exception as e:
        print(z)
    return r
def relu(z):
    return np.maximum(z,0)
def conv_forward(A_prev,W,b,hparameters):
    """
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    """
    (m,n_h_prev,n_w_prev,n_c_prev) = A_prev.shape
    (f,f,n_c_prev,n_c) = W.shape
    s = hparameters['stride']
    p = hparameters['pad']
    n_h = 1 +int((n_h_prev+2*p -f)/s)
    n_w = 1 +int((n_w_prev+2*p-f)/s)
    Z = np.zeros((m,n_h,n_w,n_c))
    A_prev_pad = zero_padding(A_prev, p)
    # 效率不行 可以转为im2col
    #TODO
    for i in range(m):
        x = A_prev_pad[i]
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    # 把卷积后的位置映射到原先的位置，为了卷积
                    v_start = h*s
                    v_end = v_start + f
                    h_start = w*s
                    h_end = h_start + f
                    a_slice_prev = x[v_start:v_end,h_start:h_end,:]
                    Z[i,h,w,c] = np.sum(np.multiply(a_slice_prev,W[:,:,:,c])+b[:,:,:,c])
    cache = (A_prev, W, b, hparameters)
    return Z,cache

def pool_forward(A_prev,hparameters,mode = 'max'):
    (m,n_h_prev,n_w_prev,n_c_prev) = A_prev.shape
    
    f = hparameters["f"]
    s = hparameters["stride"]
    n_h = int(1+(n_h_prev -f)/s)
    n_w = int(1+(n_w_prev -f)/s)
    n_c = n_c_prev
    A = np.zeros((m,n_h,n_w,n_c))
    for i in range(m):
        for h in range(n_h):
            for w in range(n_w):
                for c in range(n_c):
                    v_start = h*s
                    v_end = v_start +f
                    h_start = w*s
                    h_end = h_start + f
                    a_prev_slice = A_prev[i,v_start:v_end,h_start:h_end,c]
                    if mode == 'max':
                        A[i,h,w,c] = np.max(a_prev_slice)
                    elif mode =='average':
                        A[i,h,w,c] = np.mean(a_prev_slice)
    cache = (A_prev,hparameters)
    return A,cache

def tanh_prime(z):
    return 1- tanh(z)**2
def conv_backward(dZ,cache):
    (A_prev, W, b, hparameters) = cache
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_c) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m,n_H,n_W,n_C) = dZ.shape
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))
    A_prev_pad = zero_padding(A_prev, pad)
    dA_prev_pad = zero_padding(dA_prev, pad)
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h*stride
                    v_end = v_start + f
                    h_start = w*stride
                    h_end = h_start+ f
                    a_slice = a_prev_pad[v_start:v_end,h_start:h_end,:]
                    da_prev_pad[v_start:v_end,h_start:h_end,:]+=W[:,:,:,c]*dZ[i,h,w,c]
                    dW[:,:,:,c]+=a_slice *dZ[i,h,w,c]
                    db[:,:,:,c]+=dZ[i,h,w,c]
        dA_prev[i,:,:,:] =da_prev_pad[pad:-pad, pad:-pad, :]
    return dA_prev,dW,db

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    
    ### START CODE HERE ### (≈1 line)
    mask = (x == np.max(x))
    ### END CODE HERE ###
    
    return mask  
def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix of dimension shape
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from shape (≈1 line)
    (n_H, n_W) = shape
    
    # Compute the value to distribute on the matrix (≈1 line)
    average = dz / (n_H * n_W)
    
    # Create a matrix where every entry is the "average" value (≈1 line)
    a = np.ones(shape) * average
    ### END CODE HERE ###
    
    return a

def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    
    ### START CODE HERE ###
    
    # Retrieve information from cache (≈1 line)
    (A_prev, hparameters) = cache
    
    # Retrieve hyperparameters from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    f = hparameters['f']
    # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    # Initialize dA_prev with zeros (≈1 line)
    dA_prev = np.zeros_like(A_prev)
    for i in range(m):                       # loop over the training examples
        # select training example from A_prev (≈1 line)
        a_prev = A_prev[i]
        
        for h in range(n_H):                   # loop on the vertical axis
            for w in range(n_W):               # loop on the horizontal axis
                for c in range(n_C):           # loop over the channels (depth)
                    
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    # Compute the backward propagation in both modes.
                    if mode == "max":
                        
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i, vert_start, horiz_start, c]
                        
                    elif mode == "average":
                        
                        # Get the value a from dA (≈1 line)
                        da = dA[i, vert_start, horiz_start, c]
                        # Define the shape of the filter as fxf (≈1 line)
                        shape = (f, f)
                        # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
    ### END CODE ###
    # Making sure your output shape is correct
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev
