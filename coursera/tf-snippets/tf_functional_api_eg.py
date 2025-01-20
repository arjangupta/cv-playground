# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tfl.Conv2D(filters=8, kernel_size=(4,4), strides=1, padding="same")(input_img)
    # Z1 = None
    ## RELU
    # A1 = None
    A1 = tfl.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    # P1 = None
    P1 = tfl.MaxPool2D(pool_size=(8,8), strides=8, padding='same')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    # Z2 = None
    Z2 = tfl.Conv2D(filters=16, kernel_size=(2,2), strides=1, padding='same')(P1)
    ## RELU
    A2 = tfl.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    # P2 = None
    P2 = tfl.MaxPool2D(pool_size=(4,4), strides=4, padding='same')(A2)
    ## FLATTEN
    # F = None
    F = tfl.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    # outputs = None
    outputs = tfl.Dense(units=6, activation='softmax')(F)
    # YOUR CODE STARTS HERE
    
    
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model