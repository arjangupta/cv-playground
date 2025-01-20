def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tfl.ZeroPadding2D(padding=3, input_shape=(64,64,3)),
            
            ## Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(filters=32, kernel_size=(7,7), strides=1),
            
            ## BatchNormalization for axis 3
            tfl.BatchNormalization(axis=3),
            
            ## ReLU
            tfl.ReLU(),
            
            ## Max Pooling 2D with default parameters
            tfl.MaxPool2D(),
            
            ## Flatten layer
            tfl.Flatten(),
            
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tfl.Dense(units=1, activation="sigmoid")
            # YOUR CODE STARTS HERE
            
            
            # YOUR CODE ENDS HERE
        ])
    
    return model