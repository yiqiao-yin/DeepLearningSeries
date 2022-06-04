# library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# define class
class DL:
  
  # define new neuralnet function
  def NN3(
    x_train,
    y_train,
    x_test,
    y_test,
    validation_split=0.2,
    epochs=10
  ):
    
    # rescale
    x_train = x_train / 255
    x_test = x_test / 255

    # use sequential api to build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(), # neural network requires the input layer to be a vector instead of 2D array
        ## Your Changes Start Here ##
        # Things to change:
        # number of hidden layers
        # number of neurons per hidden layer
        # the activation functions: Please refer this website: https://keras.io/api/layers/activations/
        # the kernel_regularizer: Please refer this website: https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer
        tf.keras.layers.Dense(1024, activation='relu', use_bias=True), # input units (usually starts with 128) and activation (it's a choice, usually relu)
        tf.keras.layers.Dense(512, activation='relu', use_bias=True), # create more hidden layer if desired
        tf.keras.layers.Dense(64, activation='relu', use_bias=True), # create more hidden layer if desired
        # ... you can have however many you want
        ## Your Changes Ends Here ##
        tf.keras.layers.Dense(10, activation='softmax') # output layer or end layer | you have to match the number of classes
    ])
    
    # compile
    model.compile(
        ## Your Changes Start Here ##
        # optimizer: https://keras.io/api/optimizers/
        # loss: https://keras.io/api/losses/
        optimizer="adam",
        loss="categorical_crossentropy",
        ## Your Changes Ends Here ##
        metrics=['accuracy']
    )
    
    # fit and train
    history = model.fit(
            x_train, y_train,
            validation_split=validation_split,
            ## Your Changes Start Here ##
            # there is a sweet spot, you don't want to overfit, but you also want to make sure your model learned something
            epochs=epochs
            ## Your Changes Ends Here ##
        )
    
    # quick evaluation on test set
    this_final_loss_, this_final_acc_ = model.evaluate(x_test, y_test)
    print('Test Result: Loss is '+str(this_final_loss_)+', and accuracy is '+str(this_final_acc_))
    
    # return
    return {
      'History': history,
      'Model': model
    }
