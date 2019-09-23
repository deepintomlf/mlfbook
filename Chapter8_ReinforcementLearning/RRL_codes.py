# import keras and tensorflow modules
import keras
import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer
from keras.layers import Input, Lambda, Add, Flatten,Multiply, Concatenate, Subtract, SimpleRNN, Dot
from keras.models import Model, Sequential
from keras import optimizers

# define lambda function to split
# split_func1(x)
# output: the time series of p-lagged values
def split_func1(x):    
    p = x._keras_shape
    split1, split2 = tf.split(x, [p[-1]-1, 1], -1)
    return split1

# define lambda function to split
# output: the time series of the next return
def split_func2(x):  
    p = x._keras_shape
    split1, split2 = tf.split(x,  [p[-1]-1, 1], -1)
    return split2
    
# define the customized loss function - negative Sharpe ratio of the output layer 
def trading_return(x, delta):
    F_t_Layer, r_tplus1_Layer = tf.split(x, [1, 1], -1)
    p = x._keras_shape
    F_tminus1_layer1, f1 = tf.split(F_t_Layer, [p[-2]-1, 1], -2)
    f1, F_t_layer1 = tf.split(F_t_Layer, [1, p[-2]- 1], -2)
    transaction_part = delta*tf.abs(tf.subtract(F_t_layer1, F_tminus1_layer1))
    f1= tf.zeros(tf.shape(f1), dtype = float )
    transaction_part = tf.concat([f1, transaction_part], -2)
    output_layer  =  Multiply()([F_t_Layer, r_tplus1_Layer]) -transaction_part
    return output_layer
  
def sharpe_ratio_loss(yTrue,yPred):
  y_shape = K.shape(yPred)
  B = K.mean(K.square(yPred))
  A = K.mean(yPred)
  return -A/((B-A**2)**0.5) 
  
def RRL_Model(input_dim, delta):
  model = Sequential()
  # initilize the input tensor with given shape (input_dim)
  input_layer = Input(shape= input_dim)
  # Step 1: Split the input layer into two parts: X_t_layer and r_tplus1_layer
  X_t_layer = Lambda(split_func1)(input_layer)
  r_tplus1_Layer = Lambda(split_func2)(input_layer)   
  # Step 2: Map X_t_alyer to F_t_layer using SimpleRNN()
  F_t_Layer = SimpleRNN(1, input_shape=X_t_layer._keras_shape[-2:],
    activation = 'tanh', return_sequences = True, use_bias=True)(X_t_layer) # 
  # Step 3: Concatenate F_t_Layer with r_tplus1_Layer
  hidden_layer =  Concatenate()([F_t_Layer, r_tplus1_Layer])
  # Map hidden layer to output layer, which represents the trading return series
  output_layer  = Lambda(trading_return, arguments={'delta':delta})(hidden_layer )
  model = Model(inputs=input_layer, outputs=output_layer)

  sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss=sharpe_ratio_loss, optimizer=sgd)
  return model