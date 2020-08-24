# -------------------------------------------------------------------------------- #
#   Given the solutions in the notebook 12_LSTM_vs_1DConv_solution.ipynb this script
# uses tensorflow2
# -------------------------------------------------------------------------------- #

# ## Prediction of time series with different neural networks architectures
import os, sys
import numpy as np
from matplotlib import pyplot as plt
import h5py

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, Convolution1D,LSTM, SimpleRNN

# ------------------------------- functions -------------------------------------- # 
def gen_data(size=1000, noise=0.1,seq_length=128,look_ahead=10):
  ''' Create 1000 realizations of the process, consisting each of them of 138 observations,
  i.e. seq_length + look_ahead. Return these realizations split into a part that has been 
  observed (first 128 observations in each realization) and a part to be forecasted.

  '''
  s = seq_length + look_ahead
  d = np.zeros((size, s,1))
  for i in range(size):
    start = np.random.uniform(0, 2*np.pi) # Random start point
    d[i,:,0] = np.sin(start + np.linspace(0, 20*np.pi, s)) * np.sin(start + np.linspace(0, np.pi, s)) + np.random.normal(0,noise,s)
  return d[:,0:seq_length], d[:,seq_length:s], np.arange(0,s)

def slice(x, slice_length):
    return x[:,-slice_length:,:]

def m1D(ks, lah, X):
  ''' 1D Convolution without dilation rate
  Here we define a Neural network with 1D convolutions and "causal" padding, in a later step 
  we will also use a dilation rate.
  '''
  m = Sequential()
  m.add(Convolution1D(filters=32, kernel_size=ks, padding='causal', input_shape=X.shape[1:]))
  m.add(Convolution1D(filters=32, kernel_size=ks, padding='causal'))
  m.add(Convolution1D(filters=32, kernel_size=ks, padding='causal'))
  m.add(Convolution1D(filters=32, kernel_size=ks, padding='causal'))
  m.add(Dense(1))
  m.add(Lambda(slice, arguments={'slice_length':lah}))

  return m

def m1D_dilation(ks, lah, X, drL):
  ''' 1D Convolution with dilation rate
  
  '''
  m = Sequential()
  m.add(Convolution1D(filters=32, kernel_size=ks, padding='causal', dilation_rate=1, 
    input_shape=X.shape[1:]))

  for i in range(1, len(drL)):
    m.add(Convolution1D(filters=32, kernel_size=ks, padding='causal', dilation_rate=drL[i]))
  m.add(Dense(1))
  m.add(Lambda(slice, arguments={'slice_length':lah}))

  return m

def mRNN(nunits, lah, X):
  '''Fully connected RNN, number of hidden states defined by nunits.
  '''
  m = Sequential()
  m.add(SimpleRNN(nunits,return_sequences=True,input_shape=X.shape[1:]))
  m.add((Dense(1)))
  m.add(Lambda(slice, arguments={'slice_length':lah}))

  return m

def mLSTM(noi, in_sh, lah):

  m = Sequential()
  m.add(LSTM(noi,return_sequences=True,input_shape=in_sh))
  m.add((Dense(1)))
  m.add(Lambda(slice, arguments={'slice_length':lah}))

  return m

def _predict_steps(model, x_, lah, steps):

  yhat = model.predict(x_)
  Yhat = [yhat]
  i0=lah
  for j in range(1,steps):
  
    x_ = np.concatenate((x_, yhat), axis=1)
    yhat = model.predict(x_[:,i0:,:])
    i0 += lah
  
    Yhat.append(yhat) 
  
  return Yhat

def predict_steps(model, data, steps):
  
  pred = np.empty(0)
  for j in range(0,steps):
    res = model.predict(data)
    data= np.concatenate((data,res), axis=1)[:,-data.shape[1]:,:]
    pred=np.append(pred,res)
  
  return pred

def plt_sequences():
  for i in range(1):
    figure(num=None, figsize=(13,5))  
    plot(range(0, seq_length),X[i,:,0],'b-', label='input')
    plot(range(seq_length, seq_length + look_ahead),Y[i,:,0],'b-',color='orange', label='10 steps ahead')
    legend()

def _plt__predict(Yhat):
  figure(num=None, figsize=(13,5))
  plot(range(0,128),data.reshape(-1),color='blue')
  axvline(x=128)
  
  i0=seq_length
  for i in range(steps): 
    _ise = range(i0,i0+look_ahead)
    #plot(_ise,pred, marker='s', color='black', linestyle=':',
    #  markerfacecolor='None')
    plot(_ise, Yhat[i].reshape(-1), color='red', linewidth=2, alpha=0.5)
  
    i0+=look_ahead
# ----------------------------- Simulate data ------------------------------------------------ #
# First we generate train and validation data. We multiply a fast sine wave with a 
#slower sine wave and add a bit random noise. The goal is to learn from the past of time series 
#and predict the next 10 steps and later even more than "only"  10 steps.
# 
np.random.seed(1) # Fixing the seed, so data generation is always the same
seq_length = 128  # Sequence length used for training
look_ahead =  10  # The number of data points the model should predict 

X,Y, iseq = gen_data()
# plt_sequences()

# -------------------------------- simple RNN ------------------------------------------------- #
model_simple_RNN = mRNN(12, look_ahead, X)
model_simple_RNN.summary()
model_simple_RNN.compile(optimizer='adam', loss='mean_squared_error')

history = model_simple_RNN.fit(X[0:800], Y[0:800], epochs=500, batch_size=128,
  validation_data=(X[800:1000],Y[800:1000]),verbose=1)

# ---------------------- 1D Convolution without dilation rate --------------------------------- #
ks = 5
model_1Dconv = m1D(ks, look_ahead, X)
model_1Dconv.compile(optimizer='adam', loss='mean_squared_error')
model_1Dconv.summary()
model_1Dconv.fit(X[0:800], Y[0:800], epochs=100, batch_size=128,
                    validation_data=(X[800:1000],Y[800:1000]),
                    verbose=1)

# ----------------------- 1D Convolution with dilation rate ----------------------------------- #
ks = 5
drL = [1,2,4,8] # dilation rates
model_1Dconv_w_d=m1D_dilation(ks, look_ahead, X, drL)  
model_1Dconv_w_d.compile(optimizer='adam', loss='mean_squared_error')
model_1Dconv_w_d.summary()

model_1Dconv_w_d.fit(X[0:800], Y[0:800], epochs=100, batch_size=128,
                    validation_data=(X[800:1000],Y[800:1000]), verbose=1)

# --------------------- Use a model to predict 10 steps ahead -------------------------------- # 
i=950 # select one realization
steps = 1 # (!) since look_ahead is 10, this effectively means 10 steps 
data=X[i:i+1]  # (1, 128, 1) # (*!) notice that this sequence was used as part of the validation
model = model_1Dconv_w_d # select model
model_name = '1D convolution with dilation rate'
pred = model.predict(data)[0].reshape(-1,) #batch_size=32, steps=None *!?

plt.figure(num=None, figsize=(13,5))
plt.plot(iseq[:seq_length],data.reshape(-1,),color='blue', label='input')
plt.plot(iseq[seq_length:],pred,color='orange', label='prediction')
plt.title(model_name)
plt.legend()
plt.axvline(x=seq_length)

# In addition, we want to predict for longer than just 10 steps, we will  
#just predict the next 10 steps and take the predictions as new "true" observations 
#and feed these values into the model, 
#when we do that we can predict for any length we want. 

### predict 8 times, updating x
i=950
steps = 8
data = X[i:i+1]

#Yhat = _predict_steps(model_1Dconv, data, look_ahead, steps)
#_plt__predict(Yhat)

pred =  predict_steps(model_1Dconv, data, steps)

# -------------------------------------- LSTM ------------------------------------------- #
# (!)'../data/models/lstm_model_500_epochs.h5' fails to load
fn = '../data/models/lstm_model_500_epochs_without_optimizer_states.h5' # (!) I saved this one
                                                                     # without optimizer steps
flg_save=False # change to True to save model after training

if os.path.exists(fn):
  print('Loading saved model {:}'.format(fn))
  model_LSTM = load_model(fn) # load and compile
else:
  print('Building model ... ')
  noi =12
  n_out = 1
  model_LSTM = mLSTM(noi, X.shape[1:], look_ahead)
  model_LSTM.compile(optimizer='adam', loss='mean_squared_error')

  history = model_LSTM.fit(X[0:800], Y[0:800,:,:], epochs=500, batch_size=128, 
          validation_data=(X[800:1000],Y[800:1000,:,:]), verbose=1)
  if flg_save:
    model_LSTM.save(fn, optimizer=False)

model_LSTM.summary()

pred =  predict_steps(model_LSTM, data, steps)
plt.figure(num=None, figsize=(13,5))
plt.plot(iseq[:seq_length], data.reshape(-1,), color='blue')
plt.plot(range(seq_length, seq_length+len(pred)), pred, color='orange')
plt.axvline(x=seq_length)

# ----------------------------- compare models ------------------------------------------ #
models = [model_simple_RNN, model_1Dconv, model_1Dconv_w_d, model_LSTM]
mnames = ['RNN', '1Dconv', '1Dconv_w_d', 'LSTM']
D ={}
for m,n in zip(models, mnames):
  D[n] = predict_steps(m,data,steps)

for i in D:
  plt.plot(D[i], label=i)