# -------------------------------------------------------------------------------- #
#   Given the solutions in the notebook 12_LSTM_vs_1DConv_solution.ipynb this script
# presents some amendments. It is meant to be run from the ipython terminal, 
# -------------------------------------------------------------------------------- #

# ## Prediction of time series with different neural networks architectures
# load required libraries:
# (!) activate %pylab
import matplotlib.pyplot as plt
import h5py

import keras 
from keras.models import Sequential
from keras.layers import Dense, Lambda, Convolution1D,LSTM, SimpleRNN

# --- functions --- # 
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

def plt_sequences():
  for i in range(1):
    plt.figure(num=None, figsize=(13,5))  
    plt.plot(range(0, seq_length),X[i,:,0],'b-')
    plt.plot(range(seq_length, seq_length + look_ahead),Y[i,:,0],'b-',color='orange')
  
  plt.show()
  print('The training data X is the blue line and we want to forecast the next 10 steps Y, the orange line.')

# --------- #

# ### Simulate data
# First we generate train and validation data. We multiply a fast sine wave with a 
#slower sine wave and add a bit random noise. The goal is to learn from the past of time series 
#and predict the next 10 steps and later even more than "only"  10 steps.
# 
np.random.seed(1) # Fixing the seed, so data generation is always the same
seq_length = 128  # Sequence length used for training
look_ahead =  10  # The number of data points the model should predict 

X,Y, iseq = gen_data()
plt_sequences()

print(X.shape)
print(Y.shape)

# ### 1D Convolution without dilation rate
ks = 5
model_1Dconv = m1D(ks, look_ahead)
model_1Dconv.compile(optimizer='adam', loss='mean_squared_error')
model_1Dconv.summary()
model_1Dconv.fit(X[0:800], Y[0:800], epochs=100, batch_size=128,
                    validation_data=(X[800:1000],Y[800:1000]),
                    verbose=1)

# Now we want to use the trained model to predict for the next 10 steps, 
#remember that is what the model was trained on.  
i=950
steps = 1 # select one realization
data=X[i:i+1]  # (1, 128, 1) # (*!) notice that this sequence was used as part of the validation
model = model_1Dconv

pred = model.predict(data)[0].reshape(-1,) #batch_size=32, steps=None *!?

plt.figure(num=None, figsize=(13,5))
plt.plot(iseq[:seq_length],data.reshape(-1,),color='blue')
plt.plot(iseq[seq_length:],pred,color='orange')
plt.axvline(x=seq_length)

# In addition, we want to predict for longer than just 10 steps, we will  
#just predict the next 10 steps and take the predictions as new "true" observations 
#and feed these values into the model, 
#when we do that we can predict for any length we want. 

### predict for 80 steps
i=950
steps = 8
pred8 = np.empty(0)
model = model_1Dconv
data=X[i:i+1] 
for j in range(0,steps): # (!) this is the same as stacking pred 8 times
  res = model.predict(data)[0].reshape(10)
  pred8=np.append(pred8,res)
 
### predict 8 times, updating x
x_ = X[i:i+1]
i0=look_ahead
yhat = model.predict(x_)
Yhat = [yhat]
for j in range(1,steps):

  x_ = np.concatenate((x_, yhat), axis=1)
  yhat = model.predict(x_[:,i0:,:])
  i0 += look_ahead

  Yhat.append(yhat) 

# ---------------------- plots --------------------------------------- #
plt.figure(num=None, figsize=(13,5))
plt.plot(range(0,128),data.reshape(-1),color='blue')
plt.plot(range(128,128+steps*10),pred8,color='orange',marker='.')
plt.axvline(x=128)

i0=seq_length
for i in range(steps): 
  _ise = range(i0,i0+look_ahead)
  plot(_ise,pred, marker='s', color='black', linestyle=':',
    markerfacecolor='None')
  plot(_ise, Yhat[i].reshape(-1), color='red', linewidth=2, alpha=0.5)

  i0+=look_ahead
# ------------------------------------------------------------------- #
