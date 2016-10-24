from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Merge, Lambda
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.regularizers import l2, activity_l2, l1
from keras.initializations import glorot_normal, identity
from keras.models import model_from_json
from keras.layers import Input, merge
from keras import backend as K
from keras import objectives