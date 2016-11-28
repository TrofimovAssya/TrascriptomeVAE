from keras.layers.core import Dense, Dropout, Activation, Merge
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential , model_from_json, Model
from keras.callbacks import Callback
from keras.regularizers import l2, activity_l2, l1
from keras.initializations import glorot_normal, identity
from keras.layers import Input, merge
from keras import objectives

from numpy import random
import numpy
from sklearn import preprocessing
numpy.random.seed(32689)


def warp(data_x,y_train):
	batch_size = 10
	nb_epoch = 10
	x_train = data_x

	###construct the model
	model = Sequential()
	model.add(Dense(2, input_shape=(2,), activation='tanh'))
	model.add(Dense(2,activation='softmax'))

	###compile and summarize model
	model.summary()
	model.compile(loss='binary_crossentropy',
	          optimizer=RMSprop(lr=1e-2),
	          metrics=['accuracy'])
	###training
	model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, shuffle=True)

	##loading warping datasets
	y_train2 = numpy.copy(y_train)
	for i in xrange(len(y_train2)):
		if y_train2[i][1] == 0:
			y_train2[i] = [0.,1.]



	###construction of the warp setup with added bias layer in the beggining
	warp = Sequential()
	warp.add(Dense(2,input_shape = (2,), init='identity'))
	warp.add(Dense(2, input_shape = (2,),  activation='tanh', weights = [model.layers[0].W.get_value(), model.layers[0].b.get_value()]))
	warp.add(Dense(2, activation = 'softmax', weights = [model.layers[1].W.get_value(), model.layers[1].b.get_value()]))

	for i in xrange(len(model.layers)):
		warp.layers[i+1].trainable = False


	### making sure only bias is allowed to change
	warp.layers[0].non_trainable_weights.append(warp.layers[0].trainable_weights[0])
	warp.layers[0].trainable_weights = warp.layers[0].trainable_weights[1:]


	warp.summary()
	warp.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-3), metrics=['accuracy'])

	def stop_training(model, x_train, y_train2, bias):
		return numpy.argmax(y_train2) == numpy.argmax(model.predict(x_train+bias))

	nb_epoch=300
	sgd = RMSprop(lr=1e-3)
	biases = []
	"""  make a Callback for monitoring for each point"""
	for point in xrange(x_train.shape[0]):
		print "data point #" + str(point)
		print " "
		### setting the biases to zero between each data point
		warp.layers[0].b.set_value(numpy.zeros(warp.layers[0].b.get_value().shape[0],dtype="float32"))
		### warping....
		epochs=0
		while epochs<nb_epoch:
			if numpy.argmax(y_train2[point:point+1][0]) == numpy.argmax(model.predict(x_train[point:point+1]+warp.layers[0].b.get_value())):
				break
			print "Epoch "+str(epochs) + " - patient " +str(point)
			print " "
			print(model.predict(x_train[point:point+1]))
			print((warp.predict(x_train[point:point+1])))
			warp.fit(x_train, [y_train2],
	                batch_size=batch_size, nb_epoch=1,
	                verbose=0, shuffle=True)
			epochs+=1
		biases.append(warp.layers[0].b.get_value())
	return biases

def gaussian(x):
    mu = numpy.array([numpy.mean(x[:,i]) for i in xrange(x.shape[1])])
    cov = numpy.cov(x)
    return mu,cov

def gaussian_align(data,separator):
	mu1,cov1 = gaussian(data[:separator,:])
	mu2,cov2 = gaussian(data2[separator:,:])
	new_data = data[:separator,:]-mu1+mu2
	return new_data
