import numpy
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives

"""
My implementation of a Variational Autoencoder
Inspired by Francois Chollet (Keras) https://github.com/fchollet/keras
and Carl Doersch (Tutorial on VAE) https://arxiv.org/pdf/1606.05908v2.pdf
Data used: 100 transcriptomes (RNA-Seq data) from Lung and Kidney cells obtained from The Cancer Genome Atlas (https://tcga-data.nci.nih.gov)
Input: 20532 gene expression values per sample.
Latent space: 2D, Gaussian.
"""

"""
Defining a couple things about my VAE.
input dimensions, size of latent space etc. 
"""
original_dim = 20532
batch_size = 10
latent_dim = 2
hidden_dim = 1500
nb_epoch = 10
epsilon_std = 1.0


"""
Plotting function - totally optionnal!

"""
def draw_graph(data, classes):
    plt.scatter(data[:, 0], data[:, 1], c=classes)
    plt.colorbar()
    plt.show()

"""
Loading data.
Y is a file containing the transcriptomes of Lung cells
Z is a file containing the transcriptomes of Kidney cells.
"""

data1 = numpy.loadtxt(".gitignore/y")
data2 = numpy.loadtxt(".gitignore/z")
x_test = data1[:,63:66]
data1 = data1[:,:63]

y_train = numpy.concatenate((numpy.zeros(data1.shape[1]), numpy.ones(data2.shape[1])))

x_train = numpy.hstack((data1,data2)).transpose()
x_train = x_train.astype('float32') / numpy.max(x_train)
x_train = x_train.reshape((len(x_train), numpy.prod(x_train.shape[1:])))


original_dim = 20532
batch_size = 10
latent_dim = 2
hidden_dim = 1500
nb_epoch = 10
epsilon_std = 2.0


"""
Key players in the VAE:
X are inputs. They are the data that goes into the model.
z are values in latent space. 
For each X there is a z in latent space. The neural network learns to chose a good z to describe each X (reconstruct)
We want to maximise the sampled z for X, to make sure the z chosen by the model accurately describes X. This means we want to maximise P(X|z).
We can solve this by sampling but it might take a lot of samples, since Z domain is vast and so P(X|z) ~ 0 most of the time.
Since p(x) \approx   \frac{1}{n}\sum_{i}p(X|z_{i}) is too vast, we could define another distribution Q(z|X), that is contained in P(X|z) and is a subset.
We will then make Q(z|X) most likely with P(X|z), by minimizing the Kullback-Liebler Divergense between them.
"""



"""
Creating a sampling function, that samples for z  through Q (which we chose to be Gaussian (mean=0, covar=Identity))
"""
def sampling(args):
z_mean, z_log_var = args
epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                          std=epsilon_std)
return z_mean + K.exp(z_log_var / 2) * epsilon

"""
Defining layers (input, hidden, z_mean and z_log_var (parameters of the latent space))
"""
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(hidden_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

"""
Helpful Keras Lambda function.
"""
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
decoder_h = Dense(hidden_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

"""
Building a model to decode from the latent space
"""
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
decoder_from_z = Model(decoder_input, _x_decoded_mean)

"""
Making an encoder that will transfer inputs to latent space.
"""
encoder = Model(x, z_mean)

"""
Defining the loss. We want to penalize two things:
1 - The reconstruction loss from X to reconstructed X. (Autoencoder part of the VAE)
2 - The Kullback-Liebler Divergence between Q(z|X) and P(z) (making sure z is accurately chosen to represent X)
"""
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

"""
Instantiation of the model and fitting the data
"""
vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)
vae.fit(x_train, x_train, shuffle=True, nb_epoch=nb_epoch, batch_size=batch_size)

"""
Encoding the input data for it's representation in 2D latent space + plotting the data.
"""

x_train_encoded = encoder.predict(x_train)
#draw_graph(x_train_encoded,y_train)
