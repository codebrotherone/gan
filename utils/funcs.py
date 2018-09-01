import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

# GLOBAL SETTINGS
os.environ["KERAS_BACKEND"] = "tensorflow"
# REPRODUCIBILITY
np.random.seed(17)
# The dimension of our random noise vector.
random_dim = 100

# Functions
def load_minst_data():
    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)

# You will use the Adam optimizer
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def create_disc_model(optimizer):
	"""This will create a discriminator model for our GAN network using Keras
	Args:
		Optimizer (keras.optimizers): Adam
	Returns:
		discriminator (keras.SequentialModel): discriminator model
	"""
	discriminator = Sequential()
	# first layer
	discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
	discriminator.add(LeakyReLU(.2))
	discriminator.add(Dropout(.3))
	# second layer
	discriminator.add(Dense(512))
	discriminator.add(LeakyReLU(.2))
	discriminator.add(Dropout(.3))
	# third layer
	discriminator.add(Dense(256))
	discriminator.add(LeakyReLU(.2))
	discriminator.add(Dropout(.3))

	discriminator.add(Dense(1, activation='sigmoid'))
	discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
	return discriminator

def create_gen_model(optimizer):
	"""This will create the generator model which will use tanh activation for output
	Notes:
		https://stackoverflow.com/questions/41489907/generative-adversarial-networks-tanh
	Args:
		Optimizer (keras.optimizers): Adam
	Returns:
		generator (keras.SequentialModel): generator model
	"""
	generator = Sequential()
	generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
	generator.add(LeakyReLU(0.2))

	generator.add(Dense(512))
	generator.add(LeakyReLU(0.2))

	generator.add(Dense(1024))
	generator.add(LeakyReLU(0.2))

	generator.add(Dense(784, activation='tanh'))
	generator.compile(loss='binary_crossentropy', optimizer=optimizer)
	return generator

def create_gan_network(discriminator, random_dim, generator, optimizer):
	# We initially set trainable to False since we only want to train either the
	# generator or discriminator at a time
	discriminator.trainable = False
	# gan input (noise) will be 100-dimensional vectors
	gan_input = Input(shape=(random_dim,))
	# the output of the generator (an image)
	x = generator(gan_input)
	# get the output of the discriminator (probability if the image is real or not)
	gan_output = discriminator(x)
	gan = Model(inputs=gan_input, outputs=gan_output)
	gan.compile(loss='binary_crossentropy', optimizer=optimizer)
	return gan

# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
	noise = np.random.normal(0, 1, size=[examples, random_dim])
	generated_images = generator.predict(noise)
	generated_images = generated_images.reshape(examples, 28, 28)

	plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0], dim[1], i+1)
		plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
		plt.axis('off')
	plt.tight_layout()
	plt.savefig('gan_generated_image_epoch_%d.png' % epoch)

