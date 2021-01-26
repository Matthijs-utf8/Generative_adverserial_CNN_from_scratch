# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:55:31 2020
@author: Matthijs Schrage and Sietse Schröder
"""

import sys
import os
sys.path.insert(0, os.getcwd())

#Import modules
import Preprocessing
import Model

#Import other necessary packages
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

np.random.seed(0)

#Load and modify traindata so it contains all specified numbers
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Genormaliseerde data werkt veel beter
DP = Preprocessing.DataPreprocessing()
X_train = DP.scale(X_train)
X_test = DP.scale(X_test)

#We moeten voordat we de data in het model stoppen nog even deze stap doen
X_train = np.reshape( X_train, ( X_train.shape + (1,) ) )
X_test = np.reshape( X_test, ( X_test.shape + (1,) ) )

#Prepare data so that it only contains one number (and some noise)
def prepare_data(X, y, number=3, noise=0):
	train_data = [(X[i], 1) for i in range(len(X)) if y[i] == number]
	other_data = [X[i] for i in range(len(X))  if y[i] != number]
	random.shuffle(other_data)
	train_data += [(other_data[i], 0) for i in range(noise)]
	X, y = np.array([x[0] for x in train_data]), np.array([x[1] for x in train_data])
	return X, y

#Create both generator and discriminator models for GAN
def create_models():
	
	generator = Model.Sequential((28,28, 1), batch_norm=False)
	
	generator.add_conv(stride=2, filter_size=3, nr_of_filters=12, activation="relu", pooling=False, K_size=2, method="max")
	generator.add_dense(size=512, activation='leakyrelu')
	generator.add_dense(size=784, activation="leakyrelu")
	generator.add_dense(size=784, activation='sigmoid')
	
	generator.adjust_optimizer(method="adam", learning_rate=0.002, decay=1e-4, epsilon=1e-7, momentum=0.9, rho=0.95)
	
	discriminator = Model.Sequential((28,28,1), batch_norm=False)
	discriminator.add_dense(256, activation='leakyrelu')
	discriminator.add_dense(2, activation='softmax')
	
	discriminator.adjust_optimizer(method="adam", learning_rate=0.002, decay=1e-4, epsilon=1e-7, momentum=0.9, rho=0.95)
	
	return generator, discriminator

class GAN:
	def __init__(self, gen, dis):
		"""
		Hier moeten we nog een zooitje errors gooien zodat beide netwerken altijd op elkaar aansluiten
		"""
		
		
		self.generator = gen
		self.discriminator = dis
		
		#Test if output generator fits on discriminator
		try:
			np.reshape(np.random.random(self.discriminator.input_shape), self.generator.layers[-1].output_shape)
		except:
			raise ValueError("Output of the generator was not accepted as input by the discriminator")
		
		#Initialize batch size to avoid errors, gets changed when train is called
		self.batch_size=32
		
	def generate(self, nr_batches):

		#Generate images
		output = self.generator.forward(np.random.normal(size=((nr_batches * self.batch_size,) + self.generator.input_shape)), batch_size=self.batch_size)
		
		#Save shape of output for backward pass
		self.generator.output_shape = output.shape
		
		#Reshape images so they fit on discriminator
		output = np.reshape(output, ((nr_batches * self.batch_size,) + self.generator.input_shape))
		
		#Return list of generated datapoints
		return output, np.array([0 for _ in range(len(output))])
	
	#Train the discriminator with specified data and labels
	def train_discriminator(self, X, y):
		#Train discriminator
		self.discriminator.fit(X, y, batch_size=self.batch_size)

	#Train the generator
	def train_generator(self, X):
		#Define classifications; array of ones since we want to calculate loss of generator
		y = np.array([1 for _ in range(len(X))])
		
		#Perform forward pass through discriminator network with traindata
		output_discriminator = self.discriminator.forward(X, batch_size=self.batch_size)
		
		#Calculated losses based on traindata and classifications
		losses_discriminator = self.discriminator.loss(output_discriminator, y, batch_size=self.batch_size)[3]

		#Calculate dvalues of generator by backwards pass through discriminator without training
		dvalues_generator = self.discriminator.backward(losses_discriminator, batch_size=self.batch_size)

		#Reshape dvalues so that they fit on the generator
		final_dvalues = np.reshape(dvalues_generator, self.generator.output_shape)

		#Perform backwards pass through generator to train it on dvalues from discriminator
		self.generator.backward(final_dvalues, batch_size = self.batch_size)
	
	def train(self, X, y, batches_real=1, batches_fake=1, epochs=1, batch_size=32):
		
		#Set batch size
		self.batch_size=batch_size
		
		#Combine data and labels to shuffle
		data = [(X[i], y[i]) for i in range(len(X))]
		
		#Train GAN one or more epochs
		for epoch in range(epochs):
			
			#Take random train sample of real data
			train_data = random.sample(data, batches_real * self.batch_size)
		
			#Get final data and labels to train on
			X_train, y_train = np.array([p[0] for p in train_data]), np.array([p[1] for p in train_data])
		
			#Train discriminator on real data
			self.train_discriminator(X_train, y_train)

			#Let generator make fake data
			X_fake, y_fake = self.generate(batches_fake)

			#Train discriminator on fake data
			self.train_discriminator(X_fake, y_fake)

			#Train generator on the same data, with loss caused by discriminator correctly classifying fake data as fake
			self.train_generator(X_fake)
			
			#Test by printing a image generated by generator
			#Dit kan nog wel ietsje soepeler denk ik
			image = np.reshape(self.generate(1)[0][0], (28, 28))
			plt.imshow(image, cmap='gray')
			plt.show()

def main():
	g, d = create_models()
	X, y = prepare_data(X_train, y_train, number=0, noise=1000)
	
	gan = GAN(g, d)
	gan.train(X, y, batches_real=1, batches_fake=2, epochs=100, batch_size=32)

if __name__ == "__main__":
	main()

