# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:04:02 2020
@author: Sietse Schöder und Matthijs Schräge
"""

import Layers as AA
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, variance
import random
import time
import os
import pickle
import warnings
import keyboard

random.seed(0)
np.random.seed(0)


class Sequential:
	
	def __init__(self, input_shape, batch_norm=False):
		
		self.input_shape = input_shape
		
		#Initialize the layers attribute that keeps track of how the NN is built up. 
		self.layers = []
		
		self.batch_norm = batch_norm
		
		#Initialize an optimizer. Optimizer can be changed with the "adjust_optimizer" function.
		self.optimizer = AA.Optimizer()
		self.loss_function = AA.Loss_CategoricalCrossentropy()
		
	def check_data(self, inputs):
		
		#Check if the datatype is a list. If so, convert it to an array.
		if type(inputs) == list:
			inputs = np.array(inputs)
			print("Converted type 'list' data to numpy.ndarray. New shape: ", inputs.shape)
		
		#If the datatype is not an array at this point, raise an error.
		if type(inputs) != np.ndarray:
			raise TypeError("Only numpy arrays accepted as datatype. Your type: ", type(inputs))
		
		#The nr of dimensions in the data should be 4. So do nothing.
		if inputs.ndim != 4:
			raise ValueError("Incorrect number of input dimensions: ", len(inputs.shape))
		
		#The new inputs (converted if something was wrong with the input shape or type, otherwise just the same)
		return inputs
	
	#Add a dense layer to the network with an activation funtion.
	def add_dense(self, size=64, activation="relu"):
		
		#Check if this is the first layer of the network. If it is, we take the input shape of the data, otherwise we take the output shape of the previous layer.
		input_shape = self.layers[-1].output_shape if self.layers else self.input_shape
		
		#If the previous layer is not a dense layer (which has an output dimension of 1), we add a Flatten layer first.
		if len(input_shape) == 3:
			self.layers.append( AA.Flatten(input_shape) )
			input_shape = self.layers[-1].output_shape
			
			#Now we add the dense layer the the model
			self.layers.append(AA.Layer_Dense(input_shape, size))
			
		elif len(input_shape) == 1:
			
			#Else we add the dense layer the the model
			self.layers.append(AA.Layer_Dense(input_shape, size))
		
		#Just a check for the dimensions of the input
		else:
			raise ValueError("Probeert een Flatten of Dense layer aan te maken terwijl de input niet klopt")
		
		#Ad the specified activation function to the model as well
		self.add_activation(activation=activation)

	#Add a convolutional layer to the network with an activation fucntion
	def add_conv(self, stride=1, filter_size=3, nr_of_filters=9, activation="relu", pooling=False, K_size=2, method="max"):
		
		#Check if this is the first layer of the network. If it is, we take the input shape of the data, otherwise we take the output shape of the previous layer.
		input_shape = self.layers[-1].output_shape if self.layers else self.input_shape
		
		#Add the convoluitonal layer to the model
		self.layers.append(AA.Layer_Conv(input_shape, stride, filter_size, nr_of_filters))
		
		#Add the specified activation function to the model as well
		self.add_activation(activation=activation)
		
		#If we specified pooling=True, then add a pooling layer after the activation
		if pooling:
			self.add_pooling(K_size=K_size, method=method)
	
	#Method for adding a pooling layer to the network.
	def add_pooling(self, K_size=2, method="max"):
		
		#Check if the method exists
		if method != "mean" and method != "max":
			raise ValueError("Method " + str(method) + " does not exist.")
		
		#Check if this is the first layer of the network. If it is, we take the input shape of the data, otherwise we take the output shape of the previous layer.
		input_shape = self.layers[-1].output_shape if self.layers else self.input_shape
		
		#If the previous layer is not 4 dimensional (so if ppoling is not added after a convolutional layer), raise an error
		if len(input_shape) < 3:
			raise TypeError("The input_shape of the pooling layer should be 3 dimensions, but it is " + str(len(input_shape)) + " dimensions.")
		
		#Add the pooling layer.
		self.layers.append(AA.Layer_Pooling(input_shape, K_size=K_size, method=method))
	
	#A method for adding activation layers to the network.
	def add_activation(self, activation="relu"):
		
		#If this is the first layer of the network, raise an error
		if not self.layers:
			raise TypeError("ReLU can not be the first layer of a network!")
		
		self.layers.append(AA.Activation(self.layers[-1].output_shape, method=activation, batch_norm=self.batch_norm))
	
	#Adjust the optimizer at any point
	def adjust_optimizer(self, method="adam", learning_rate=1, decay=0, epsilon=1e-7, momentum=0.9, rho=0.999):
		self.optimizer = AA.Optimizer(method=method, learning_rate=learning_rate, decay=decay, epsilon=epsilon, momentum=momentum, rho=rho)
	
	def adjust_loss_function(self):
		pass
	
	def forward(self, inputs, batch_size=32):
		#Check if given input is equal or bigger than batch size
		if inputs.shape[0] < batch_size:
			raise ValueError("Number of data points smaller than batch size")
		
		#Execute a regular forward pass
		elif inputs.shape[0] >= batch_size and inputs.shape[0] < 2 * batch_size:

			#Peform a forward pass through the network
			for index, layer in enumerate(self.layers):
				
				#If we are at the first layer in the network, take the batch as input, else the output from the previous layer
				layer.forward(inputs[:batch_size]) if index == 0 else layer.forward(self.layers[index - 1].output)

			#Return the output of the last layer
			return layer.output
		
		#If the input is at least equal to twice the batch size, pass multiple times through the network
		elif inputs.shape[0] >= 2 * batch_size:

			#Define number of output batches
			nr_batches = inputs.shape[0] // batch_size
			output_batches = []
			
			#Perform forward pass for each batch
			for batch_nr in range(nr_batches):
				
				#The batch that we are putting through the network
				batch = inputs[batch_nr*batch_size : (batch_nr+1)*batch_size]
				
				#Recursively add output batches to the result
				output_batches.append(self.forward(batch, batch_size=batch_size))
			
			return np.concatenate(output_batches)
	
	def loss(self, inputs, classification, batch_size=32):
		
		#Raise error if batch size is bigger than input size
		if inputs.shape[0] < batch_size:
			raise ValueError("Not enough dvalues for batch size")
			
		#If this is True: then we just use 1 batch.
		elif inputs.shape[0] >= batch_size and inputs.shape[0] < 2 * batch_size:
			
			loss = self.loss_function.forward(inputs, classification)
		
			#Compute the accuracy
			predictions = np.argmax(self.layers[-1].output, axis=1)
			accuracy = np.mean(predictions==classification)
			
			#Calculate dvalues for backward pass
			dvalues = self.loss_function.backward(inputs, classification)
			
			return loss, predictions, accuracy, dvalues
		
		#If inputs contains multiple batches, calculate all and return resulting array
		elif inputs.shape[0] >= 2 * batch_size:

			#Specify number of batches inputs contains
			nr_batches = inputs.shape[0] // batch_size
			loss_batches = []
			
			#Perform forward pass for each batch
			for batch_nr in range(nr_batches):
				
				#The batch that we are putting through the network
				batch = inputs[batch_nr*batch_size : (batch_nr+1)*batch_size]
				classifications_batch = classification[batch_nr*batch_size : (batch_nr+1)*batch_size]
				
				#Recursively add losses of batches to the result
				batch_loss = self.loss(batch, classifications_batch, batch_size=batch_size)
				
				loss_batches.append(batch_loss[3])
				
			#Return losses of all batches concatenated
			return None, None, None, np.concatenate(loss_batches)
	
	def backward(self, dvalues, train=True, batch_size=32):
		#Throw error if there are less points than batch size
		if dvalues.shape[0] < batch_size:
			raise ValueError("Not enough dvalues for batch size")
		
		#Execute a regular backward pass
		elif dvalues.shape[0] >= batch_size and dvalues.shape[0] < 2 * batch_size:
			
			#Reverse order of layers in network to pass backwards
			self.layers = list(reversed(self.layers))
			
			#Calculate decay
			self.optimizer.pre_update_params()
			
			#Perform backward pass
			for index, layer in enumerate(self.layers):
				#If we are at the first (actually the last) layer in the network, use dvalues from function call.
				if index == 0:
					layer.backward(dvalues)

				#Else, we take the dvalues from the previous (or next) layer. We update the layers weights and biases if it has the attribute "weights"
				else:
					layer.backward(self.layers[index-1].dvalues)
					if hasattr(layer, "weights"):
						if train:
							self.optimizer.update_params(layer)

			
			#Re-reverse the list for the next forward pass
			self.layers = list(reversed( self.layers ) )
			
			if train:
				self.optimizer.post_update_params()
			
			#Return dvalues of first layer of the network
			return layer.dvalues
		
		#If dvalues contains multiple batches, pass backwards for all of them
		elif dvalues.shape[0] >= 2 * batch_size:
			#Specify how many batches dvalues contains
			nr_batches = dvalues.shape[0] // batch_size
			dvalues_batches = []
			
			#Perform forward pass for each batch
			for batch_nr in range(nr_batches):
				
				#The batch that we are putting through the network
				batch = dvalues[batch_nr*batch_size : (batch_nr+1)*batch_size]

				#Recursively add output batches to the result
				dvalues_batches.append(self.backward(batch, train=train, batch_size=batch_size))
			
			return np.concatenate(dvalues_batches)
	
	#Fit the data to the network.
	def fit(self, inputs, classifications, epochs=1, batch_size=32, shuffle=False, train=True):
		
		#Check if the batch_size is not too big
		if batch_size > inputs.shape[0]:
			raise ValueError("Batch size ", batch_size, " is bigger thean the number of datapoints ", inputs.shape[0])
		
		accuracies = []
		losses = []
		
		#Add inputs to model
		inputs = self.check_data(inputs)
		
		#Calculate the number of batches per epoch
		nr_of_batches = inputs.shape[0] // batch_size
		
		#Train the model on x number of epochs
		for epoch in range(epochs):
			
			#Staat helemaal onderaan Model.py
			if shuffle == True:
				inputs, classifications = shuffle_data(inputs, classifications)
			
			#Train the model in batches
			for batch_nr in range(nr_of_batches):
				
				#The batch that we are putting through the network
				batch = inputs[batch_nr*batch_size : (batch_nr+1)*batch_size]
				
				#The classes that accompany the current batch
				classification = np.array(classifications[batch_nr*batch_size : (batch_nr+1)*batch_size])
				
				#Forward pass
				last_output = self.forward(batch, batch_size=batch_size)
				
				#Compute loss and dvalues
				loss, predictions, accuracy, dvalues = self.loss(last_output, classification, batch_size=batch_size)
				accuracies.append(accuracy)
				losses.append(loss)
				
				print('epoch:', epoch, 'batch:', batch_nr, 'acc:', round(accuracy, 4), 'loss:', round(loss, 4))
				
				#Backward pass
				self.backward(dvalues, train=train, batch_size=batch_size)
		
		plt.plot(accuracies, color="red")
		plt.show()
		
		plt.plot(losses, color="green")
		plt.show()
		
	def test(self, test_data, classifications):
		
		#Perform forward pass
		last_output = self.forward(test_data, batch_size=test_data.shape[0])
		
		#Compute the loss
		loss, predictions, accuracy, dvalues = self.loss(last_output, classifications, batch_size=test_data.shape[0])
		
		print("Mean loss: ", loss )
		print("Mean accuracy: ", accuracy)
		self.test_accuracy = accuracy
	
	def save(self, filename=None):
		
		#Create saved_models folder if one not already exists
		if not os.path.exists('saved_models'):
			os.mkdir('saved_models')
		
		#Create filename if none is specified
		if not filename:
			try:
				filename = 'test' + str(len(os.listdir('saved_models/')) + 1) + '_accuracy_' + str(round(self.test_accuracy,3))
			except:
				#Warn that save has failed and quit method
				warnings.warn("Save failed. Model has not been tested yet.")
				return
		
		#Initialize filename by adding .pickle if necessary
		filename = filename + ".pickle" if filename[-7:] != '.pickle' else filename
		
		stop = False
		
		#Confirm overwriting file if file already exists
		if os.path.exists('saved_models/' + filename):
			print("File", filename, "already exists. Do you want to overwrite file?")
			print("Press enter to continue. Press backspace to abort.")
			b = True
			
			while b:
				a = keyboard.read_key()
				if a == 'enter':
					b = False
				if a == 'backspace':
					b = False
					stop = True
				
				time.sleep(0.2)
		
		#Try saving; this fails if model has not been tested yet
		if not stop:
			try:
				#Specify some information to be saved
				optimizer_details = vars(self.optimizer)
	
				#Delete information to reduce storage space used
				self.inputs = None
				
				#Specify final information to be saved
				to_save = (self, optimizer_details, self.test_accuracy)

				#Save model
				with open('saved_models/' + filename, 'wb') as f:
					pickle.dump(to_save, f)
		
				#Confirm succesful save
				print("Model saved in saved_models folder, under filename", filename)
			
			#Issue a warning if saving failed
			except:
				warnings.warn("Save failed. Model has not been tested yet.")

def load(filename=None, flag='latest'):
	#Load file from filename if filename is specified
	if filename:
		
		#Add .pickle to filename if ne
		filename = filename + ".pickle" if filename[-7:] != '.pickle' else filename
		
		#Try loading the file. If the file does not exist, throw an error
		try:
			with open('saved_models/' + filename, 'rb') as f:
				model, details, accuracy = pickle.load(f)
				
			#Print details of loaded model
			print("\nModel with filename", filename, "loaded. Details model:")
			print("Test Accuracy:", accuracy)
			for key in details.keys():
				print(str(key) + ': ' + str(details[key]))
				
			return model
		#Trow error if file does not exist
		except:
			raise FileNotFoundError("File with filename " + filename + " does not exist.")
	
	else:
		#Specify values
		best = (0, 0, 0)
		latest = 0
		name = None
		
		#Load all saved files to compare their scores
		for filename in os.listdir('saved_models/'):
			
			#Load file
			with open('saved_models/' + filename, 'rb') as f:
				model, details, accuracy = pickle.load(f)
				
				#Test if loaded model scored better than the best model yet
				if flag == 'best':
					best = (model, details, accuracy) if accuracy > best[2] else best
					name = filename
					
				#Test if loaded model is modified later than latest model yet
				elif flag == 'latest':
					if latest < os.path.getmtime(os.path.join('saved_models/', filename)):
						latest = os.path.getmtime(os.path.join('saved_models/', filename))
						best = (model, details, accuracy)
						name = filename
				
				#Throw error if invalid flag is used
				else:
					raise ValueError("This flag is not valid.")

		#If the folder contains a file
		if best != (0, 0, 0):
			
			#Get details of model
			model, details, accuracy = best
			
			#Print details of loaded model
			print("\nModel with filename", name, "loaded. Details model:")
			print("Test Accuracy:", accuracy)
			for key in details.keys():
				print(str(key) + ': ' + str(details[key]))
			print('')
			
			#Return the loaded model
			return best[0]

		#Throw error if no files are saved in the folder
		else:
			raise FileNotFoundError("There are no saved models.")

def shuffle_data(inputs, classifications):
	
	data = [(inputs[i], classifications[i]) for i in range(len(inputs))]
	random.shuffle(data)
	
	inputs = [i[0] for i in data]
	classifications = [i[1] for i in data]
	
	return np.array(inputs), np.array(classifications)