# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:12:25 2020

@author: Sietse Schröder and Matthijs Schrage
"""

import numpy as np
import tensorflow  as tf
import time
# np.set_printoptions(precision=1, suppress=True)

np.random.seed(0)

### PROFILING FUNCTION TO CHECK THE SPEED OF OUR CODE ###
import cProfile, pstats, io
def profile(fnc):

# """A decorator that uses cProfile to profile a function"""

	def inner(*args, **kwargs):
		pr = cProfile.Profile()
		pr.enable()
		retval = fnc(*args, **kwargs)
		pr.disable()
		s = io.StringIO()
		sortby = 'cumulative'
		ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
		ps.print_stats()
		print(s.getvalue())
# 		with open("calc_times.txt", "wb") as file:
# 			file.write(s.getvalue().encode())
# 			file.close()
		return retval
	return inner

class Layer_Conv:
	
	def __init__(self, input_shape, stride=2, filter_size=3, nr_of_filters=6):
		"""
		input_shape has 4 parameters: 
		input_shape [0] and input_shape[1] are the height and width of the images
		input_shape[2] is the depth of the input (RGB images have a depth of 3, B&W images have a depth of 1)
		
		self.output_shape has the same parameters as input_shape. 
		The self.output_shape is calculated with the strides. 
		The depth of the output is nr_of_filters.
		
		self.weights has 4 parameters:
		self.weights[0] has to be the same as the depth of the images
		self.weights[1] and self.weights[2] are the width and height of each kernel
		self.weights[3] represents the number of different filters
		"""
		
		#Initializig hyperparameters
		self.nr_of_filters = nr_of_filters
		self.filter_size = filter_size
		self.stride = stride
		
		#Getting input and output size as attributes
		
		pad = (self.filter_size - 1) // 2
		self.input_shape = input_shape
		self.input_height, self.input_width, self.input_depth = self.input_shape
		
		#Check if the filter fits
		if self.filter_size == self.stride:
			if input_shape[0] % filter_size != 0 or input_shape[1] % filter_size != 0:
				raise ValueError("Voorlopig gebruiken we alleen een filter dat precies in het plaatje past jongeman")
			else:
				self.output_height, self.output_width = int(self.input_height/self.stride), int(self.input_width/self.stride)
		else:
			self.output_height, self.output_width = int( (self.input_height - self.filter_size + 2 * pad)//self.stride + 1), int( (self.input_width - self.filter_size + 2 * pad)//self.stride + 1)

		
		self.output_shape = ( self.output_height, self.output_width, self.nr_of_filters )
# 		print(self.output_shape)
		
		#Creating initial weights
		self.weights = np.random.randn( self.input_depth, self.filter_size, self.filter_size, self.nr_of_filters)
		
		##### Uncomment when creating initial weights for forward_fft #####
# 		self.padded_output_shape = (self.input_shape[0], self.output_shape[1]+2*pad, self.output_shape[2]+2*pad, self.output_shape[3] )
# 		self.weights = np.random.randn( self.input_depth, self.filter_size, self.filter_size, self.nr_of_filters)
		
		##### Creating biases #####
		self.biases = np.zeros(( self.output_height, self.output_width, self.nr_of_filters))
	
	def forward(self, inputs):
		
		#Making these vairables local
		stride = self.stride
		filter_size = self.filter_size
		
		#Padding the input
		pad = (self.filter_size - 1) // 2
		self.padded_inputs = np.pad( inputs, ( (0,0), (pad, pad), (pad, pad), (0,0) ) )
		
		#Initializing the output
		nr_of_datapoints = inputs.shape[0]
		output = np.zeros( ( nr_of_datapoints,) + self.output_shape )
		
		#Convolving the input data with the filters
		for y in range(self.output_height):
			
			#Get the y coordinates in a slice (slices are faster)
			y_slice = slice(y * stride, y * stride + filter_size)
			
			for x in range(self.output_width):
				
				#Get the x coordinates in a slice (slices are faster)
				x_slice = slice(x * stride, x * stride + filter_size)
				
				#The region of interest consists of a region that's the shape of the kernel (self.weights) times the number of input images.
				ROI = self.padded_inputs[:, y_slice, x_slice, :]
				
				output[:, y, x, :] = (ROI * self.weights.T[:,None,...]).sum( axis = (2, 3, 4) ).T
		
		#Make the output an attribute
		self.output = np.add( output, self.biases )
	
	def backward(self, dvalues):
		
		#Making these variables local
		nr_of_datapoints = dvalues.shape[0]
		stride = self.stride
		filter_size = self.filter_size

		#Padding the dvalues
		pad = (filter_size - 1) // 2
		new_dvalues = np.zeros_like(self.padded_inputs, dtype="float")
		
		#Initializing the weight gradients
		dweights = np.zeros_like( self.weights )
		
		#Convolving backwards to alter the weights slightly and computing the new dvalues for every filter slice in the layer.
		for y in range(self.output_height):
			
			#Get the y coordinates in a slice (slices are faster)
			y_slice = slice(y * stride, y * stride + filter_size)
			
			for x in range(self.output_width):
				
				#Get the x coordinates in a slice (slices are faster)
				x_slice = slice(x * stride, x * stride + filter_size)
				
				#The ROI consists of the x,y,z-coordinate of the dvalues of the previous layer. We reshape the ROI to be able to calculate with it.
				ROI = dvalues[:, y, x, :]
				ROI = ROI.reshape((self.nr_of_filters, nr_of_datapoints , 1, 1, 1)) #Dit is not wel een klein beetje hard gecodeert, maar werkt voor de mnist set volgesn mij.
				
				#The dweights (gradient of the kernels) are computed by multiplying the ROI of the dvalues with the inputs of the current layer.
				dweights += (ROI * self.padded_inputs[None, :, y_slice, x_slice, :]).sum(1).T / nr_of_datapoints
				new_dvalues[:, y_slice, x_slice, :] += (self.weights.T[:,None,...] * ROI).sum(0) / self.nr_of_filters
			
		#Remove the padding and give the computed gradient as attributes
		if pad == 1:
			self.dvalues = np.delete( np.delete( np.delete( np.delete( new_dvalues, obj=(-1), axis=(1) ), obj=(0), axis=(1) ), obj=(-1), axis=(2) ), obj=(0), axis=(2) )
		elif pad > 1:
			raise ValueError("KLopt iets niet in padding of je filter_size is groter dan 3 (wordt aan gewerkt")
		else:
			self.dvalues = new_dvalues
		
		self.dweights = dweights
		self.dbiases = dvalues.sum(0) / nr_of_datapoints

class Layer_Pooling:
	
	def __init__(self, input_shape, K_size=2, method="max"):
		
		#Making K_size and method an attribute
		self.K_size = K_size
		self.method = method
		
		if input_shape[0] % K_size != 0 or input_shape[1] % K_size != 0:
			raise ValueError("Voorlopig gebruiken we alleen een pooling filter dat precies in het plaatje past jongeman")
		
		#Calculate the output_shape from the specified input_shape
		self.input_shape = input_shape
		self.output_shape = (int(input_shape[0] / K_size), int(input_shape[1] / K_size), input_shape[2]) 
	
	def forward(self, inputs):
		
		#Get a matrix that represents all the windows that determine the output
		shape = (inputs.shape[0], self.output_shape[0], self.output_shape[1], self.input_shape[2], self.K_size, self.K_size)
		strides = (int(inputs.strides[0]), int(inputs.strides[1] * self.K_size), int(inputs.strides[2] * self.K_size), int(inputs.strides[3]), int(inputs.strides[1]), int(inputs.strides[2]))
		self.all_windows = np.lib.stride_tricks.as_strided(inputs, shape=shape, strides=strides)
		
		if self.method == "max":
			
			#Get the maximum of each window
			output = np.max(self.all_windows, axis=(4,5), keepdims=True) #Neemt 80% - 90% vd tijd v pooling in beslag...
			
			#Make a matrix where everytime we encounter a max in a window, it returns a 1 in the matrix, else it returns a 0.
			self.indices = np.where( (self.all_windows == output), 1, 0)
			
			#Reshape the indices to something we can work with (input_shape) in backwards
			shaped_indices_shape = ( (inputs.shape[0],) + self.input_shape )
			shaped_indices_strides = (int(inputs.strides[0]/self.K_size), int(inputs.strides[1]/self.K_size), int(inputs.strides[2]/self.K_size), int(inputs.strides[3]/self.K_size))
			self.shaped_indices_maxima = np.lib.stride_tricks.as_strided(self.indices, shape=shaped_indices_shape, strides=shaped_indices_strides)
			
			#Return the output as an attribute
			self.output = output[...,0,0]
		
		elif self.method == "mean":
			
			#Compute the mean of every pooling window
			self.output = np.mean(self.all_windows, axis=(4,5))
	
	def backward(self, dvalues):
		
		#Initiate new dvalues
		new_dvalues = np.empty( (dvalues.shape[0],) + self.input_shape )
		
		#Fill the new dvalues matrix with the old dvalues in a weaving pattern
		for y in range(self.K_size):
			for x in range(self.K_size):
				new_dvalues[:, y::self.K_size, x::self.K_size, :] = dvalues
		
		if self.method == "max":
		
			#Multiply the weaved matrix with the indices. The max value of a window has a dvalue, the rest is returned as 0.
			self.dvalues = new_dvalues * self.shaped_indices_maxima
		
		elif self.method == "mean":
			
			#The new dvalues are just the old ones times 1 / (pooling_window.size)
			self.dvalues = new_dvalues * ( 1 / (self.K_size**2))

# # # # Create a class for the dense layers. A dense layer is just a layer of nodes that is connected to every node in the previous and next layer. # # # #
class Layer_Dense:
	
	def __init__(self, input_shape, nr_of_neurons):
		
		#Initialize weights. Will give an array of inputs x neurons array
		#If nr of input features (dimensions) == 2 and we want a layer with 64 neurons, that means our weights have 2 rows and 64 columns.
		#We initialize small weight because otherwise the values of arrays will explode down the line
		nr_of_features_per_input = input_shape[0]
		self.weights = 0.01*np.random.randn(nr_of_features_per_input, nr_of_neurons)
		
		#Initialize all biases as an array of (1 by x) with values of zero (1 row, X columns)
		self.biases = np.zeros(shape=(nr_of_neurons))
		self.output_shape = np.empty( nr_of_neurons ).shape
	
	# Forward pass through dense layer
	def forward(self, inputs):
		
		#Get the inputs
		self.inputs = inputs
		
		#This  gets the output of each connection between neurons.
		#300 input points and 64 neurons in the first layer means we would run this function 300 times if we ran it seperately for each datapoint.
		#We actually do it in 1 go. Represented as a 300 by 64 array.
		self.output = np.dot(inputs, self.weights) + self.biases
	
	# Backwards pass through dense layer. It is based on using the chain rule to get the gradients of all the layers. The gradients are stores in dvalues.
	def backward(self, dvalues):
		
		#The dvalues for the weights is just the amount that the weight was off, multiplied by the input value to that weight to get a sense of how much we need to correct the output of the neuron
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0)
		
		#Gradient on values
		self.dvalues = np.dot(dvalues, self.weights.T)

class Flatten:
	
	def __init__(self, input_shape):
		
		#Make the input size an attribute
		self.input_shape = input_shape
		
		#Compute the output size. It is just the number of elements in every datapoint in the batch. If we have 30 datapints in a batch with dimensions 10x10x4, then our output size is (30, 400)
		self.output_shape = np.empty( int(np.prod(input_shape) ) ).shape
	
	def forward(self, inputs):
		
		#Make the inputs an attribute
		self.inputs = inputs
		
		nr_of_datapoints = inputs.shape[0]
		
		#Flatten the input.
		self.output = np.reshape(inputs, ( ( (nr_of_datapoints,) + self.output_shape) ) )
	
	def backward(self, dvalues):
		
		nr_of_datapoints = dvalues.shape[0]
		
		#Reshape the flat outputs back the their original shape to be used in the convolulion layer.
		self.dvalues = np.reshape(dvalues, ( ( (nr_of_datapoints,) + self.input_shape) ) )

class Activation:
	
	def __init__(self, input_shape, method, alpha=0.2, batch_norm=False):
		
		"""
		An activation function has the same input_shape as output_shape.
		
		Which activation function you want can be specified with the method argument.
		
		Alpha is an argument for leakyrelu
		
		Batch_normalization always takes place after an activation function,
		so here can be specified if we want batch_norm or not.
		"""
		
		self.input_shape = input_shape
		self.output_shape = input_shape
		self.method = method
		self.batch_norm = batch_norm
		self.alpha = alpha #For leakyrelu
		
		#Check if we're using batch normalization
		if self.batch_norm == True:
			self.weights = np.random.standard_normal(self.output_shape) #For batch_norm
			self.biases = np.zeros(self.output_shape) #For batch_norm
			self.eps = 1e-5 #For stabilizing batch_norm
	
	def forward(self, inputs):
		
		#Get the inputs as an attribute
		self.inputs = inputs
		
		if self.method == "relu":
			
			#If the value of the connection is less then 0, the output is 0.
			self.output = np.maximum(0, inputs)
		
		elif self.method == "leakyrelu":
			
			#Check if alpha is given
			if self.alpha:
			
				#Inputs get scaled with some factor alpha
				self.output = np.maximum(self.alpha * inputs, inputs)
		
		elif self.method == "softmax":
			
			#Scale all the connection outputs between 1 and 0 with an exponent function
			exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #We do '-' the maximum of the whole row, to keep all the inputs below 0, so the exponent does not go off to infinity.
			probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		
			#Give the probablilities as an output
			self.output = probabilities
		
		elif self.method == "sigmoid":
			
			self.output = 1 / (1 + np.exp(-inputs))
		
		elif self.method == "tanh":
			
			self.output = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
		
		else:
			raise ValueError("Method " + str(self.method) + " does not exist.")
		
		if self.batch_norm and self.method != "softmax":
			
			#Calculate the mean and standard deviation
			self.batch_mean = np.mean(inputs, axis=0)
			self.batch_standard_deviation = np.sqrt( np.var( inputs, axis=0 )  + self.eps)
			
			#Map the normalization function to the array and convert the outcome to an array
			self.output = (inputs - self.batch_mean) / self.batch_standard_deviation
			self.output = (self.output * self.weights) + self.biases
	
	def backward(self, dvalues):
		
		nr_of_datapoints = dvalues.shape[0]
		
		#Calculate the gradients the the weights and biases
		if self.batch_norm and self.method != "softmax":
			
			#Calculate the gradients of the weights and biases
			self.dbiases = dvalues.sum(axis=0)
			self.dweights = (dvalues * self.inputs).sum(axis=0)
			
			#Calculate the gradient of the new dvalues
			d_standardized_output = dvalues * self.weights
			
			new_dvalues = (1 / (nr_of_datapoints * self.batch_standard_deviation) ) * ( (nr_of_datapoints * d_standardized_output) - d_standardized_output.sum(axis=0) - self.output * (d_standardized_output * self.output).sum(axis=0) )  
			
			dvalues = new_dvalues
		
		if self.method == "relu" or self.method == "leakyrelu":
			
			#The slope of the relu function is 0 for all values smaller than 0. Otherwise the slope is just the slope of the previous backward function.
			self.dvalues = dvalues.copy()
			self.dvalues[self.inputs < 0] = 0
		
		elif self.method == "sigmoid" or self.method == "tanh":
			
			self.dvalues = dvalues * (1 - dvalues)
		
		elif self.method == "softmax":
			
			#We combine the backwards pass with the loss backwards pass so we only have to copy here. This is faster.
			self.dvalues = dvalues.copy()
		
		else:
			raise ValueError("Method " + str(self.method) + " does not exist.")

# # # # Create a class for calculating the loss of the function. This is what we will be judging the performance of the NN with # # # #
class Loss_CategoricalCrossentropy():
	
	def __init__(self):
		pass
	
	#This function is very, very fast
	def forward(self, y_pred, y_true):
		
		if len(y_true.shape) > 1:
			raise ValueError("A single class can only have 1 dimension")
		
		#Number of samples in batch
		samples = y_pred.shape[0]
		
		#This gives back the value in y_pred that we should maximize
		y_pred = y_pred[range(samples), y_true]
		
		#Paar verschillende manieren om loss te berekenen. Lijken even goed te werken.
# 		negative_log_likeliyhoods = -np.log(y_pred)
		squared_loss = np.sum(1/2 * (1 - y_pred)**2, keepdims=True)
		
		#The mean loss
		self.data_loss = np.mean(squared_loss) / samples
		
		return self.data_loss
	
	def backward(self, dvalues, y_true):
		
		#Get the number of samples we need to compute the gradient for
		samples = dvalues.shape[0]
		
		#Copy so we can safely modify
		self.dvalues = dvalues.copy()
		
		#We compute the difference in expected output versus the actual output.
		#So we take the neuron that should have lit up as 1 and subtract the expected value 1.
		### Volgens mij wordt de afgeleide hier negatief als het getal omhoog moet en andersom.
		#Vandaar dat we in the optimizer in update_params steeds die min zien. ###
		self.dvalues[range(samples), y_true] -= 1
		
		#We divide the difference by the number of samples to get the slope
		#Because all weighted inputs together should add up to 1.
		#We calculate the amount that each weighted sum was off by dividng the total amount it was of by the number of inputs we had.
		self.dvalues = ( self.dvalues )
		
		return self.dvalues

class Optimizer:
	
	#Initialize the optimizer so we can access all these parameters throughout the whole function.
	def __init__(self, method="adam", learning_rate=0.002, decay=0, epsilon=1e-7, momentum=0.9, rho=0.999):
		
		#Check if all arguments are valid
		if decay < 0:
			raise ValueError("Decay can't be smaller than 0.")
		if momentum > 1 or momentum < 0:
			raise ValueError("Momentum must be between 1 and 0.")
		if rho > 1 or rho < 0:
			raise ValueError("Rho must be between 1 and 0.")
		
		self.method = method
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.momentum = momentum
		self.rho = rho
	
	# Call once before any parameter updates
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.current_learning_rate * (1. / (1. + self.decay * self.iterations))
	
	# Update weigths and biases
	def update_params(self, layer):
		
		if self.method == "doegewoonnormaal":
			layer.weights -= (self.learning_rate * layer.dweights)
			layer.biases -= (self.learning_rate * layer.dbiases)
		
		#########################################################################################
		if self.method == "sgd":
			#If we use momentum
			if self.momentum:
				#If layer does not contain momentum arrays, create ones filled with zeros with the shape of the weights and biases arrays
				if not hasattr(layer, 'weight_momentums'):
					layer.weight_momentums = np.zeros_like(layer.weights)
					layer.bias_momentums = np.zeros_like(layer.biases)
				"""
				We take the previous weight_update step and multiply that by our momentum. This give us the adjusted momentum.
				Then we take our current_learning_rate, which depends on what epoch we are in
				and multiply that by the slope of every weight. Let's call that the adjusted dweight
				We subtract the adjusted momentum by the adjusted dweight. This give us the property
				Then a weight can't get stuck in a small local mimum after it just went down a steep slope.
				"""
				weight_updates = ( (self.momentum * layer.weight_momentums) - (self.current_learning_rate * layer.dweights) )
				layer.weight_momentums = weight_updates
				
				#For biases it works the same as for the weights
				bias_updates = ( (self.momentum * layer.bias_momentums) - (self.current_learning_rate * layer.dbiases) )
				layer.bias_momentums = bias_updates
				
				#Update the weights of the layer incrementally
				layer.weights += weight_updates
				layer.biases += bias_updates
			
			#If we don't use momentum, just update the weights with the learning rate
			else:
				layer.weights -= (self.learning_rate * layer.dweights)
				layer.biases -= (self.learning_rate * layer.dbiases)
		
		########################################################################################
		if self.method == "adagrad":
			#If layer does not contain cache arrays, create ones filled with zeros
			if not hasattr(layer, 'weight_cache'):
				layer.weight_cache = np.zeros_like(layer.weights)
				layer.bias_cache = np.zeros_like(layer.biases)
			
			#Update cache with squared current gradients
			layer.weight_cache += layer.dweights ** 2
			layer.bias_cache += layer.dbiases ** 2
			"""
			We keep track of how far we have drifted of the starting weight. We do this by adding the 
			square of each dweight in cache. This way, the dweight of every step gets smaller, the more
			steps we take. Larger steps have more influence on the scaling factor. We also add epsilon
			to the square root of the weight_cache, because this prevents the dweights to become too large 
			when the cache is still small.
			When the cache is lower than 1, the dweights get scaled up, and after a certain number of steps
			(which differs for every weight individualy) the dweight will get scaled down.
			Everything goes the same for biases.
			"""
			#Normal SGD parameter update + normalization with square rooted cache
			layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
			layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
		
		####################################################################################
		if self.method == "rmsprop":
			#If layer does not contain cache arrays, create ones filled with zeros
			if not hasattr(layer, 'weight_cache'):
				layer.weight_cache = np.zeros_like(layer.weights)
				layer.bias_cache = np.zeros_like(layer.biases)
			
			# Update cache with squared current gradients
			layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
			layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
			"""
			This is exactly the same as adagrad, but the cache is computed differently.
			We use a adjustment fator rho to alter the way the cache grows. This function often fits
			a lot of data better.
			"""
			#Normal SGD parameter update + normalization with square rooted cache
			layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
			layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
		
		#########################################################################################
		if self.method == "adam":
			# If layer does not contain cache arrays, create them filled with zeros
			if not hasattr(layer, 'weight_cache'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				layer.weight_cache = np.zeros_like(layer.weights)
				layer.bias_momentums = np.zeros_like(layer.biases)
				layer.bias_cache = np.zeros_like(layer.biases)
			"""
			Adam is essentially a combination of RPMProp with momentum. This function is even better for 
			a lot of data.
			"""
			# Update momentum  with current gradients
			layer.weight_momentums = self.momentum * layer.weight_momentums + (1 - self.momentum) * layer.dweights
			layer.bias_momentums = self.momentum * layer.bias_momentums + (1 - self.momentum) * layer.dbiases
			
			# Get corrected momentum. self.iteration is 0 at first pass and we need to start with 1 here
			weight_momentums_corrected = layer.weight_momentums / (1 - self.momentum ** (self.iterations + 1))
			bias_momentums_corrected = layer.bias_momentums / (1 - self.momentum ** (self.iterations + 1))
			
			# Update cache with squared current gradients
			layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
			layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
			
			# Get corrected cachebias
			weight_cache_corrected = layer.weight_cache / (1 - self.rho ** (self.iterations + 1))
			bias_cache_corrected = layer.bias_cache / (1 - self.rho ** (self.iterations + 1))
			
			# Normal SGD parameter update + normalization with square rooted cache
			layer.weights += -self.current_learning_rate * weight_momentums_corrected /(np.sqrt(weight_cache_corrected) + self.epsilon)
			layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
		
		else:
			raise ValueError("Method " + str(self.method) + " does not exist.")
	
	# Call once after any parameter updates
	def post_update_params(self):
		self.iterations += 1