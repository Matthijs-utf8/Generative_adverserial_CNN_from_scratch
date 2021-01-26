# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:16:15 2020
@author: Sietse Schröder and Matthijs Schrage
"""

import sys
import os
sys.path.insert(0, os.getcwd())

#Import modules
import Preprocessing
import Model

#Import other necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.datasets import mnist
import time

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Genormaliseerde data werkt veel beter
DP = Preprocessing.DataPreprocessing()
X_train = DP.scale(X_train)
X_test = DP.scale(X_test)

#We moeten voordat we de data in het model stoppen nog even deze stap doen
X_train = np.reshape( X_train, ( X_train.shape + (1,) ) )
X_test = np.reshape( X_test, ( X_test.shape + (1,) ) )

def main():
	
	model = Model.Sequential( X_train.shape[1:], batch_norm=True )
	model.add_conv(stride=2, filter_size=3, nr_of_filters=3, activation="leakyrelu", pooling=False, K_size=2, method="max")
	model.add_conv(stride=2, filter_size=3, nr_of_filters=6, activation="leakyrelu", pooling=False, K_size=2, method="max")
	model.add_dense(size=128, activation="leakyrelu")
	model.add_dense(size=10, activation="softmax")
	model.adjust_optimizer(method="adam", learning_rate=0.3, decay=1e-5, epsilon=1e-7, momentum=0.9, rho=0.95)
	model.fit( X_train, y_train, epochs=3, batch_size=128, shuffle=True, train=True)
# 	model.fit( X_train, y_train, epochs=2, batch_size=512, shuffle=True, train=True)
# 	model.fit( X_train, y_train, epochs=2, batch_size=2048, shuffle=True, train=True)
	model.test(X_test, y_test)
	model.save()


if __name__ == "__main__":

	main()


	# model = Model.Sequential.load(flag='best')
	#
	# model.add_inputs(X_train)
	#
	# model.fit(y_train, epochs=1, optimizer='adam')
	#
	# model.adjust_optimizer(learning_rate=0.01, decay=1e-9, epsilon=1e-7, momentum=0.9, rho=0.95)
	#
	# model.test(X_test, y_test)
	#
	# model.save()
