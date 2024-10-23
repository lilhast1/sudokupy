#!/usr/bin/env python3
from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
 
# load an image and predict the class
def run_example():
	# load the image
	img = load_image('sample.png')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(digit)

def classify(image):
	model = load_model('final_model.h5')
	predict_value = model.predict(image)
	digit = argmax(predict_value)
	if predict_value[0, digit] < 0.5:
		digit = 0
	print(f'prediction vector: {predict_value}, digit: {digit}')
	return digit, predict_value 

#np.argsort(np.max(predict_value, axis=0))[-2]

if __name__=='__main__':
	run_example()