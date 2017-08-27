import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import data_reader
import hog
from sklearn.model_selection import train_test_split
import parameters
import features
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier

def train():
	cars, notcars = data_reader.get_data()

	color_space = parameters.color_space
	spatial_size = parameters.spatial_size
	hist_bins = parameters.hist_bins
	orient = parameters.orient
	pix_per_cell = parameters.pix_per_cell
	cell_per_block = parameters.cell_per_block
	hog_channel = parameters.hog_channel
	spatial_size = parameters.spatial_size
	hist_bins = parameters.hist_bins
	hist_range = parameters.hist_range
	spatial_feat = parameters.spatial_feat
	hist_feat = parameters.hist_feat
	hog_feat = parameters.hog_feat

	t=time.time()

	print ("Number of images is: ", len(cars))

	n = len(cars)
	
	car_features = features.extract_features(cars[:n], color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

	notcar_features = features.extract_features(notcars[:n], color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat) 

	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to extract HOG features...')
	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
		scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using:',orient,'orientations',pix_per_cell,
		'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))
	# Use a linear SVC 
	#svc = SVC(kernel='sigmoid')
	# We are using a Multi-layer Perceptron classifier because it worked better than SVM
	svc = MLPClassifier(solver='lbfgs')
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	
	return svc, X_scaler

if __name__ == "__main__":
	
	train()

	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	# Check the prediction time for a single sample
	t=time.time()
	n_predict = 10
	print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
	print('For these',n_predict, 'labels: ', y_test[0:n_predict])
	t2 = time.time()
	print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
