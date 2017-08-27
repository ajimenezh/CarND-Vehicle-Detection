import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
import data_reader
import hog

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for file in imgs:
		# Read in each one by one
		image = mpimg.imread(file)

		features.append(extract_features_img(image, color_space, spatial_size, hist_bins, 
                    hist_range, orient, 
                    pix_per_cell, cell_per_block, 
                    hog_channel, spatial_feat, 
                    hist_feat, hog_feat))
	
	# Return list of feature vectors
	return features

def extract_features_img(img, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
	# Create a list to append feature vectors to
	features = []

	# apply color conversion if other than 'RGB'
	if color_space != 'RGB':
		if color_space == 'HSV':
		    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
		    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
		    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
		    feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
	else: feature_image = np.copy(img)      

	if spatial_feat:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
	else:
		spatial_features = []

	if hist_feat:
		hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
	else:
		hist_features = []

	if hog_feat:
		hog_features = []
		if hog_channel == 'GRAY':
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			hog_features.append(hog.get_hog_features(gray, orient, pix_per_cell, cell_per_block, 
													vis=False, feature_vec=True))
			hog_features = np.ravel(hog_features)
			print (len(hog_features))
		else:
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(hog.get_hog_features(feature_image[:,:,channel], 
								        orient, pix_per_cell, cell_per_block, 
								        vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)        
			else:
				hog_features = hog.get_hog_features(feature_image[:,:,hog_channel], orient, 
							pix_per_cell, cell_per_block, vis=False, feature_vec=True)
	else:
		hog_features = []

	# Append the new feature vector to the features list
	features_img = (np.concatenate((spatial_features, hist_features, hog_features)))
	
	# Return list of feature vectors
	return features_img

if __name__ == "__main__":

	cars, notcars = data_reader.get_data()

	car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256))
	notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
		                    hist_bins=32, hist_range=(0, 256))

	if len(car_features) > 0:

		# Create an array stack of feature vectors
		X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
		# Fit a per-column scaler
		X_scaler = StandardScaler().fit(X)
		# Apply the scaler to X
		scaled_X = X_scaler.transform(X)
		car_ind = np.random.randint(0, len(cars))
		# Plot an example of raw and scaled features
		fig = plt.figure(figsize=(12,4))
		plt.subplot(131)
		plt.imshow(mpimg.imread(cars[car_ind]))
		plt.title('Original Image')
		plt.subplot(132)
		plt.plot(X[car_ind])
		plt.title('Raw Features')
		plt.subplot(133)
		plt.plot(scaled_X[car_ind])
		plt.title('Normalized Features')
		fig.tight_layout()

		plt.show()
	else: 
		print('Your function only returns empty feature vectors...')
