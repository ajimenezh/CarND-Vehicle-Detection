import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import data_reader
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features

def extract_features(imgs, orient, pix_per_cell, cell_per_block):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        hog_features = get_hog_features(gray, orient, 
		                    pix_per_cell, cell_per_block)

        features.append(hog_features)

    return features


if __name__ == "__main__":

	cars, notcars = data_reader.get_data()

	# Generate a random index to look at a car image
	ind = 100
	# Read in the image
	image = mpimg.imread(cars[ind])
	#gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# Define HOG parameters
	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	# Call our function with vis=True to see an image output
	features, hog_image = get_hog_features(image[:,:,2], orient, 
		                    pix_per_cell, cell_per_block, 
		                    vis=True, feature_vec=False)


	# Plot the examples
	fig = plt.figure()
	plt.subplot(121)
	plt.imshow(image, cmap='gray')
	plt.title('Example Car Image')
	plt.subplot(122)
	plt.imshow(hog_image, cmap='gray')
	plt.title('HOG Visualization')

	#plt.show()

	plt.savefig("output_images/hog_example1.jpg")
	
