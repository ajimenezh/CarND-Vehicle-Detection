import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import classifier
import pickle
import hog
import parameters
import features
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

heat = np.zeros((1280, 720)).astype(np.int32)
windows_list = []

def add_heat(bbox_list):
	global heat
	for box in bbox_list:
		print (box)
		heat[box[0][0]:box[1][0], box[0][1]:box[1][1]] += 1

def remove_heat(bbox_list):
	global heat
	for box in bbox_list:
		heat[box[0][0]:box[1][0], box[0][1]:box[1][1]] -= 1

def get_hot_windows():
	global heat
	heatmap = np.copy(heat)
	heatmap[heatmap < min(2*len(windows_list), 12) + 1] = 0
	labels = label(heatmap)

	hot_boxes = []

	print (labels[1])

	for elem in range(1, labels[1]+1):

		nonzero = (labels[0] == elem).nonzero()

		nonzeroy = np.array(nonzero[1])
		nonzerox = np.array(nonzero[0])

		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

		hot_boxes.append(bbox)
	
	print (hot_boxes)
	return hot_boxes

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

	#1) Create an empty list to receive positive detection windows
	on_windows = []
	#2) Iterate over all windows in the list
	for window in windows:
		#3) Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      

		img_features = features.extract_features_img(test_img, color_space, spatial_size, hist_bins, hist_range, orient, pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(img_features).reshape(1, -1))
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)
	#8) Return windows for positive detections
	return on_windows

def find_cars(img, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

	on_windows = []    
	
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

	img_tosearch = feature_image[ystart:ystop,:,:]
	if scale != 1:
		imshape = img_tosearch.shape
		img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

	ch1 = img_tosearch[:,:,0]
	ch2 = img_tosearch[:,:,1]
	ch3 = img_tosearch[:,:,2]

	# Define blocks and steps as above
	nxblocks = (img_tosearch.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (img_tosearch.shape[0] // pix_per_cell) - cell_per_block + 1 
	nfeat_per_block = orient*cell_per_block**2

	# 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step

	# Compute individual channel HOG features for the entire image
	hog1 = hog.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = hog.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = hog.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)


	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# Scale features and make a prediction
			test_features = X_scaler.transform(hog_features).reshape(1, -1)    
			#test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
			test_prediction = svc.predict(test_features)

			if test_prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
		        
	return on_windows

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def solve_video(path):
	white_output = 'output_videos/' + path.split('/')[-1]
	## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
	## To do so add .subclip(start_second,end_second) to the end of the line below
	## Where start_second and end_second are integer values representing the start and end of the subclip
	## You may also uncomment the following line for a subclip of the first 5 seconds
	##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
	clip1 = VideoFileClip(path)
	white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
	white_clip.write_videofile(white_output, audio=False)

idx = 0
def process_image(img):

	global idx
	cv2.imwrite("tmp/test_" + str(idx) + ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
	idx += 1

	image = mpimg.imread("tmp/test_" + str(idx-1) + ".jpg")
	#image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	
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
	y_start_stop = [400, 650] # Min and max in y to search in slide_window()

	windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
		                xy_window=(32, 32), xy_overlap=(0.5, 0.5))

	windows4 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
		                xy_window=(64, 64), xy_overlap=(0.75, 0.75))

	windows1 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
		                xy_window=(96, 96), xy_overlap=(0.75, 0.75))

	windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
		                xy_window=(128, 128), xy_overlap=(0.75, 0.75))

	windows = windows1 + windows2 + windows4

	#hot_windows = search_windows_hog_opt(image, windows, svc, X_scaler, color_space=color_space, 
#		                    spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
#		                    orient=orient, pix_per_cell=pix_per_cell, 
#		                    cell_per_block=cell_per_block, 
#		                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
#		                    hist_feat=hist_feat, hog_feat=hog_feat, y_start_stop=y_start_stop)  

	windows1 = find_cars(image, color_space, 400, 650, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

	windows2 = find_cars(image, color_space, 400, 650, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

	windows3 = find_cars(image, color_space, 400, 650, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

	hot_windows = windows1 + windows2 + windows3                     
	
	print (hot_windows)
	global windows_list
	windows_list.append(hot_windows)

	add_heat(hot_windows)

	if len(windows_list) > 15:
		remove_heat(windows_list[0])
		windows_list = windows_list[1:]

	hot_windows = get_hot_windows()

	window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)      
          
	return window_img

init = False

if not init:
	try:
		with open('svc.pkl', 'rb') as f:
		    svc = pickle.load(f)
		with open('xscaler.pkl', 'rb') as f:
			X_scaler = pickle.load(f)
	except (OSError, IOError) as e:
		print ("Saved data not found")
		init = True

if init:
    svc, X_scaler = classifier.train()

    with open('svc.pkl', 'wb') as f:
        pickle.dump(svc, f)
    with open('xscaler.pkl', 'wb') as f:
        pickle.dump(X_scaler, f)

TEST = True

if TEST:
	#image = mpimg.imread('test_images/test6.jpg')

	images = ['tmp/test_750.jpg', 'tmp/test_900.jpg', 'tmp/test_1050.jpg']
	#images = ['tmp/test_256.jpg', 'tmp/test_450.jpg', 'tmp/test_580.jpg']
	
	k = 1
	fig = plt.figure()

	for i in images:

		image = mpimg.imread(i)
		#image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		draw_image = np.copy(image)

		# Uncomment the following line if you extracted training
		# data from .png images (scaled 0 to 1 by mpimg) and the
		# image you are searching is a .jpg (scaled 0 to 255)
		#image = image.astype(np.float32)/255

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
		y_start_stop = [400, 650] # Min and max in y to search in slide_window()

		windows3 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
				            xy_window=(32, 32), xy_overlap=(0.5, 0.5))

		windows4 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
				            xy_window=(64, 64), xy_overlap=(0.75, 0.75))

		windows1 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
				            xy_window=(96, 96), xy_overlap=(0.75, 0.75))

		windows2 = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
				            xy_window=(128, 128), xy_overlap=(0.75, 0.75))

		windows = windows1 + windows2 + windows4

		#hot_windows = search_windows_hog_opt(image, windows, svc, X_scaler, color_space=color_space, 
	#		                    spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
	#		                    orient=orient, pix_per_cell=pix_per_cell, 
	#		                    cell_per_block=cell_per_block, 
	#		                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
	#		                    hist_feat=hist_feat, hog_feat=hog_feat, y_start_stop=y_start_stop)  

		windows1 = find_cars(image, color_space, 400, 650, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

		windows2 = find_cars(image, color_space, 400, 650, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

		windows3 = find_cars(image, color_space, 400, 650, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

		windows = windows1 + windows2 + windows3

		add_heat(windows)

		hot_windows = get_hot_windows()                     

		window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

		#plt.imshow(window_img)
		#plt.show()
		#plt.savefig("output_images/example1.jpg")
	
		#heatmap = np.copy(np.transpose(heat))
		#heatmap[heatmap < 2] = 0
		#labels = label(heatmap)
	
		plt.subplot(3, 2, k)
		plt.imshow(draw_image)
		plt.subplot(3, 2, k+1)
		plt.imshow(window_img)
		fig.tight_layout()

		remove_heat(windows)

		k += 2

	fig.tight_layout()
	#plt.show()
	plt.savefig("output_images/examples3.jpg")

else:
	solve_video("./project_video.mp4")

