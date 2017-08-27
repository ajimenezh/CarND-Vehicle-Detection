##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/hog_example1.jpg
[image3]: ./output_images/hog_example2.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/example1.jpg
[image6]: ./output_images/heatmap_1.jpg
[image7]: ./output_images/heatmap_2.jpg
[image8]: ./output_images/labels_1.jpg
[image9]: ./output_images/labels_2.jpg
[image10]: ./output_images/example2.jpg
[image11]: ./output_images/example3.jpg
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images. I did this in the file `data_reader.py` in the function `get_data()`, that recursively gets all the images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Also, I experimented with different color spaces, which I later found that that had more relevance in obtaining a good solution.

Here is an example using the blue color in `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, and another using the gray scale to see the differences. Although the difference seems small, later in the classification, we will see that using the correct color space is essential:


![alt text][image2]

![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and in the end I decided to use as parameters `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. Also as the color space, at first I used the gray scale, which performed relatively well, but in the end I used all the channels in the `HSV` color space, because it worked the best.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained the classifier in the file `classifier.py`. At first, I used a linear SVC, which later I substituted with a sigmoid kernel. With this classifier I trained various models with different HOG features (color spaces, parameters and color channels), and also I have mixed the HOF features with an histogram of the images to improve the results.

After this, I experimented with other classifiers, and decide to use the `MLPClassifier` (Multi-layer Perceptron classifier)  which uses neural networks. With this, I trained the model only with HOG features with all the channels in the `HSV` color space.

I trained the model with the HOG features, and 20% of tests.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The function to find the windows is `slide_window` in `program.py`, but I ended up using an alternative algorithm to only perform the HOG step once per size of the window, this can be found in the function `find_cars`.

First, I noticed that I only needed to use a subset of the image, without the top and bottom which corresponds to the sky and the car respectively. Then, I divided the image in overlapping windows, with an overlapping coefficient of 0.75. This is because, although the algorithm runs slower because there are more windows, I can detect false positives more easily and obtain more accurate solutions.
I used windows of three sizes: 64x64, 96x96, 128x128, which captured well the cars in every position, when they are nearer and when they are further away.

Here is a image with every window.

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In summary, I've only used HOG features in `HSV` color space, using all the channels, and a Multi-layer Perceptron classifier. I searched in windows of three sizes (64, 96, 128) with an overlapping of 75%. Here there is an example of the pipeline working in an image.

![alt text][image5]

To optimize the algorithm, I crop out parts of the image that are not used, and I only calculate the HOG features once.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To prevent false positives, I keep a heatmap with the detections of the last 15 frames (if a windows is classified as car, I sum 1 to the window), and I filter from the heatmap all values less than 12 (there is a special case with the first frames, where the limit is lower).

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

![alt text][image7]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image8]

![alt text][image9]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image10]

![alt text][image11]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

