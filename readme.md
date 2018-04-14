# **Traffic Sign Recognition**

---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_visualization.png "Visualization"
[image2]: ./examples/classOccurance.png "histogram"
[image3]: ./examples/cnnArchitecture.jpg "cnnArchitecture"
[image4]: ./examples/im1.PNG "Traffic Sign 1"
[image5]: ./examples/im2.PNG "Traffic Sign 2"
[image6]: ./examples/im3.PNG "Traffic Sign 3"
[image7]: ./examples/im4.PNG "Traffic Sign 4"
[image8]: ./examples/im5.PNG "Traffic Sign 5"
[image9]: ./examples/im1results.PNG "Traffic Sign 1"
[image10]: ./examples/im2results.PNG "Traffic Sign 2"
[image11]: ./examples/im3results.PNG "Traffic Sign 3"
[image12]: ./examples/im4results.PNG "Traffic Sign 4"
[image13]: ./examples/im5results.PNG "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jkpld/SDC_ND_2/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Summary of data set

The code for this step is contained in the 3rd code cell of the IPython notebook.  

I used numpy to compute the summary statistics:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th, 5th, and 6th cells of the IPython notebook.  

I show the first and last image of each class in the training set.

![example images][image1]

I also show a histogram of the class occurrence in the training set.

![example images][image2]


### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques?

The code for this step is in the 7th-11th cells in the IPython notebook.

I first convert the images grayscale, where my grayscale image is the lightness (L) channel of the images in the Lab color space. I then scale the grayscale images to be between 0 and 100 (at 0.5% and 99.5% respectively).
I originally tried using the full color images after normalizing them. (I normalized using the sqrt of the L1 norm of each pixel and then scaled the images so that the minimum and maximum color values were 0 and 1). This produced nice images, but did not perform as well. (This code is not included in the notebook.)

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for this step is (also) in the 7th-11th cells in the IPython notebook.

The file of data I downloaded already included, training, validation, and test data sets; therefore, I did not break up the training data further to form validation set.

Given the large class imbalance of the training set, I produced a new training set with equal number of examples for each class. (I used 2000 images per class for a total of 86000 training images). When new images needed to be created, I applied a random rotation in the range of +-15 degrees, and random scale factor in the range of 0.8 to 1.2, and added random noise with a random amplitude between 0 and 10. (This is performed with the function dither().)

As an example, if a class only had 200 training images, then for each of these images I would create 9 new images by applying the dither function.

If the class had more than 2000 images to begin with, then I simply took a random 2000 images from the class.

The code for creating the new data set is the function |normalize_train()| in the 7th cell.

As a note, the class imbalance can lead to errors when training and when evaluating because it is easy to predict majority classes; this is why it was fixed.

As a final step, I shuffled the training data.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 12th cell of the ipython notebook.

My final model is depicted in the image below.
![model][image3]

I first used a convolution layer to extract base features. I then split the data into two branches: one branch contains a slice of the center 16x16x6 pixels, the second branch is formed by an avg. pooling with 2x2 and stride 2 to produce another 16x16x6 layer. I use these two branches to obtain higher resolution information of the center of the image (for detecting the symbol on the sign) while lower resolution of the hole image (for detecting the shape of the sign). I thought this could be a good way to reduce parameters, and as shown below, it works quite well.

To both of the branches I apply 5x5x15 conv2d layers, and then I flatten the result out and concatenate them together to form a 1d layer with 1080 elements.

At this point I use a dropout layer followed by a fully connected layer with 256 elements. Again I use a dropout layer and then a fully connected layer to the final layer with 43 elements (which is the number of classes). The dropouts were set to 0.5 for training.

Everywhere I use linear layers (W*x + b) followed by sigmoid activations. Sigmoids did better than the ReLU layers in my initial testing.


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 13th and 14th cell of the ipython notebook.

I used used mean cross entropy with the Adam optimizer. My hyperparameters were

- learning rate = 0.002
- batch size = 256
- epochs = 15

The batch size was mostly chosen for speed (it just took a few minutes to train). The learning rate was increased from the default 0.001 to further speed the training up. I used 15 epochs because after 15 the results did not improve.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is in the 14th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.982
* test set accuracy of 0.968

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

In the beginning I simply use the LeNet architecture; with my pre-processing, I was able to get just over 0.93 validation accuracy. I then switched the ReLU layers to sigmoid layers and was able to get ~0.96-0.97 validation accuracy. I wanted to try making an architecture though.
The first architecture I tried was essentially the final architecture I showed in the image above, but without the first convolution (I immediately created the two branches.) However, I soon modified this to add the initial convolution layer as the first layer should identify the low level features. I then realized the model was overfitting the training set, so I added in the dropouts.
The result of this network was a validation accuracy quite close to the values listed in the table of the Paper linked to the ipython notebook, so I stopped. I do not know if this is a well known architecture or not, so I cannot comment on that, but I explained why I used this architecture above.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web:

![im1][image4] ![im2][image5] ![im3][image6]
![im4][image7] ![im5][image8]

The third and fourth signs are the most interesting. The third sign does not have a counterpart in the image set (which only goes up to speeds of 120). The fourth image is seen on an angle has as quite different lighting.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook, and the results are displayed in the 19th through 23rd cells. Below I show each of the test images along with a bar plot of the first 5 predictions (in log scale) and I include an example image of the top 5 results from the training set.

![im1r][image9]
![im2r][image10]
![im3r][image11]
![im4r][image12]
![im5r][image13]

Image 1, 2, and 5, have very large probabilities of being the correct class.

Image 3 (speed 130) comes up with the highest probability of being speed 30; this makes sense as a 3 is quite distinctive. The second best result is speed 80 and the third best is speed 100; base on these results I would say that the 1 (a vertical line) is not a strong predictor, but an 8 looks very similar to a 3. From there the next best are speed 50 and then speed 120 (I would guess the convolution of a 2 with a 3 is much smaller than a 5 with a 3.)

Image 4 is labeled correctly, but with just 60%. The 2nd best (and ! point, which is very similar in grayscale) has a bit over 30%.

The overall accuracy on these images is 80% (but this is the best it could possibly be)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

I did this above.


## Improvements
Two final notes on improvements.
1. Looking at some German signs, I saw that several of the signs include the mirror image of the symbols. The model I trained in this project would likely not perform well with these signs. A fix would be to include (left-right) mirroring as a possible modification when creating the training data set.
2. I think some color would be helpful to the model. I would first try including color as follows. I would not use the full color layers, as that did not perform well; instead I would try inputing all three channels of the Lab color space images and before any convolution layer, I would split the gray channel from the two color channels. The gray channel would follow the same model as above, the two color channels would go through a large Avg Pooling layer (e.g. 8x8) and the result (a 4x4x2 image) would be directly flattened out and concatenated onto either the large 1080 or the smaller 256 layers.
