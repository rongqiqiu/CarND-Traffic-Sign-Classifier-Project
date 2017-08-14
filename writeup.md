###Traffic Sign Recognition

[//]: # (Image References)

[bar_train]: ./bar_train.png "Bar Chart Training"
[bar_valid]: ./bar_valid.png "Bar Chart Validation"
[bar_test]: ./bar_test.png "Bar Chart Test"
[new1]: ./new_data/newpic1.png "Traffic Sign 1"
[new2]: ./new_data/newpic2.png "Traffic Sign 2"
[new3]: ./new_data/newpic3.png "Traffic Sign 3"
[new4]: ./new_data/newpic4.png "Traffic Sign 4"
[new5]: ./new_data/newpic5.png "Traffic Sign 5"

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 34799

* The size of the validation set is 4410

* The size of test set is 12630

* The shape of a traffic sign image is (32, 32, 3)

* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the distribution of labels in training/validation/test sets.

![alt text][bar_train]
![alt text][bar_valid]
![alt text][bar_test]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized all images by using the per-pixel mean and standard deviation of training images. The reason for doing this is that the distribution of training images does not have a mean (or deviation) of 128, so (pixel - 128) / 128 does not work well.

Then, I shuffled the training set.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6  					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU 					| 												|
| Max pooling			| 2x2 stride, outputs 5x5x16					|
| Reshape				| outputs 400									|
| Fully connected		| outputs 120									|
| RELU 					| 												|
| Dropout 				|												|
| Fully connected		| outputs 84									|
| RELU 					| 												|
| Dropout 				|												|
| Fully connected		| outputs 43									|
| Softmax				| 	        									|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, batch size of 128, epochs of 50, learning rate of 0.001, l2 regularization factor of 0.001, dropout probability of 0.5

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.961
* test set accuracy of 0.946

I started with a standard LeNet model with 3 input channels. But it seemed to overfit the training set because the training accuracy is high while validation accuracy is relatively low. So I added L2 regularization to the loss function, and applied dropout to the fully connected layers. Number of epochs is also increased for the training process to converge. 

I explored different possibilities of adding dropout (convolutional layers and/or fully connected layers). Adding dropout to fully connected layers gives the best performance in practice. 

I also tested converting into grayscale image, but then decided to preserve color information for better performance.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][new1] ![alt text][new2] ![alt text][new3] 
![alt text][new4] ![alt text][new5]

Generally, the quality of these 5 images is high: not much noise, not blurred, good constrast with background, no occlusion. The resolution (32x32) might be small for accurate classification though, because small features / distinctions might be lost with this resolution.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing      		| No passing									| 
| Roundabout mandatory	| Roundabout mandatory							|
| Double curve			| General caution								|
| Wild animals crossing	| No passing for vehicles over 3.5 metric tons	|
| Slippery Road			| Slippery road									|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is lower than the accuracy of the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| No passing   									| 
| .00     				| End of no passing 							|
| .00					| No passing for vehicles over 3.5 metric tons	|
| .00	      			| Dangerous curve to the left					|
| .00				    | Vehicles over 3.5 metric tons prohibited      |


For the second image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .95         			| Roundabout mandatory   						| 
| .03     				| Keep right 									|
| .01					| Go straight or right							|
| .003	      			| Go straight or left					 		|
| .002				    | Keep left      								|


For the third image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .91         			| General caution   							| 
| .06     				| Dangerous curve to the right 					|
| .03					| Road work										|
| .001	      			| Traffic signals					 			|
| .000				    | Speed limit (80km/h)      					|


For the fourth image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998         			| No passing for vehicles over 3.5 metric tons	| 
| .002     				| Vehicles over 3.5 metric tons prohibited		|
| .000					| End of no passing by vehicles over 3.5 tons	|
| .000	      			| Dangerous curve to the right				 	|
| .000				    | Right-of-way at the next intersection     	|


For the fifth image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Slippery road   								| 
| .00     				| Double curve 									|
| .00					| Dangerous curve to the left					|
| .00	      			| Wild animals crossing					 		|
| .00				    | Beware of ice/snow      						|


