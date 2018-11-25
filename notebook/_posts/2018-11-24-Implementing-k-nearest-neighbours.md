---
layout: post
title: "Implementing k-Nearest Neighbor classifier"
date:   2018-11-24
category: notebook
comments: true
math: true
author: "Pavan Kumar Polavarapu"
---

> If you are currently doing CS231n first assignment, Please be advised that article contains code implementation and now is the right time to close the tab.

> In my previous post, I spoke little about K-Nearest-Neighbours Classifier and in this post, we will implement one.

Since, I am following CS231n course content, I would be using notations and variable naming associated with the assignment. 

### Loading the CIFAR10 dataset
Before we can proceed, we need to understand image data. CIFAR10 dataset contains 60,000 labeled images. These 60, 000 images are divided into 10 categories with each category containing 6000 images. Each image is of 32X32 resolution with each pixel containing 3 color bits.

The logic for loading images is already taken care in data_utils.py file assuming that we have placed the dataset in cs231n/datasets. 

Below is sample visualization of dataset 

![CIFAR Sample](/assets/notebook-images/knn/cifar10-dataset.png){:width="367px"}  
__Figure 1:__ _Samples of the CIFAR-10 Dataset_
{: style="text-align: center;"}

As you can see there are 7 samples from each of these 10 categories. These images are tiny compared to our 10 mega pixel mobile images but from the quality of color in these images you can easily say that these images are compressed for computational efficiency. 

### Compute Distance
In our previous post, we have discussed about the two approaches (L1(Manhattan) and L2(Euclidean)) for distance calculations. These distances are calculated pixel by pixel and two images are considered to be close when square root of sum of the distances between pixels is minimum. 

Considering two images $$ X_{i}, X_{j} $$. L2 distance between these two images is given as below

$$ d_{2}(X_{i}, X_{j}) = \sqrt{\sum_{p}(X_{i}^{p} - X_{j}^{p})^{2}} $$

Where $$ X_{i}^{p} $$ represent single pixel.

## <a name="twoloop">Two Loop Implementation</a>
We want to calculate difference for each test image with all the training images. We have two different arrays for training and testing images, naive and easiest approach would be is to loop through each test image and loop through each training image and compute the distance between train and test images. However, as you can imagine the run time complexity would be $$ O(n_^{2}) $$.

```Python
def compute_distances_two_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #np.linalg.norm calculates the L2 distance
        #self.X_train type 500 X 3072 and X is of type 5000 X 3072
        dists[i, j] = np.linalg.norm(self.X_train[j, :]-X[i, :])
        # dists[i, j] = np.sqrt(np.sum((X[i, :] - self.X_train[j, :]) ** 2))
    return dists
```

## <a name="oneloop"> One Loop Implementation </a>
One loop implementation supposed to improve run time performance by eliminating the secondary loop on training images. However, emperical results differ. Below is the logic for one loop implementation.

```Python
def compute_distances_one_loop(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #self.X_train type 5000 X 3072 and X is of type 500 X 3072
      # axis = 1 breaks 5000 in 5000 X 3072 and calculates for every  row
      dists[i, :] = np.linalg.norm(self.X_train - X[i,:], axis = 1)
    return dists          
```

## <a name="noloop"> No Loop Implementation </a>
Run time performance can be greatly improved if we can compute the distance between these images without any loops. Assuming each image as a vector represented in n-dimensional space is the approach that we can use to eliminate the loops. 

Distance between any two vectors is given by the equation

$$ d_{2}(\mathbf{X}{i}, \mathbf{X}{j}) = \left\lvert \left\lvert \mathbf{X}{i} - \mathbf{X}{j} \right\rvert \right\rvert = \sqrt{\left\lvert \left\lvert\mathbf{X}{i} \right\rvert \right\rvert^{j} + \left\lvert \left\lvert \mathbf{I}{j}\right\rvert \right\rvert^{j} - 2 \mathbf{X}{i} \cdot \mathbf{X}{j}} $$

For theoretical purposes, Please find proof of distance between two vectors with two basis vectors of x, y co-ordinate plane.
![Distance Calculation](/assets/notebook-images/knn/distancecalculation.jpg)  
__Figure 2:__ _Distance Calculation_
{: style="text-align: center;"}

Following is the python code implementation of distance between two vectors.

```Python
def compute_distances_no_loops(self, X):
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    dists = np.sqrt(-2 * np.dot(X, self.X_train.T) + np.sum(self.X_train ** 2, axis=1) + np.sum(X**2, axis=1)[:, np.newaxis])    
```

## <a name="visualization"> Visualizing the distances between images</a>
Once you have the distances of test images with all that of training images, we can plot the distances and visualize the distance matrix. Output will look something like below image

![Distance between Training and Testing Images](/assets/notebook-images/knn/distances.png)  
__Figure 3:__ _Distance between Training and Testing Images_
{: style="text-align: center;"}

One thing that you can observe is, there are structured patterns with different levels of darkness. Brightness indicates greater distance and Darkness indicates smaller distance between training and testing images. 

Since image distance calculation is pixel to pixel, I assume that images with similar background cancel each other out to produce smaller distance irrespective of the object in the image. For example, bird in sky and aeroplane in sky, horse in grass and a cow in grass. It is possible that, horizontal dark or light line is when testing image has common/unsimilar colors and vertical dark and light line is when training image has common/unsimilar colors. To be honest, I was expecting a graph that looks like stars in the sky but somehow many images have strong/weak correlation with many other images.

## <a name="prediction"> Predicting labels </a>
Since we have taken first 500 images of test dataset, Prediction accuracy should be same for all. It is 27.4% with K=1 and 27.8% with K=5. So, 1 of 4 images classified are correct. That's a very low accuracy but the execution finished in 0.3 seconds on my PC. Ideally, we want both accuracy and speed but that's a good start.

## <a name="timetaken"> Time Taken </a>
In my PC, following is the time taken by each of the three different versions of implementation
```
Two loop version took 18.966102 seconds
One loop version took 70.411869 seconds
No loop version took 0.283223 seconds 
```

As I mentioned earlier, One loop is kind of outlier, don't know the exact reason why it took more than two loop version and TA explanation is not so convincing. Will figure it out later.

## <a name="crossvalidation">Cross Validation to Find Best K</a>
One of the approaches to improve the efficiency of KNN algorithm is to compare distance with more neighbours and classify. We have tried K as 1 and 4 but we don't know what would be an ideal value of K and that's when Cross validation helps. So let's cross validate the dataset with different values of K and observe efficiency of algorithm for each value of K.

Since we have subsampled the training dataset to 5000 images, it is easy for us to divide the lot into 5 folds and during each iteration we can pick one fold as cross validation set. The values of K are already chosen for the assignment in the book, we can alternatively change are chose to run with this K. Following is the code implementation and plot of performance over K.

```Python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
#duplicate keys in dictionary are not allowed so creating a list
k_to_accuracies = {}
for k in k_choices:
    k_to_accuracies[k] = []

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
    for fold in range(num_folds):
        #Cross Validation
        num_test_crossval = 1000
        #Every Single time pick one fold in total folds for test validation
        X_test_crossval = X_train_folds[fold]
        y_test_crossval = y_train_folds[fold]
        #Pick rest of the folds as training data
        X_train_crossval = np.vstack(X_train_folds[0:fold] + X_train_folds[fold+1:])
        y_train_crossval = np.hstack(y_train_folds[0:fold]+y_train_folds[fold+1:])

        #Training the classifier
        classifier.train(X_train_crossval, y_train_crossval)
        #Calculating the L2 distance for test data
        dists_crossval = classifier.compute_distances_no_loops(X_test_crossval)
        #Predicting the output with current k value
        y_test_pred = classifier.predict_labels(dists_crossval, k)

        #Calculating the accuracy
        num_correct = np.sum(y_test_pred == y_test_crossval)        
        accuracy = float(num_correct) / num_test_crossval

        k_to_accuracies[k].append(accuracy)
        
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
```
![Accuracy over K](/assets/notebook-images/knn/crossvalidation.png)  
__Figure 4:__ _Accuracy for different values of K_
{: style="text-align: center;"}

Seems like best accuracy is between 10 and 15, we can try more values in the range and decide the ideal K.

## <a name="finalthoughts"> Final Thoughts </a>
Comparing images pixel by pixel and classifying will definitely not be accurate. Having said that, KNN classified 1/4th of 5000 images correctly in 0.2 seconds, which in my opinion is awesome. 27.8% Accuracy is not something that we are hoping but it is a good start. 