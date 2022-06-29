# Churn-Classification
### The dataset is the details of the customers in a company , the main task was to make classification where the dataset is imbalanced class

## What is imbalanced data?
### Imbalanced data refers to those types of datasets where the target class has an uneven distribution of observations, where one class label has a very high number of observations and the other has a very low number of observations .
### lass imbalanced is generally normal in classification problems. But, in some cases, this imbalance is quite acute where the majority class’s presence is much higher than the minority class.
### the main problem with imbalanced dataset prediction is how accurately are we actually predicting both majority and minority class , Sometimes when the records of a certain class are much more than the other class, our classifier may get biased towards the prediction .

## How to to deal with the imbalanced dataset proble ?
### Here I discuss some of the few techniques which can deal with this problem. There is no right method or wrong method in this, different techniques work well with different problems :

## 1- Evaluation Metric :
### The accuracy of a classifier is the total number of correct predictions by the classifier divided by the total number of predictions. This may be good enough for a well-balanced class but not ideal for the imbalanced class problem. The other metrics such as precision is the measure of how accurate the classifier’s prediction of a specific class and recall is the measure of the classifier’s ability to identify a class.
### For an imbalanced class dataset F1 score is a more appropriate metric. It is the harmonic mean of precision and recall

## 2- Resampling (Oversampling and Undersampling) :
### This technique is used to upsample or downsample the minority or majority class. When we are using an imbalanced dataset, we can oversample the minority class using replacement. This technique is called oversampling. Similarly, we can randomly delete rows from the majority class to match them with the minority class which is called undersampling. After sampling the data we can get a balanced dataset for both majority and minority classes. So, when both classes have a similar number of records present in the dataset, we can assume that the classifier will give equal importance to both class .
### SMOTE (Synthetic Minority Oversampling Technique):
### is another technique to oversample the minority class. Simply adding duplicate records of minority class often don’t add any new information to the model , SMOTE looks into minority class instances and use k nearest neighbor to select a random nearest neighbor, and a synthetic instance is created randomly in feature space.

## 3- Class Weights :
### giving different weights to both the majority and minority classes. The difference in weights will influence the classification of the classes during the training phase. The whole purpose is to penalize the misclassification made by the minority class by setting a higher class weight and at the same time reducing weight for the majority class.

## 4- BalancedBaggingClassifier :
### A BalancedBaggingClassifier is the same as a sklearn classifier but with additional balancing. It includes an additional step to balance the training set at the time of fit for a given sampler

## All this approachs on Churn dataset to make better classification , after doing all of them i get the best approach results and make a doployment using fastapi 

## Resources 
- https://machinelearningmastery.com/what-is-imbalanced-classification/
- https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/
- https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/
- https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
- https://imbalanced-learn.org/stable/user_guide.html#user-guide
