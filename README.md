# Credit Risk Analysis

## Overview

Jill commended me for all my hard work. Piece by piece, I’ve been building up my skills in data preparation, statistical reasoning, and machine learning. I am now ready to apply machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, I’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asked me to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once I'm done, I’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Results

We resample the dataset using 'Machine Learning' utilizing 'Python' libraries:'scikit-learn' and 'imbalanced-learn' analyze the findings and offer a comparison for our study, as described in the introduction.

In the first quarter of 2019, the original dataset includes 115,675 loan applications. To establish whether the application was considered "low" or "high" risk, we utilized the "loan status." The applications with "current" as the "loan status" were categorized as "low risk," while the others were categorized as "high risk." This decreased the total number of applications to 68,817, with 99 percent of them being classed as "low risk." 



The training set was divided into 51,366 "low risk" and 246 "high risk" applications using the 75/25 percent approach to partition the data for training vs. testing. 



### Oversampling

Until both categories are equal, the **'RandomOverSampler Model'** randomly chooses from the minority class and adds it to the training set. The results divided 51,366 data into two categories: High Risk and Low Risk.



 - Balanced accuracy score: 64%.

  ![oversampleacc](https://github.com/amylio/Credit_Risk_Analysis/blob/main/Images/oversampleacc.png)

 - The accuracy rate for "High Risk" was just 1%, while the recall was 66 percent, giving this model an F1 score of 2%.
 - Low Risk had an accuracy rate of 100% and a recall rate of 62%.  
  

  


**'SMOTE (Synthetic Minority Oversampling Technique) Model,' like 'RandomOverSampler,' increases the size of the minority class by producing new values depending on the value of the minority class's nearest neighbors rather than random selection.

 - The overall accuracy score rose to 65.1 percent.



 - The "High Risk" precision rate was just 1%, with the recall reduced to 61 percent, giving this model an F1 score of 2%, similar to 'RandomOverSampler.'
 - The accuracy rate for "Low Risk" was 100 percent, with a recall rate of 69 percent.  


  


### Undersampling

**'ClusterCentroids Model,'** an approach for generating synthetic data points that are indicative of clusters by identifying clusters of the majority class. The algorithm divided 246 data into two categories: High Risk and Low Risk.



 - The balanced accuracy score was 54.5 percent, which was lower than the oversampling models.



 - The accuracy rate for "High Risk" was just 1%, while the recall was only 69 percent, giving this model an F1 score of 1%.
 - In comparison to the oversampling models, "Low Risk" had an accuracy rate of 100% and a recall rate of 40%.


  


### Combination Sampling

**'SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model'** combines oversampling and undersampling techniques. The algorithm assigned a High Risk rating to 68,460 records and a Low Risk rating to 62,011.



 - When utilizing a mixed sampling strategy, the balanced accuracy score increased to 64.5 percent.



 - The "High Risk" accuracy rate remained unchanged at 1%, but the recall climbed to 72 percent, giving this model an F1 score of 2%.
 - "Low Risk" had an accuracy rate of 100% and a recall rate of 57%.
  




### Use Ensemble Classifiers to Predict Credit Risk

Compare two new 'Machine Learning' algorithms for predicting credit risk that eliminate bias. The algorithms assigned 51,366 people to the High Risk category and 246 to the Low Risk category.



Two trees of the same size and equal size to the minority class are produced to represent one for the majority class and one for the minority class in the **'BalancedRandomForestClassifier Model.'**

 - For this model, the balanced accuracy score climbed to 78.9%.



 - The accuracy rate for "High Risk" increased to 3%, with a recall of 70%, giving this model an F1 score of 6%.
 - "Low Risk" nevertheless had a 100% accuracy rate and an 87 percent recall rate. 
  * The "total rec prncp" was the most important characteristic, accounting for 7.9% of the total.


  




A collection of classifiers called the **'EasyEnsembleClassifier Model'** combines individual judgments to categorize fresh samples.

 - With this model, the balanced accuracy score climbed to 93.2 percent.



 - The accuracy rate for "High Risk" climbed to 9%, with a recall of 92 percent, giving this model an F1 score of 16%.
 - The accuracy rate for "Low Risk" remained at 100 percent, with the recall currently at 94 percent.


  


## Summary

When all six models were compared, the 'EasyEnsembleClassifer' model produced the best results, with a 93.2 percent accuracy rate and a 9% precision rate when identifying "High Risk candidates." In comparison to the other models, the sensitivity rate (also known as recall) was the greatest at 92 percent. With a sensitivity rate of 94 percent and an F1 score of 97 percent, the result for predicting "Low Risk" was likewise the best. As a result, if a model for doing this sort of analysis were to be proposed, this would be the clear winner.

### Models are ranked in descending order based on their "High Risk" results:
* `EasyEnsembleClassifer`: 93.2% accuracy, 9% precision, 92% recall, and 16% F1 Score
* `BalancedRandomForestClassifer`: 78.9% accuracy, 3% precision, 70% recall and 6% F1 Score
* `SMOTE`: 65.2% accuracy, 1% precision, 61% recall and 2% F1 Score
* `SMOTEENN`: 64.5% accuracy, 1% precision, 72% recall and 2% F1 Score
* `RandomOverSampler`: 64.0% accuracy, 1% precision, 66% recall and 2% F1 Score
* `ClusterCentroids`: 54.5% accuracy, 1% precision, 69% recall and 1% F1 Score

One thing to keep in mind is that the initial dataset had 99 percent of the applications classed as "Low Risk," with only 1% of the data classified as "High Risk." This might drastically bias the findings, since there's a chance that the 'Machine Learning' algorithms are building clusters from a dataset of genuine "High Risk" apps that is too tiny. Banks might not be willing to tolerate such a high risk margin.
