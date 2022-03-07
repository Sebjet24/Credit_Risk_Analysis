# Credit Risk Analysis

## Overview

Jill commended me for all my hard work. Piece by piece, I’ve been building up my skills in data preparation, statistical reasoning, and machine learning. I am now ready to apply machine learning to solve a real-world challenge: credit card risk.

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, I’ll need to employ different techniques to train and evaluate models with unbalanced classes. Jill asked me to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Once I'm done, I’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Results

We resample the dataset using 'Machine Learning' utilizing 'Python' libraries:'scikit-learn' and 'imbalanced-learn' analyze the findings and offer a comparison for our study, as described in the introduction.

In the first quarter of 2019, the original dataset includes 115,675 loan applications. To establish whether the application was considered "low" or "high" risk, we utilized the "loan status." The applications with "current" as the "loan status" were categorized as "low risk," while the others were categorized as "high risk." This decreased the total number of applications to 68,817, with 99 percent of them being classed as "low risk." 

![datacount](https://user-images.githubusercontent.com/91230277/156951404-a1e6d05c-c027-40a0-ab84-d89301efbd45.png)

The training set was divided into 51,366 "low risk" and 246 "high risk" applications using the 75/25 percent approach to partition the data for training vs. testing. 

![trainingdata](https://user-images.githubusercontent.com/91230277/156951447-a5a58607-fdb7-49fc-83c1-2eaa31b779f1.png)

### Oversampling

Until both categories are equal, the **'RandomOverSampler Model'** randomly chooses from the minority class and adds it to the training set. The results divided 51,366 data into two categories: High Risk and Low Risk.

![oversamplecount](https://user-images.githubusercontent.com/91230277/156951484-ae91bf6e-f60e-43c4-97fb-9e1aa00749ab.png)

 - Balanced accuracy score: 64%.

  ![oversampleacc](https://github.com/amylio/Credit_Risk_Analysis/blob/main/Images/oversampleacc.png)

 - The accuracy rate for "High Risk" was just 1%, while the recall was 66 percent, giving this model an F1 score of 2%.
 - Low Risk had an accuracy rate of 100% and a recall rate of 62%.  
  
![oversamplecm](https://user-images.githubusercontent.com/91230277/156951538-ac508e4e-82e2-488b-b276-bbb2904274b9.png)

![oversampleclass](https://user-images.githubusercontent.com/91230277/156951566-61cdf3ea-3814-4142-9ebe-857bd993ceb4.png)

**'SMOTE (Synthetic Minority Oversampling Technique) Model,' like 'RandomOverSampler,' increases the size of the minority class by producing new values depending on the value of the minority class's nearest neighbors rather than random selection.

 - The overall accuracy score rose to 65.1 percent.

![Smoteacc](https://user-images.githubusercontent.com/91230277/156951610-0946d899-caea-4fa3-b11d-5bd14987e5a5.png)

 - The "High Risk" precision rate was just 1%, with the recall reduced to 61 percent, giving this model an F1 score of 2%, similar to 'RandomOverSampler.'
 - The accuracy rate for "Low Risk" was 100 percent, with a recall rate of 69 percent.  

![SmoteCM](https://user-images.githubusercontent.com/91230277/156951621-f8744e73-5903-4fdd-b70a-a8fe46a4cf28.png)

![SmoteClass](https://user-images.githubusercontent.com/91230277/156951627-9dbf63a0-7b8f-4a41-ab20-36158af03e96.png)

### Undersampling

**'ClusterCentroids Model,'** an approach for generating synthetic data points that are indicative of clusters by identifying clusters of the majority class. The algorithm divided 246 data into two categories: High Risk and Low Risk.

![undersamplecount](https://user-images.githubusercontent.com/91230277/156951657-807d552d-2cdd-45c1-bf02-55a1b3807ab7.png)

 - The balanced accuracy score was 54.5 percent, which was lower than the oversampling models.

![underacc](https://user-images.githubusercontent.com/91230277/156951663-72106234-aca9-41fd-8447-7218ce2d4bae.png)

 - The accuracy rate for "High Risk" was just 1%, while the recall was only 69 percent, giving this model an F1 score of 1%.
 - In comparison to the oversampling models, "Low Risk" had an accuracy rate of 100% and a recall rate of 40%.

![undercm](https://user-images.githubusercontent.com/91230277/156951729-dcf9884b-9e1c-4e09-8a8e-6b5a9dea419d.png)

![underclass](https://user-images.githubusercontent.com/91230277/156951740-893e549f-5d30-422e-8f0a-f36f12c55fbc.png)  

### Combination Sampling

**'SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model'** combines oversampling and undersampling techniques. The algorithm assigned a High Risk rating to 68,460 records and a Low Risk rating to 62,011.

![SMOTEENNcount](https://user-images.githubusercontent.com/91230277/156951779-fe429efd-b95b-4c36-b44a-c9c80fe878a7.png)

 - When utilizing a mixed sampling strategy, the balanced accuracy score increased to 64.5 percent.

![SMOTEENNacc](https://user-images.githubusercontent.com/91230277/156951791-cd54fe87-bdaa-4653-a859-9c351d4d5a87.png)

 - The "High Risk" accuracy rate remained unchanged at 1%, but the recall climbed to 72 percent, giving this model an F1 score of 2%.
 - "Low Risk" had an accuracy rate of 100% and a recall rate of 57%.
  
![SMOTEENNcm](https://user-images.githubusercontent.com/91230277/156951804-f190c11c-07ef-4616-bc37-867aed184c6c.png)

![SMOTEENNclass](https://user-images.githubusercontent.com/91230277/156951809-bf2f19ce-6c46-4ecc-ab2a-f33da4f9afc8.png)

### Use Ensemble Classifiers to Predict Credit Risk

Compare two new 'Machine Learning' algorithms for predicting credit risk that eliminate bias. The algorithms assigned 51,366 people to the High Risk category and 246 to the Low Risk category.

![balancedcount](https://user-images.githubusercontent.com/91230277/156951837-f3cb6e69-1d79-4605-87b6-ea3eb696e093.png)

Two trees of the same size and equal size to the minority class are produced to represent one for the majority class and one for the minority class in the **'BalancedRandomForestClassifier Model.'**

 - For this model, the balanced accuracy score climbed to 78.9%.

![Balancedacc](https://user-images.githubusercontent.com/91230277/156951864-f9449095-89ca-493d-96d5-5730c6313cb6.png)

 - The accuracy rate for "High Risk" increased to 3%, with a recall of 70%, giving this model an F1 score of 6%.
 - "Low Risk" nevertheless had a 100% accuracy rate and an 87 percent recall rate. 
 - The "total rec prncp" was the most important characteristic, accounting for 7.9% of the total.

![Balancedcm](https://user-images.githubusercontent.com/91230277/156951937-060ca4b8-321e-48a3-850c-5cb8cc37d9a3.png)

![balancedclass](https://user-images.githubusercontent.com/91230277/156951947-a25fd00f-feb6-4b0a-be32-dc5c27a11fe0.png)

![BalancedFeature](https://user-images.githubusercontent.com/91230277/156951955-8b6cedca-edb5-4c8d-8203-7cb49f9aaf7d.png)

A collection of classifiers called the **'EasyEnsembleClassifier Model'** combines individual judgments to categorize fresh samples.

 - With this model, the balanced accuracy score climbed to 93.2 percent.

![Easyacc](https://user-images.githubusercontent.com/91230277/156952143-4c16f8bc-4268-4d53-842f-4c92a9c94a58.png)

 - The accuracy rate for "High Risk" climbed to 9%, with a recall of 92 percent, giving this model an F1 score of 16%.
 - The accuracy rate for "Low Risk" remained at 100 percent, with the recall currently at 94 percent.

![Easycm](https://user-images.githubusercontent.com/91230277/156952084-cb16fda0-8c80-454b-b5de-5f085fe48c23.png)

![Easyclass](https://user-images.githubusercontent.com/91230277/156952099-ccd7d17f-45a9-4fa2-acac-b3fec8540ace.png)

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
