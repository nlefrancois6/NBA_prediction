#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 16:29:09 2020

@author: noahlefrancois
"""
import pandas as pd
from sklearn import ensemble, metrics
import matplotlib.pyplot as plt

#Load the pre-processed data
model_data_df = pd.read_csv("pre-processedData_n8.csv")

#Select the model features
away_features = ['teamFG%','teamEFG%','teamOrtg','teamEDiff']
home_features = ['opptTS%','opptEFG%','opptPPS','opptDrtg','opptEDiff','opptAST/TO','opptSTL/TO']
features = ['teamAbbr','opptAbbr','Season'] + away_features + home_features
output_label = ['teamRslt'] 
#Note teamRslt = Win means visitors win, teamRslt = Loss means home wins

#Separate training and testing set
training_df = model_data_df.sample(frac=0.8, random_state=1)
indlist=list(training_df.index.values)

testing_df = model_data_df.copy().drop(index=indlist)

#Define the features (input) and label (prediction output) for training set
training_features = training_df[features]
training_label = training_df[output_label]

#Define features and label for testing set
testing_features = testing_df[features]
testing_label = testing_df[output_label]

#Train a Gradient Boosting Machine on the data
gbc = ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.02, max_depth=1)
gbc.fit(training_features, training_label)

#Predict the outcome from our test set and evaluate the prediction accuracy for each model
predGB = gbc.predict(testing_features) 
pred_probsGB = gbc.predict_proba(testing_features) #probability of [results==True, results==False]
accuracyGB = metrics.accuracy_score(testing_label, predGB)


#Plot feature importances
feature_importance = gbc.feature_importances_.tolist()
f2=plt.figure()
plt.bar(features,feature_importance)
plt.title("Gradient Boosting Classifier: Feature Importance")
plt.xticks(rotation='vertical')
plt.show()


#Could filter the testing set to only "bet" on games where we meet a minimum confidence
#Will need to check whether this raises or lowers profit since we might be missing upsets and actually lose money since we only bet on strong favourites
min_conf_filter = False
if min_conf_filter:
    min_confidence = 0.7

    predGB_minConfSatisfied = []
    predGB_label_minConfSatisfied = []

    labels_arr = testing_label['teamRslt'].array

    for i in range(len(predGB)):
        if (pred_probsGB[i,0] > min_confidence) or (pred_probsGB[i,1] > min_confidence):
            predGB_minConfSatisfied.append(predGB[i])
            predGB_label_minConfSatisfied.append(labels_arr[i])

    accuracy_minConf = metrics.accuracy_score(predGB_label_minConfSatisfied, predGB_minConfSatisfied)


