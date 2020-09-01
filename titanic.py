import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

trainingData = pd.read_csv('train.csv')
testingData = pd.read_csv('test.csv')
combinedData = [trainingData, testingData]

#Question 1
#Show training set features
print()
print(trainingData.columns.values)

#Question 2,3,4
#Show a few rows to find which features are categorical, numerical and mixed data types
print()
print(trainingData.head(5))

#Question 5, 6
print()
print(trainingData.info())
print()
print(testingData.info())

#Question 7
print()
print(trainingData.describe())

#Question 8
print()
print(trainingData.describe(include=['O']))
print()
print(testingData.describe(include=['O']))

#Question 9
print()
#Prints a table with columns Pclass and Survived so we can observe the correlation between the two
print(trainingData[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#Question 10
print()
#Prints a table with columns Sex and Survived so we can observe the correlation between the two
print(trainingData[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#Question 11
#Creates two histograms with 20 'bins' each for grouping age groups to visualize what age groups were likely to survive
graph = sns.FacetGrid(trainingData, col='Survived')
graph.map(plt.hist, 'Age', bins=20)
plt.show()

#Question 12
#Creates six histograms with 20 'bins' each for grouping age groups to visualize which age groups in each class were more likely to survive
gridOfGraphs = sns.FacetGrid(trainingData, col='Survived', row='Pclass')
gridOfGraphs.map(plt.hist, 'Age', alpha=0.5, bins=20)
gridOfGraphs.add_legend()
plt.show()

#Question 13
#Creates six histograms to visualize how embarkation location and fare correlated to survival rate
gridOfGraphs = sns.FacetGrid(trainingData, row='Embarked', col='Survived')
gridOfGraphs.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)
gridOfGraphs.add_legend()
plt.show()

#Question 14, 15
#Answers were obtained from looking at print(trainingData.info()) and print(testingData.info())

#Question 16
#Converting the Sex feature from string data to numerical data in trainingData
trainingData['Sex'] = trainingData['Sex'].map({'male':0, 'female':1}).astype(int)
#Changing the name of the Sex feature to Gender in trainingData
trainingData.rename(columns={'Sex': 'Gender'}, inplace=True)
#Converting the Sex feature from string data to numerical data in testingData
testingData['Sex'] = testingData['Sex'].map({'male':0, 'female':1}).astype(int)
#Changing the name of the Sex feature to Gender in testingData
testingData.rename(columns={'Sex': 'Gender'}, inplace=True)
print()
print(trainingData.head())

#Question 17
#Fill missing Age values by generating random numbers between mean and standard deviation
#trainingData
trainingDataAgeFeatureNullsRemoved = trainingData.Age.dropna()
meanOfTrainingDataAgeFeature = trainingDataAgeFeatureNullsRemoved.mean()
standardDeviationOfTrainingDataAgeFeature = trainingDataAgeFeatureNullsRemoved.std()
randomAge = rnd.uniform(meanOfTrainingDataAgeFeature - standardDeviationOfTrainingDataAgeFeature, meanOfTrainingDataAgeFeature + standardDeviationOfTrainingDataAgeFeature)
randomAge = int(randomAge)
trainingData['Age'] = trainingData['Age'].fillna(randomAge)

print()
print("Training Data Random Age: " + str(randomAge))
print(trainingData.info())

#testingData
testingDataAgeFeatureNullsRemoved = testingData.Age.dropna()
meanOfTestingDataAgeFeature = testingDataAgeFeatureNullsRemoved.mean()
standardDeviationOfTestingDataAgeFeature = testingDataAgeFeatureNullsRemoved.std()
randomAge = rnd.uniform(meanOfTestingDataAgeFeature - standardDeviationOfTestingDataAgeFeature, meanOfTestingDataAgeFeature + standardDeviationOfTestingDataAgeFeature)
randomAge = int(randomAge)
testingData['Age'] = testingData['Age'].fillna(randomAge)

print()
print("Testing Data Random Age: " + str(randomAge))
print(testingData.info())

#Question 18
#Find most common Embarked feature occurence
mostCommonEmbarkationPort = trainingData.Embarked.dropna().mode()[0]
print()
print("Most common Embarkation Port: " + str(mostCommonEmbarkationPort))

#Fill the null values in trainingData and testingData with the mostCommonEmbarkationPort value
trainingData['Embarked'] = trainingData['Embarked'].fillna(mostCommonEmbarkationPort)
testingData['Embarked'] = testingData['Embarked'].fillna(mostCommonEmbarkationPort)

print()
print(trainingData.info())

#Question 19
#Find most common Fare feature occurence
mostCommonFareValue = testingData['Fare'].dropna().mode()[0]
print()
print("Most Common Fare Value: " + str(mostCommonFareValue))

#Fill the one null value in the testingData set
testingData['Fare'] = testingData['Fare'].fillna(mostCommonFareValue)
print()
print(testingData.info())

#Question 20
#Convert Fare feature to ordinal values based on the FareBand defined in the hw pdf
#Converting trainingData
trainingData.loc[trainingData['Fare'] <= 7.91, 'Fare'] = 0
trainingData.loc[(trainingData['Fare'] > 7.91) & (trainingData['Fare'] <= 14.454), 'Fare'] = 1
trainingData.loc[(trainingData['Fare'] > 14.454) & (trainingData['Fare'] < 31), 'Fare'] = 2
trainingData.loc[trainingData['Fare'] > 31, 'Fare'] = 3
trainingData['Fare'] = trainingData['Fare'].astype(int)
print()
print(trainingData.head(5))

#Converting testingData
testingData.loc[testingData['Fare'] <= 7.91, 'Fare'] = 0
testingData.loc[(testingData['Fare'] > 7.91) & (testingData['Fare'] <= 14.454), 'Fare'] = 1
testingData.loc[(testingData['Fare'] > 14.454) & (testingData['Fare'] < 31), 'Fare'] = 2
testingData.loc[testingData['Fare'] > 31, 'Fare'] = 3
testingData['Fare'] = testingData['Fare'].astype(int)
print()
print(testingData.head(5))