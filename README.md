# coffeandcode
coffeandcode dataMining
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import cluster

dataSet = pd.read_csv('./CoffeeAndCode.csv')
dataSet.head()
dataSet.info()
#processing missing values
dataSet.isnull().any()
dataSet.isnull().sum()
data1 = dataSet.dropna() #remove 3 rows
data2 = dataSet.dropna(axis=1) # remove 2 columns

#replaces the NULL values with a specified value
data3 = dataSet.fillna(method='bfill') #backward fill the missing values in the dataset
data3_1 = dataSet.fillna(method='ffill') # forward fill the missing values in the dataset
data4 = dataSet.copy()
m = data4.mean()
data4.fillna(m)
data5 = dataSet.copy()
m = data5.median()
data4.fillna(m)
sns.boxplot(x=dataSet['CodingHours'])
sns.boxplot(x=dataSet['CoffeeCupsPerDay'])
dataSet.dtypes 
dataSet.shape
dataSet.describe()
dataSet.duplicated().any() #true
dataSet.duplicated().sum() #3
dataSet.index[dataSet.duplicated()]
dataSet.drop_duplicates()
#boxplot
dataSet.boxplot()
#Histogram
dataSet.hist(bins=50, layout=(1,4), figsize=(20, 4))
plt.show()
sns.countplot(x='AgeRange',data=dataSet)
sns.countplot(x='CoffeeSolveBugs',data=dataSet)
sns.countplot(data=dataSet, y='CoffeeTime')
sns.countplot(data=dataSet, y='CoffeeTime',hue = 'CoffeeTime')

#Frequency 
print(dataSet['Gender'].value_counts(normalize=True))
print('--------------------')
print(dataSet['Gender'].value_counts(normalize=True)*100)
from sklearn.model_selection import train_test_split
x = dataSet.iloc[:,:8]
y = dataSet.iloc[:,:8] #labels

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=45)
print(x_train.shape) #(85, 8)
print(x_test.shape) #(15, 8)
print(classification_report(y_test, y_pred))
