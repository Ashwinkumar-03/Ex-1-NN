### NAME : Ashwin Kumar. S
### REGISTER NO : 212222240013</H3>
<H3>EX. NO.1</H3>
<H3>DATE : </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))

```

## OUTPUT:

### DATASET:
![image](https://github.com/Ashwinkumar-03/Ex-1-NN/assets/118663725/30755c22-36b6-44f7-8238-32815423eda5)

### DROPPING THE UNWANTED DATASET:
![image](https://github.com/Ashwinkumar-03/Ex-1-NN/assets/118663725/afdb3f88-afd8-4ae6-93d4-112e76fc91b2)

### CHECKING NULL VALUES:
![image](https://github.com/Ashwinkumar-03/Ex-1-NN/assets/118663725/96f9d9cf-c455-4b78-9dae-d7353dc60b66)

### CHECKING FOR DUPLICATION:
![image](https://github.com/Ashwinkumar-03/Ex-1-NN/assets/118663725/3a218ccc-a1d2-4023-a52e-c57288475310)

### DESCRIBING THE DATASET:
![image](https://github.com/Ashwinkumar-03/Ex-1-NN/assets/118663725/8397cc7e-b6c5-41ea-9919-5ccee80fda41)

### SCALING THE DATASET:
![image](https://github.com/Ashwinkumar-03/Ex-1-NN/assets/118663725/631a510e-162a-4d64-8094-14db31d264a9)

### X FEATURES:
![image](https://github.com/Ashwinkumar-03/Ex-1-NN/assets/118663725/40f48705-4a16-4f8b-81b0-a078cca187ec)

### Y FEATURES:
![image](https://github.com/Ashwinkumar-03/Ex-1-NN/assets/118663725/0c681ad1-58fc-48db-a542-20bd2d0d7e4b)

### SPLITTING THE TRAINING AND TESTING DATASET:
![image](https://github.com/Ashwinkumar-03/Ex-1-NN/assets/118663725/66add820-b0dd-4b98-9a6f-380b0d25dfb4)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


