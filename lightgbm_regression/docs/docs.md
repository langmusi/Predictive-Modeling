---
title: "docs"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Data Preprocessing

Preparing EHR & Tabular Data for Neural Networks (Code Included!):

https://glassboxmedicine.com/2019/06/01/everything-you-need-to-know-about-preparing-tabular-data-for-machine-learning-code-included/

To understand: how to work with class and def


import numpy as np
import pandas as pd
import sklearn.preprocessing

class Splits(object):
    """Split data and labels into train, validation, and test sets.
    If specified, perform imputation, transformation of categorical variables
    into one-hot vectors, and/or normalization of data.
    
    Variables:
    <data> is a pandas dataframe where the index is example IDs
        and the columns are variable names.
    <labels> is a pandas dataframe where the index is example IDs
        and the columns are variable names.
    The indices of <data> and <labels> must match.
    <train_percent>, <valid_percent>, and <test_percent> are floats between
        0 and 1 that specify the proportion of data in each split.
    <impute>: True or False. If True, impute missing values
    <impute_these_categorical>: list of strings. Column names of categorical
        variables to impute with the mode
    <impute_these_continuous>: list of strings. Column names of continuous
        variables to impute with the median
    <one_hotify>: True or False. If True, transform categorical variables into
        one-hot vectors
    <one_hotify_these_categorical>: list of strings. Column names of categorical
        variables that should be represented using one-hot vectors.
    <normalize_data>: True or False. If True, normalize continuous variables
    <normalize_these_continuous>: list of strings. Column names of continuous
        variables to normalize.
    <seed>: int that determines data shuffling"""
    def __init__(self,
             data,
             labels,
             train_percent,
             valid_percent,
             test_percent,
             impute, #if True, impute
             impute_these_categorical, #columns to impute with mode
             impute_these_continuous, #columns to impute with median
             one_hotify, #if True, one-hotify specified categorical variables
             one_hotify_these_categorical, #columns to one hotify
             normalize_data, #if True, normalize data based on training set
             normalize_these_continuous, #columns to normalize
             seed): #seed to determine shuffling order before making splits
        assert (train_percent+valid_percent+test_percent)==1
        assert data.index.values.tolist()==labels.index.values.tolist()
        self.clean_data = data
        self.clean_labels = labels
        self.impute_these_categorical = impute_these_categorical
        self.impute_these_continuous = impute_these_continuous
        self.one_hotify_these_categorical = one_hotify_these_categorical
        self.normalize_these_continuous = normalize_these_continuous
        
        self.train_percent = train_percent
        self.valid_percent = valid_percent
        self.test_percent = test_percent
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(0,10e6)
        
        self._get_split_indices() #defines self.trainidx and self.testidx
        self._shuffle_before_splitting()
        if impute:
            self._impute()
        if one_hotify:
            self._one_hotify()
        if normalize_data:
            self._normalize()
        else:
            print('WARNING: you elected not to normalize your data. This could'+
                  ' lead to poor performance.')
        self._make_splits() #creates self.train, self.valid, and self.test
    
    def _get_split_indices(self):
        """Get indices that will be used to split the data into train, test,
        and validation."""
        self.trainidx = int(self.clean_data.shape[0] * self.train_percent)
        self.testidx = int(self.clean_data.shape[0] * (self.train_percent
                                                       +self.test_percent))

    def _shuffle_before_splitting(self):
        idx = np.arange(0, self.clean_data.shape[0])
        np.random.seed(self.seed)
        print('Creating splits based on seed',str(self.seed))
        np.random.shuffle(idx)
        self.clean_data = self.clean_data.iloc[idx]
        self.clean_labels = self.clean_labels.iloc[idx]

    def _impute(self):
        """Impute categorical variables using the mode of the training data
        and continuous variables using the median of the training data."""
        #impute missing categorical values with the training data mode
        #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mode.html
        print('Imputing categorical variables with mode:\n',
              str(self.impute_these_categorical))
        training_data = self.clean_data.iloc[0:self.trainidx,:]
        imputed_with_modes = (self.clean_data[self.impute_these_categorical]).fillna((
            training_data[self.impute_these_categorical]).mode().iloc[0])
        self.clean_data[self.impute_these_categorical] = imputed_with_modes  
        
        #impute missing continuous values with the training data median
        print('Imputing continuous variables with median:\n',
              str(self.impute_these_continuous))
        imputed_with_medians = (self.clean_data[self.impute_these_continuous]).fillna((
            training_data[self.impute_these_continuous]).median())
        self.clean_data[self.impute_these_continuous] = imputed_with_medians
        
        print('Done imputing')

    def _one_hotify(self):
        """Modify self.clean_data so that each categorical column is turned
        into many columns that together form a one-hot vector for that variable.
        E.g. if you have a column 'Gender' with values 'M' and 'F', split it into
        two binary columns 'Gender_M' and 'Gender_F'"""
        print('One-hotifying',str(len(self.one_hotify_these_categorical)),
              'categorical variables')
        print('\tData shape before one-hotifying:',str(self.clean_data.shape))
        #one hotify the categorical variables
        self.clean_data = pd.get_dummies(data = self.clean_data,
                                         columns = self.one_hotify_these_categorical,
                                         dummy_na = False)
        print('\tData shape after one-hotifying:',str(self.clean_data.shape))

    def _normalize(self):
        """Provide the features specified in self.normalize_these_continuous
        with approximately zero mean and unit variance, based on the
        training dataset only."""
        train_data = (self.clean_data[self.normalize_these_continuous].values)[0:self.trainidx,:]
        #http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler
        scaler = sklearn.preprocessing.StandardScaler().fit(train_data)
        print('Normalizing data:\n\tscaler.mean_',str(scaler.mean_),
              '\n\tscaler.scale_',str(scaler.scale_))
        assert (len(self.normalize_these_continuous)
                ==scaler.mean_.shape[0]
                ==scaler.scale_.shape[0](
        self.clean_data[self.normalize_these_continuous] = scaler.transform((
            self.clean_data[self.normalize_these_continuous]).values)
        
    def _make_splits(self):
        """Split up self.clean_data and self.clean_labels
        into train, test, and valid data."""
        assert self.clean_data.index.values.tolist()==self.clean_labels.index.values.tolist()
        data_matrix = self.clean_data.values
        labels_matrix = self.clean_labels.values
        
        self.train_data = data_matrix[0:self.trainidx,:]
        self.train_labels = labels_matrix[0:self.trainidx]
        self.test_data = data_matrix[self.trainidx:self.testidx,:]
        self.test_labels = labels_matrix[self.trainidx:self.testidx]
        self.valid_data = data_matrix[self.testidx:,:]
        self.valid_labels = labels_matrix[self.testidx:]
        print('Finished making splits')
        print('\tTrain data shape:',str(self.train_data.shape))
        print('\tValid data shape:',str(self.valid_data.shape))
        print('\tTest data shape:',str(self.test_data.shape))
        print('\tLength of one label:',str(self.train_labels.shape[1]))

if __name__ == '__main__':
    #Prepare fake data
    fakedata = pd.DataFrame([['diabetes',120,5.5,0],
        ['diabetes',235,6.1,2],
        ['MISSING',166,5.2,0],
        ['hypertension','MISSING',5.3,'MISSING'],
        ['cancer','MISSING',5.8,3],
        ['hypertension',133,6.5,2]],
        index = ['patient1','patient2','patient3','patient4','patient5','patient6'],
        columns = ['diagnosis','weight','height','painscore'])
    fakedata = fakedata.replace('MISSING',value=np.nan)
    fakedata['weight'] = pd.to_numeric(fakedata['weight'],downcast='float')
    fakedata['height'] = pd.to_numeric(fakedata['height'],downcast='float')
    fakedata['painscore'] = pd.to_numeric(fakedata['painscore'],downcast='integer')
    
    #Prepare fake labels
    fakelabels = pd.DataFrame([[1],[1],[0],[1],[0],[0]],
        index = ['patient1','patient2','patient3','patient4','patient5','patient6'],
        columns = ['outcome'])
    print('Fake data before processing is \n',fakedata)
    
    #Perform processing
    s = Splits(data = fakedata,
               labels = fakelabels,
               train_percent = 0.5,
               valid_percent = 0.25,
               test_percent = 0.25,
               impute = True,
               impute_these_categorical = ['diagnosis','painscore'],
               impute_these_continuous = ['weight','height'],
               one_hotify = True,
               one_hotify_these_categorical = ['diagnosis','painscore'],
               normalize_data = True,
               normalize_these_continuous = ['weight','height'],
               seed = 12345)
    print('Fake data after processing is \n', s.clean_data)


## import packages from utils

Python recognizes a directory as a package when it has a file named __init__.py in it. 

This file is empty.

3 Techniques to Effortlessly Import and Execute Python Modules:  
https://towardsdatascience.com/3-advance-techniques-to-effortlessly-import-and-execute-your-python-modules-ccdcba017b0c

## feature selection

The question from kaggle:  
If i have a dataset with 50 Categorical and 50 numerical variables then how can i perform Feature selection for my Categorical variables. I believe that we can convert those 50 Categorical variables into continuous using One Hot Encoding or Feature Hashing and apply SelectKBest or RFECV or PCA..

Is there any other technique that we can use for feature selection for categorical variables. ?

The concern is some of my categorical variables have more than 10000 distinct values and if i perform One Hot encoding , Im afraid that it will increase my Total dimensions and have an impact on model performance.

The answers:  
1. Chi2 can be used to select categorical features but for a classification problem not a regression problem. ANOVA can be used for categorical feature selection for regression problems.  
2. HeatMap: using one-hot encoding when plotting  
3. One good starting point (because it's simple and easy to check expectations) is to look at the feature importance after fitting a preliminary model. With one hot encoded variables you'l be able to see which levels are important as well as features. Trees are good for this, but be aware that feature importance is affected by correlations in the predictors and won't clearly show interactions.  

If you're worried about the carnality of the features, you could try binning before one hot encoding and/or reducing the poorly represented levels into one "other" category.


The article contains which methods to use for different combinations of input data types and output data types:  
https://www.simplilearn.com/tutorials/machine-learning-tutorial/feature-selection-in-machine-learning
