#libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

#same custom scaler

class StandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

#the actual model that will predict the data

class AbsenteeismModel():

        #get the model and scaler we had previously saved

        def __init__(self, model_file, scaler_file):
            with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None

        #i'm going to copy the lines i wrote in the preprocessing jupyter notebook

        def ETL_process(self, data):

            #get the csv
            df = pd.read_csv(data, delimiter=',')

            #copy for later
            self.df_predictions = df.copy()

            #drop the ID column
            df = df.drop(['ID'], axis=1)

            #get dummies
            reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
            
            #group up reasons for absenteeism
            reason_1 = reason_columns.loc[:, 1:14].max(axis=1)
            reason_2 = reason_columns.loc[:, 15:17].max(axis=1)
            reason_3 = reason_columns.loc[:, 18:21].max(axis=1)
            reason_4 = reason_columns.loc[:, 22:].max(axis=1)

            #we no longer need this column
            df = df.drop(['Reason for Absence'], axis=1)
            
            #insert the columns into the dataframe
            df.insert(0, 'Reason 1', reason_1)
            df.insert(1, 'Reason 2', reason_2)
            df.insert(2, 'Reason 3', reason_3)
            df.insert(3, 'Reason 4', reason_4)

            #Date column to datetime
            df.Date = pd.to_datetime(df.Date, format= '%d/%m/%Y')

            #save month and the of the week and make them their columns
            months = [df.Date[i].month for i in range(df.Date.count())]
            day_of_the_week = [df.Date[i].weekday()+1 for i in range(df.Date.count())]
            df.insert(5, 'Day of the Week', day_of_the_week)
            df.insert(6, 'Month', months)

            #drop the Date column
            df= df.drop(['Date'], axis=1)

            #map the Education column
            df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

            #map the Pets column
            df.Pets = df.Pets.map({1:1, 0:0, 4:1, 2:1, 5:1, 8:1})

            #map the Children column
            df.Children = df.Children.map({2:1, 1:1, 0:0, 4:1, 3:1})

            #separate weight classes by body mass
            underweight = []
            normal_weight = []
            overweight = []
            obese = []

            for i in range(len(df['Body Mass Index'])):
                if df['Body Mass Index'][i] < 18.5:
                    underweight.append(1)
                else:
                    underweight.append(0)

            for i in range(len(df['Body Mass Index'])):
                if df['Body Mass Index'][i] >= 18.5 and df['Body Mass Index'][i] <= 24.9:
                    normal_weight.append(1)
                else:
                    normal_weight.append(0)

            for i in range(len(df['Body Mass Index'])):
                if df['Body Mass Index'][i] > 24.9 and df['Body Mass Index'][i] <= 29.9:
                    overweight.append(1)
                else:
                    overweight.append(0)

            for i in range(len(df['Body Mass Index'])):
                if df['Body Mass Index'][i] > 29.9:
                    obese.append(1)
                else:
                    obese.append(0)

            #insert the columns into the dataframe
            df.insert(10, 'Normal Weight', normal_weight)
            df.insert(11, 'Overweight', overweight)
            df.insert(12,'Obese', obese)

            #drop the body mass index column
            df = df.drop(['Body Mass Index'], axis=1)

            #drop irrelevant columns
            df = df.drop(columns=['Day of the Week', 'Distance to Work', 'Daily Work Load Average'], axis=1)

            # replace the NaN values
            df = df.fillna(value=0)

            #save the data in case we want to use it later
            self.preprocessed_data = df.copy()

            #saving parameters for other functions to use
            self.data = self.scaler.transform(df)

        #this outputs the chance of the data to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred

        #prediction for absenteeism, either 0 or 1
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs

        #adds the probability and prediction columns to the end of the dataframe after calculations
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                #the next line simply turns the values in Probability into easier to interpret percentages
                self.preprocessed_data['Probability'] = [round(x*100,2) for x in self.preprocessed_data['Probability']]
                return self.preprocessed_data
