
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
from logging import info
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#modularization
import logging

from Handling_missing_values import handling_missing_values
from logging_code import setup_logging
logger = setup_logging('main') #creating a new logging file for main

from variable_transformation import Var_Transforamtion
from outilers import outlier

from Feature_selection import filter_method
from Categorical_to_Numerical import Cat_to_Num

from balancing_data import balance_data

from All_models import common

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pickle

from hyper_tuning import hyper_parameter


class CHURN:
    def __init__(self,path): #calling the dataset
        try:
            self.path = path
            self.df = pd.read_csv(self.path) #loading the data into df
            self.df = self.df.drop(['customerID'],axis=1)
            logger.info(self.df)
            logger.info(f'Before Updated dataset Size is: {self.df.shape}')  #Before updating column



            # Adding a new column "sim"
            def add_sim(df):
                if df['PaymentMethod'] == 'Electronic check':
                    return 'Reliance Jio'
                elif df['PaymentMethod'] == 'Mailed check':
                    return 'Airtel'
                elif df['PaymentMethod'] == 'Bank transfer (automatic)':
                    return 'Vi-idea'
                else:
                    return 'BSNL'

            self.df['Sim'] = self.df.apply(add_sim, axis=1)
            logger.info(f'After updated data is {self.df}')        #After updating dataset
            logger.info(f'After updated dataset Size is: {self.df.shape}')
            logger.info(f'After updated dataset coulmns is: {self.df.columns}')

            logger.info(f'Checking for null values')
            for i in self.df.columns:
                logger.info(f'{i} -> {self.df[i].isnull().sum()}')

            logger.info(f'======= converting the string into float ================')

            # converting the total charge's into numeric
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(" ", np.nan)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
            logger.info(f'{self.df.info()}')

            for i in self.df.columns:
                logger.info(f'{i} -> {self.df[i].isnull().sum()}')

            #spliting  the data into independent(X) and dependent(y)
            self.X = self.df.drop('Churn', axis=1)
            self.y = self.df['Churn']

            logger.info(f'checking the column names X : {self.X.columns}')
            logger.info(f'checking the shape of y : {self.y.shape}')
            logger.info(f'checking the shape of X : {self.X.shape}')

            # spliting the data into X_train,X_test & y_train,y_test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

            logger.info(f'checking the coulmns names X_train & X_test : {self.X_train.columns}')

            logger.info(f'checking the column name X_train :{self.X_train.shape}')


            logger.info(f'Before converting  the coulmns name y_train : {self.y_train}')
            logger.info(f'Before converting the column name y_test : {self.y_test}')

            #Converting the dependent coulmn into int
            self.y_train = self.y_train.map({'Yes':1,'No':0}).astype(int)
            self.y_test = self.y_test.map({'Yes':1,'No':0}).astype(int)

            logger.info(f'After converting  the coulmns name y_train : {self.y_train}')
            logger.info(f'After converting the column name y_test : {self.y_test}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def handling_missing_values(self):
        try:
            logger.info(f'=========== handling missing values=========')
            logger.info(f'Before handling missing values : {self.df.isnull().sum()}')

            self.X_train,self.X_test = handling_missing_values(self.X_train,self.X_test)

            logger.info(f'After handling missing values : {self.X_train.isnull().sum()}')
            logger.info(f'After handling missing values : {self.X_test.isnull().sum()}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')


    def data_separation(self):
        try:
            logger.info(f'============= Data separation& NUM & CATEGORICAl ==============')
            logger.info(f'Before data separation : {self.X_train.columns}')
            logger.info(f'Before data separation : {self.X_test.columns}')

            self.X_train_nums_cols = self.X_train.select_dtypes(exclude='object')
            self.X_train_cats_cols = self.X_train.select_dtypes(include='object')

            self.X_test_nums_cols = self.X_test.select_dtypes(exclude='object')
            self.X_test_cats_cols = self.X_test.select_dtypes(include='object')


            logger.info(f'After data separation : {self.X_train_nums_cols.columns} \n {self.X_train_nums_cols.shape}')
            logger.info(f'After data separation : {self.X_train_cats_cols.columns} \n {self.X_train_cats_cols.shape}')

            logger.info(f'After data separation : {self.X_test_nums_cols.columns} \n {self.X_test_nums_cols.shape}')
            logger.info(f'After data separation : {self.X_test_cats_cols.columns} \n {self.X_test_cats_cols.shape}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def variable_tran(self):
        try:
            logger.info(f'Before apply Train numerical columns and shapes variable Transformation \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'Before apply Test numerical columns and shapes variable Transformation \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')

            self.X_train_nums_cols, self.X_test_nums_cols = Var_Transforamtion(self.X_train_nums_cols,self.X_test_nums_cols)

            logger.info(f'After apply Train numerical columns and shapes variable Transformation \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'After apply Test numerical columns and shapes variable Transformation \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')
                # code in main file var trans


    def outliers(self):
        try:
            logger.info(f'Before apply Train numerical columns and shapes outlier \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'Before apply Test numerical columns and shapes oulier \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')

            self.X_train_nums_cols, self.X_test_nums_cols = outlier(self.X_train_nums_cols, self.X_test_nums_cols)

            logger.info(f'After apply Train numerical columns and shapes outlier \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'After apply Test numerical columns and shapes outlier \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def feature_selection(self):
        try:
            logger.info(f'Before applying Feature Selection Train numerical columns and shapes  \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'Beffore applying Feature Selection  Test numerical columns and shapes  \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')

            self.X_train_nums_cols, self.X_test_nums_cols = filter_method(self.X_train_nums_cols, self.X_test_nums_cols, self.y_train, self.y_test)

            logger.info(f'After applying Feature Selection Train numerical columns and shapes  \n : {self.X_train_nums_cols.columns} : {self.X_train_nums_cols.shape}')
            logger.info(f'After applying Feature Selection  Test numerical columns and shapes  \n : {self.X_test_nums_cols.columns} : {self.X_test_nums_cols.shape}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def Categorical_Numerical(self):
        try:

            self.X_train_cats_cols, self.X_test_cats_cols = Cat_to_Num(self.X_train_cats_cols,
                                                                           self.X_test_cats_cols) # transform data is present in self.X_train_cats_cols, self.X_test_cats_cols

            self.X_train_nums_cols.reset_index(drop=True, inplace=True)# matching the indexs of the  num cols and cat cols to 0,1,2,....
            self.X_train_cats_cols.reset_index(drop=True, inplace=True)
            self.X_test_nums_cols.reset_index(drop=True, inplace=True)
            self.X_test_cats_cols.reset_index(drop=True, inplace=True)

            self.Training_data = pd.concat([self.X_train_nums_cols, self.X_train_cats_cols], axis=1)
            self.Testing_data = pd.concat([self.X_test_nums_cols, self.X_test_cats_cols], axis=1)

            logger.info(f'Total Training Data is : {self.Training_data.columns} \n and it shape : {self.Training_data.shape} ')
            logger.info(f'Total Testing Data is : {self.Testing_data.columns} \n and it shape : {self.Testing_data.shape}')


        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def balancing_data(self):
       try:
           logger.info(f'========================= Data Balancing ================================')

           logger.info(f'Before Balancing the data {self.Training_data.columns} and it is shape {self.Training_data.shape}')
           self.Training_data_bal_up, self.y_train_bal_up = balance_data(self.Training_data, self.Testing_data, self.y_train,self.y_test)
           logger.info(f'After balancing the data {self.Training_data_bal_up.columns} and it is shape : {self.Training_data_bal_up.shape}')
           logger.info(f' After balancing the data {self.y_train_bal_up.shape}')


       except Exception as e:
           er_type, er_msg, er_line = sys.exc_info()
           logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def feature_scaling(self):
        try:
            print(self.Training_data_bal_up)
            sc = StandardScaler()
            sc.fit(self.Training_data_bal_up)
            self.Training_data_bal_up_scaled = sc.transform(self.Training_data_bal_up)
            self.Testing_data_scaled = sc.transform(self.Testing_data)
            print(self.Training_data_bal_up_scaled)
            #common(self.Training_data_bal_up_scaled,self.y_train_bal_up,self.Testing_data_scaled,self.y_test)
            logger.info("===Training Logistic Regression=====")
            self.reg = LogisticRegression(C=2.0, class_weight='balanced', dual=False, penalty='l2')
            self.reg.fit(self.Training_data_bal_up_scaled, self.y_train_bal_up)
            self.y_test_predictions = self.reg.predict(self.Testing_data_scaled)
            logger.info(f"Test data Accuracy : {accuracy_score(self.y_test, self.y_test_predictions)}")
            logger.info(f"Confusion Matrix : {confusion_matrix(self.y_test, self.y_test_predictions)}")
            logger.info(f"classification report : {classification_report(self.y_test, self.y_test_predictions)}")
            logger.info(f"====Saving the Scaled and Logistic Regression Model into Pickle File===========")
            with open("standard_scaler.pkl", "wb") as f:
                pickle.dump(sc, f)

            with open("CHURN_APP.pkl", "wb") as f1:
                pickle.dump(self.reg, f1)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

    def hyper_tuning(self):
        try:
            logger.info(f'========================= Hyperparameter Tuning ================================')

            self.best_params = hyper_parameter(
            self.Training_data_bal_up_scaled,
            self.Testing_data_scaled,
            self.y_train_bal_up,
            self.y_test
            )

            logger.info(f'Hyperparameter Tuning completed.')
            logger.info(f'Best Params : {self.best_params}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')

if __name__ == '__main__':
    obj = CHURN("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    obj.handling_missing_values()
    obj.data_separation()
    obj.variable_tran()
    obj.outliers()
    obj.feature_selection()
    obj.Categorical_Numerical()
    obj.balancing_data()
    obj.feature_scaling()
    #obj.hyper_tuning()









