# -*- coding: utf-8 -*-

import logging

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S', filemode='a')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
import random

class Recommender():
    def __init__(self):
        self.r_model = None
        self.users_list = [i.strip().split("::") for i in open('data/users.dat', 'r',encoding='latin-1').readlines()]
        self.ratings_df = None
        self.movies_list = [i.strip().split("::") for i in open('data/movies.dat', 'r',encoding='latin-1').readlines()]
        self.movies_df = pd.DataFrame(self.movies_list, columns = ['MovieID', 'Title', 'Genres'])
        self.movies_df['MovieID'] = self.movies_df['MovieID'].apply(pd.to_numeric)
        pass

    def train(self,path_dataset):
        self.ratings_df = pd.read_csv(path_dataset, sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'],encoding='latin-1')
        R_df = self.ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating')
        user_ratings_mean=np.array(R_df.mean(axis=1))
        R_demeaned=R_df.sub(R_df.mean(axis=1), axis=0)
        R_demeaned=R_demeaned.fillna(0).values
        U, sigma, Vt = svds(R_demeaned, k = 50)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
        preds_df.to_csv('data/baseline_model.csv', index=False)
        self.r_model = preds_df

    def predict(self,userID,num_recommendations=5):
        self.warmup()
        predictions_df=self.r_model
        movies_df=self.movies_df
        original_ratings_df=self.ratings_df


        user_row_number = userID - 1
        sorted_user_predictions_id = predictions_df.iloc[user_row_number].sort_values(ascending=False).to_frame()
        sorted_user_predictions_id['MovieID'] = sorted_user_predictions_id.index
        sorted_user_predictions_id = sorted_user_predictions_id.rename(columns = {user_row_number:'Predictions'})
        sorted_user_predictions = pd.DataFrame(predictions_df.iloc[user_row_number].sort_values(ascending=False).to_frame(), columns = ['MovieID','Predictions'])
        user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
        user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                    )

        recommendations = (movies_df[movies_df['MovieID'].isin(user_full['MovieID'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'MovieID',
               right_on = 'MovieID').
         rename(columns = {user_row_number: 'Predictions'}).
         sort_values('Predictions', ascending = False).
                       iloc[:num_recommendations, :-1]
                      )
        logging.info(f"Recommendations are implemented!!!")
        return recommendations['MovieID'].values,sorted_user_predictions_id['Predictions'].values[:num_recommendations]

    def warmup(self, model_name: str='baseline_model.csv'):
        self.r_model = pd.read_csv('data/'+ model_name,encoding='latin-1')
        self.users_list = [i.strip().split("::") for i in open('data/users.dat', 'r',encoding='latin-1').readlines()]
        self.movies_list = [i.strip().split("::") for i in open('data/movies.dat', 'r',encoding='latin-1').readlines()]
        self.movies_df = pd.DataFrame(self.movies_list, columns = ['MovieID', 'Title', 'Genres'])
        self.movies_df['MovieID'] = self.movies_df['MovieID'].apply(pd.to_numeric)
        self.ratings_df = pd.read_csv('data/ratings_train.dat', sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'],encoding='latin-1')
        logging.info(f"Model: {model_name} successfully loaded!!!")
        pass

    def find_similar():
        pass




