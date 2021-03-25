from flask import Flask, render_template, request
from Autoencoder import Autoencoder
import tensorflow.keras as tfk
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import torch
app = Flask(__name__)

books_arr = []
to_predict = []
ratings_dict = {}
readSae = Autoencoder()

def my_rmse(y_true, y_pred):
    nonzero = (y_true != 0)
    nonzero = tf.dtypes.cast(nonzero, tf.float32) 
    y_new = y_pred * nonzero
   
    error = y_true-y_new
    sqr_error = K.square(error)
    mean_sqr_error = K.mean(sqr_error)
    sqrt_mean_sqr_error = K.sqrt(mean_sqr_error)
    
    return sqrt_mean_sqr_error

@app.route("/", methods=['POST', 'GET'])
def main():
    ratings_given = []
    pred_books = []

    if request.method == 'POST':
        key = request.form.get('bookName')
        yes_no = str(request.form.get('yes_no'))
        if (request.form.get('rating') != ''):
            ratings_dict[key] = float(request.form.get('rating'))

            for key in ratings_dict.keys():
                if ratings_dict[key] > 0:
                    ratings_given.append(key + " - " + str(ratings_dict[key]))
                    idx = books_arr.index(key)
                    print(idx)
                    to_predict[idx] = float(ratings_dict[key])
        
            if (yes_no == 'No'):
                print("Will be predicting for: ")
                for i in range(len(to_predict)):
                    if to_predict[i] > 0:
                        print(i, to_predict[i])
                new_tr = np.reshape(to_predict, (1, 8706))
                pred = readSae.predict(new_tr)
    
                desc_pred = tf.sort(pred, direction='DESCENDING')
                idx = tf.argsort(pred, direction='DESCENDING')
                
                np_idx = idx.numpy()
                np_idx = np_idx[0][:]
                
                print("After prediction, first top idx are: ", np_idx[:20])
                
                for i in range(desc_pred.shape[1]):
                    if (len(pred_books) == 10):
                        break
                    elif (books_arr[np_idx[i]] not in ratings_dict.keys()):
                        pred_books.append(books_arr[np_idx[i]])
                        
                print("Pred_books: ", pred_books)
        
    return render_template('index.html', books=books_arr, ratings=ratings_given, suggested=pred_books)


if __name__ == "__main__":   
    readSae = tfk.models.load_model("tensorflowSAE", custom_objects={'my_rmse': my_rmse})
    
    books_arr = list(pd.read_csv('eng_books_sorted.csv')['0'])
    to_predict = np.zeros(shape=[len(books_arr)])
                
    app.run()
