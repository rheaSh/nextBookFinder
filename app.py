from flask import Flask, render_template, request
from SAE import SAE
import pandas as pd
import pickle
import torch
app = Flask(__name__)

books_arr = []
to_predict = []
pred_books = []
ratings_dict = {}
readSae = SAE()


@app.route("/", methods=['POST', 'GET'])
def main():
    ratings_given = []
    
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
                pred = readSae(to_predict)
                desc_pred, idx = torch.sort(pred, descending=True)
                print(desc_pred[:20], "\n", idx, max(idx), min(idx))
                
                for i in range(len(desc_pred)):
                    if (len(pred_books) == 10):
                        break
                    elif (books_arr[idx[i]] not in ratings_dict.keys()):
                        pred_books.append(books_arr[idx[i]])
        
    return render_template('index.html', books=books_arr, ratings=ratings_given, suggested=pred_books)


if __name__ == "__main__":
    with open('trainedSae','rb') as inf:
        readSae = pickle.load(inf, encoding='latin1')
    
    books_arr = list(pd.read_csv('eng_books_sorted.csv')['0'])
    to_predict = torch.zeros([len(books_arr)], dtype=torch.float32)
                
    app.run()
