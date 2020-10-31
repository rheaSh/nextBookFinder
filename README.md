# nextBookFinder
The aim of this web application is to help users find the next novel to read based on their current interests. We take a user’s ratings for some well known books and use that information to predict other novels that they may rate just as highly. This prediction is done by a stacked AutoEncoder that uses the dataset https://www.kaggle.com/zygmunt/goodbooks-10k. This is a Kaggle dataset that uses GoodReads ratings of multiple users for ten thousand popular books.

Setting up:

1. Download or clone the entire project, and open the .ipynb in Jupyter or JupyterLab application. You may run the entire notebook to create your own pickle file trainedSae, or simply use the one already provided. The notebook also contains information about the steps followed to get the autoencoder.

2. Download the dataset mentioned above and place it in the GoodReads_Ratings folder.

3. You may need to install modules for this numpy, pandas, pickle, torch, flask, if you dont have it already. Its suggested to install these in a separate virtual environment before running any code.

4. Run the application by going to the location of the repo, activating the virtualenv and running app.py

          cd path/to/novelFinder
          activate virtualenvname
          python app.py

5. Go to localhost:5000 for the application. Select a book, enter its rating and repeat this as many times as you have books to rate. Once you’re done and want to know what you should be reading next, select “No” under “Do you want to share more ratings?:” and hit Submit. The “Books to read next:” would show you 10 books you can read next.
