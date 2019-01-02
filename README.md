# IMDB-sentiment-analysis
This is a reusable NLP classification module built initially to classify IMBD reviews in [this kaggle dataset](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset) with non-deep learning classifiers using embedding.

### It features:
- Pre processing using: Stop word removal, stemming and lemmatization.
- Vectorization using: count vectorizer, tfidf and Gensim Word2Vec.
- Classification with a number of non deep learning classifiers and ensembled classifiers (extendable for deep learning as well)

**The `Main.py` file does a grid search for using different classifiers, preprocessing techniques, vectorization methods and reports the accuracies for each with bar plots.**
