# import libraries
import sys
import re
import pickle
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")

def load_data(database_filepath):
    """Load data from database"""
    engine = create_engine(f'sqlite:///{database_filepath}')
    engine.connect()
    df = pd.read_sql('SELECT * FROM messages', con=engine)
    X = df.loc[:, "message"]
    Y = df.iloc[:, 4:]

    category_names = Y.columns
    return X.values, Y.values, category_names


def tokenize(text, stop_words=stopwords.words('english')):
    """Transform a text into a list of clean tokens"""

    tokens = word_tokenize(re.sub(r'[^a-z0-9]', ' ', text.lower()))
    tokens = [tok for tok in tokens if tok not in stop_words]

    lemmatizer = WordNetLemmatizer()

    fun = lambda tok: lemmatizer.lemmatize(
                        lemmatizer.lemmatize(tok, pos='n'),
                        pos='v')

    return list(map(fun, tokens))


def build_model():
    """Build the ML model"""

    # Classification model
    cls_model = MultiOutputClassifier(
        estimator=SGDClassifier(random_state=123,
                                class_weight='balanced')
    )

    pipeline = Pipeline([
        ('tfidfvec', TfidfVectorizer(tokenizer=tokenize,
                                     stop_words=None,
                                     lowercase=False,
                                     token_pattern=None)),
        ('tsvd', TruncatedSVD (random_state=123)),
        ('cls', cls_model)
    ])

    # Perform Randomized Search CV to tune the model
    parameters = {
        'tfidfvec__smooth_idf': [True, False],
        'tfidfvec__ngram_range': [(1, 1), (1, 2)],
        'tsvd__n_components': [300, 400, 450],
        'cls__estimator__alpha': [0.01, 0.001, 0.0001],
        'cls__estimator__loss': ['hinge', 'log'],
        'cls__estimator__l1_ratio': np.arange(0, 1.05, 0.05),
        'cls__estimator__validation_fraction': np.arange(0.3, 0.5, 0.05)
    }

    cv = RandomizedSearchCV(
        pipeline,
        param_distributions=parameters,
        n_iter=10,
        # for the purpose of this model, it's somewhat better to have a model
        # with a high recall, not missing to forward important messages to
        # organizations. Many messages will require a further manual check,
        # but, to me at least, it's better to have a model with a high recall.
        # I decided here to use a f1 score metric to measure the model
        # performance, whitch is an armonic mean between precision and recall,
        # but also just the recall would be a good choice.
        # We could also consider the accuracy score, considering we are using
        # a model with 'class_weight = balanced' which takes the class
        # inbalance into account, but I haven't tried it.
        scoring='f1_weighted',
        cv=5,
        verbose=3,
        random_state=123,
        n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Print evaluation metrics for test data"""

    Y_pred = model.predict(X_test)
    print('-'*50)
    print('GLOBAL MODEL EVALUATION:')
    print('\tTest Precision weighted:',
          np.round(precision_score(Y_test, Y_pred, average='weighted'), 2))
    print('\tTest Recall weighted:',
          np.round(recall_score(Y_test, Y_pred, average='weighted'), 2))
    print('\tTest F1 weighted:',
          np.round(f1_score(Y_test, Y_pred, average='weighted'), 2))
    print('\t\tTest Precision macro:',
          np.round(precision_score(Y_test, Y_pred, average='macro'), 2))
    print('\t\tTest Recall macro:',
          np.round(recall_score(Y_test, Y_pred, average='macro'), 2))
    print('\t\tTest F1 macro:',
          np.round(f1_score(Y_test, Y_pred, average='macro'), 2))
    print('-'*50)
    print('\nSINGLE FEATURES EVALUATION:')
    for i, col in enumerate(category_names):
        print('\tFEATURE "{}": (precision: {}), (recall: {}), (f1-score: {})'.format(
            col,
            np.round(precision_score(Y_test[:, i], Y_pred[:, i]), 2),
            np.round(recall_score(Y_test[:, i], Y_pred[:, i]), 2),
            np.round(f1_score(Y_test[:, i], Y_pred[:, i]), 2)
        ))


def save_model(model, model_filepath):
    """Save model as pickle file"""
    pickle.dump(model, open( model_filepath, "wb" ) )


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/messages_db.db classifier.pkl')


if __name__ == '__main__':
    main()
