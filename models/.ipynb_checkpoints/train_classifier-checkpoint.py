# import libraries
import sys
import re
import pickle

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    """Load data from database"""
    engine = create_engine(f'sqlite:///{database_filepath}')
    engine.connect()
    df = pd.read_sql('SELECT * FROM messages', con=engine)
    X = df.loc[:, "message"]
    Y = df.iloc[:, 4:]
    
    # the output column "related" has 3 possible outcomes (0, 1, 2)
    Y.replace(2, 1, inplace=True)
    # the output column "child_alone" has just 1 possible outcomes (0)
    Y.drop('child_alone', axis=1, inplace=True)
    category_names = category_names.columns
    return X.values, Y.values, category_names


def tokenize(text):
    return word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' ', text))


def build_model():
    """Build the ML model"""
    
    cls_model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100,
                               n_jobs=-1,
                               random_state=123,
                               class_weight='balanced',
                               max_depth=20),
        n_jobs=None)
    
    pipeline = Pipeline([
        ('cvec', HashingVectorizer(tokenizer=tokenize,
                                   stop_words=stopwords.words('english'),
                                   token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('cls', cls_model)
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Print evaluation metrics for test data"""
    
    Y_pred = pipeline.predict(X_test)
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
            np.round(precision_score(Y_test.iloc[:, i], Y_pred.iloc[:, i]), 2),
            np.round(recall_score(Y_test.iloc[:, i], Y_pred.iloc[:, i]), 2),
            np.round(f1_score(Y_test.iloc[:, i], Y_pred.iloc[:, i]), 2)
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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()