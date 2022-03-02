import os

from sklearn.model_selection import train_test_split

from data import process_data
from models.train_classifier import load_data, tokenize, build_model

messages_filepath = '../data/disaster_messages.csv'
categories_filepath = '../data/disaster_categories.csv'
database_filename = 'test_db.db'

def test_load_data_X_at_least_one_row():
    # necessary steps to prepare the data for this test
    df = process_data.load_data(messages_filepath, categories_filepath)
    df = process_data.clean_data(df)
    process_data.save_data(df, database_filename)
    # test
    try:
        X, Y, categories = load_data(database_filename)
        assert X.shape[0] > 0
    finally:
        os.remove(database_filename)

def test_load_data_Y_at_least_one_row():
    # necessary steps to prepare the data for this test
    df = process_data.load_data(messages_filepath, categories_filepath)
    df = process_data.clean_data(df)
    process_data.save_data(df, database_filename)
    # test
    try:
        X, Y, categories = load_data(database_filename)
        assert Y.shape[0] > 0
    finally:
        os.remove(database_filename)

def test_load_data_CAT_at_least_one_row():
    # necessary steps to prepare the data for this test
    df = process_data.load_data(messages_filepath, categories_filepath)
    df = process_data.clean_data(df)
    process_data.save_data(df, database_filename)
    # test
    try:
        X, Y, categories = load_data(database_filename)
        assert len(categories) > 0
    finally:
        os.remove(database_filename)

def test_tokenize_sentence1():
    assert tokenize('My uncle is red!') == ['uncle', 'red']

def test_tokenize_sentence2():
    assert tokenize(
    """Everything you have ever thought is to eat until the end"""
    ) == ['everything', 'ever', 'think', 'eat', 'end']

def test_tokenize_sentence3():
    assert tokenize(
    """You must be very tired to make me wait for half an hour"""
    ) == ['must', 'tire', 'make', 'wait', 'half', 'hour']

def NO_test_build_model():
    ### This test case has been turned off because with a small
    ### subset of data many labels turns to has just one value
    ### and the training phase crushes.
    ### With complete data, instead, it would be too much long.
    
    # necessary steps to prepare the data for this test
    df = process_data.load_data(messages_filepath, categories_filepath)
    df = process_data.clean_data(df)
    process_data.save_data(df, database_filename)
    X, Y, category_names = load_data(database_filename)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01)
    # test
    try:
        model = build_model()
        model.fit(X_test, Y_test)
        assert True
    finally:
        os.remove(database_filename)
