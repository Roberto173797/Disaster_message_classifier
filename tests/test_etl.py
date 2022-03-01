import os
from data.process_data import load_data, clean_data, save_data

messages_filepath = '../data/disaster_messages.csv'
categories_filepath = '../data/disaster_categories.csv'
database_filename = 'test_db.db'

def test_load_data_at_least_one_row():
    df = load_data(messages_filepath, categories_filepath)
    assert df.shape[0] > 0

def test_load_data_at_least_one_column():
    df = load_data(messages_filepath, categories_filepath)
    assert df.shape[1] > 0

def test_clean_data_at_least_one_row():
    df = load_data(messages_filepath, categories_filepath)
    df = clean_data(df)
    assert df.shape[1] > 0

def test_clean_data_at_least_one_column():
    df = load_data(messages_filepath, categories_filepath)
    df = clean_data(df)
    assert df.shape[1] > 0

def test_save_data_once():
    df = load_data(messages_filepath, categories_filepath)
    df = clean_data(df)
    save_data(df, database_filename)
    assert os.path.exists(database_filename)
    os.remove(database_filename)

def test_save_data_twice():
    df = load_data(messages_filepath, categories_filepath)
    df = clean_data(df)
    save_data(df, database_filename)
    save_data(df, database_filename)
    assert os.path.exists(database_filename)
    os.remove(database_filename)
