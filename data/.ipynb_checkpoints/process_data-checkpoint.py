import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load messages and categories.
    
    Parameters
    ------------
    messages_filepath : str
    categories_filepath : str
    
    Returns
    ------------
    pandas.DataFrame
        The joint dataset of Messages and Categories
    """
    
    messages = pd.read_csv(f"{messages_filepath}/messages.csv")
    categories = pd.read_csv(f"{messages_filepath}/categories.csv")
    df = messages.drop_duplicates(subset='id').merge(categories.drop_duplicates(subset='id'), on='id')
    return df

def clean_data(df):
    "Perform the cleaning operations of df"
    
    # create separate columns for each categories values
    categories = df.categories.str.split(';', expand=True)
    
    # rename the columns of `categories`
    row = categories.iloc[0, :]
    category_colnames = [cat[:-2] for cat in row]
    categories.columns = category_colnames
    
    # convert each value as {0; 1}
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int64')
    
    # substitute the original categories column in df with the new columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
        
    return df


def save_data(df, database_filename):
    """Save the table df in a sqlite database"""
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()