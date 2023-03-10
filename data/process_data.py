import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads the data from messages and categories filepath
    
    returns merged dataframe
    '''
    df_messages = pd.read_csv(messages_filepath)
    df_cats = pd.read_csv(categories_filepath)
    df_messages = df_messages.merge(df_cats, how='left', on=['id'])
    return df_messages


def clean_data(df):
    '''
    Cleans df - drops duplicates, na and performs string manipulation
    
    returns df
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.transform(lambda x: x[:-2]).values
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        if column == 'related':
            categories[column] = categories[column].replace(2,1)
                      
    df.drop(columns=['categories'], inplace=True)

    df = pd.concat([df, categories], axis = 1)

    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Save df to sqlite database
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')  


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
