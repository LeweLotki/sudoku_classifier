import argparse

def initialize_parser():

    parser = argparse.ArgumentParser(description="Puzzle Difficulty Classification")
    
    parser.add_argument('--scrape', action='store_true', help='Scrape data from logicmastergermany.de and save it to an SQLite database.')
    parser.add_argument('--analysis', action='store_true', help='Run Jupyter notebook for data analysis.')
    parser.add_argument('--train', action='store_true', help='Train the machine learning model.')
    parser.add_argument('--predict', type=str, help='Predict difficulty of a given puzzle rule set using the trained model.')

    args = parser.parse_args()
    return args, parser


