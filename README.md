# Project: Puzzle Difficulty Classification

## Overview

This project aims to classify the difficulty levels of various logic puzzles (such as Sudoku) based on the rule set provided for each puzzle. The difficulty classification will be achieved through machine learning models trained on textual data of puzzle rule sets, scraped from the `logicmastergermany.de` website. The goal is to build a system that can accurately predict the difficulty of unseen puzzle rules.

## Project Workflow

1. **Data Acquisition**:
   - Scrape raw text data containing puzzle rule sets and corresponding difficulty levels from [logicmastergermany.de](https://logicmastergermany.de).
   - Store the scraped data in an SQLite database for easy access and management.

2. **Data Cleaning**:
   - Use the `nltk` toolkit to preprocess and clean the scraped textual data.
   - Tokenize, remove stop words, and normalize the text for better analysis and modeling.

3. **Exploratory Data Analysis (EDA)**:
   - Perform EDA using `spacy` to gain insights into the data.
   - Visualize text features and analyze word distributions, difficulty level distribution, and common terms within difficulty classes.

4. **Data Preparation**:
   - Use `scikit-learn` for vectorizing the cleaned text data and splitting it into training and test sets.
   - Implement additional feature engineering techniques if necessary to enhance the data for model training.

5. **Model Building**:
   - Build a machine learning model using `PyTorch`. The model architecture will be based on a 1D Convolutional Neural Network (CNN) designed to process textual data.
   - Train the model on the processed rule set data and evaluate its performance in classifying puzzle difficulty.

## Project Structure

The project is structured to be managed using `poetry` for dependency management and reproducibility. The main functionalities are encapsulated in the `./src/main.py` script with several command-line flags for different stages of the project.

### Dependencies

The project uses `poetry` for dependency management. Ensure that `poetry` is installed on your machine before proceeding.

```bash
# To install project dependencies
poetry install
```

### Running the Project

The `./src/main.py` script can be executed with the following command-line flags to run different stages of the project:

- **Data Scraping**:  
  Scrape rule set data from `logicmastergermany.de` and save it to an SQLite database.

  ```bash
  poetry run python ./src/main.py --scrape
  ```

- **Data Analysis**:  
  Run a Jupyter Notebook for exploratory data analysis on the cleaned data.

  ```bash
  poetry run python ./src/main.py --analysis
  ```

- **Model Training**:  
  Train the 1D CNN model on the prepared data.

  ```bash
  poetry run python ./src/main.py --train
  ```

- **Predicting Puzzle Difficulty**:  
  Use the trained model to predict the difficulty of new puzzle rule sets.

  ```bash
  poetry run python ./src/main.py --predict "<text data>"
  ```


