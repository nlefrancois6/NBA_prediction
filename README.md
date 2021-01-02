# NBA_prediction
Project to predict the outcomes of NBA games using scikit-learn classification &amp; regression models and calculate expected winnings from betting on the moneyline

Method Overview:
  Take team stats for both teams, averaged over the last n games, and use a classification model to predict the winner. Calculate the gain/loss if I were to bet on
  the moneyline for that game according to the classification prediction. Use a regression model to predict the gain/loss from betting on the moneyline for a game
  given the classification's confidence and the moneyline odds. Only place a bet on a game where the regression predicts a gain/loss above a specified threshold.

Scripts for the data pipeline:
  odds_data_processing.py: Take the odds data and get the columns we care about, save it to a csv in format 'nba_odds_1617.csv'
  
  NBApred_preprocessing.py: Take the stats data and the formatted odds data and combine them into one dataframe containing all of
    the info needed to train & test our layered model. Compute the averaged stats for each team going into each game, using a rolling
    average over a specified number of games, n. Save this dataframe to a csv in format 'pre-processedData_n5.csv'
    
 NBApred_runModels.py: Take the pre-processed data, select the features to use, train and test the layered classification-regression
  model on that data. Can make plots of the betting outcomes on the testing and validation data.
  
 NBApred_hyperparam_gridsearch.py: Take the pre-processed data and perform a gridsearch of hyperparameter performance for the
  layered model. Save a dataframe containing the parameter grid and the profit from each grid point to a csv in format 'hyperparam_gridSearch_coarse.csv'
 
 gridSearch_analysis.py: Take the grid search output dataframe and visualize the model performance, plotting the profit as a function of each hyperparameter
  in the grid.
  
 NBApredFuncs.py: Contains all of the helper functions used in the data pipeline
 
To-Do:
  Need to implement web scraping so that we can get up-to-date NBA stats and odds every week and seamlessly load it into the pipeline.
  
  Could try predicting score instead of winner, then we could try to predict spread, O/U, etc and see what we can be most effective at. This would probably
  be done by replacing the classification step with a regression for the score (maybe one for each team). We would need to find odds data for spread and O/U
  in order to make this useful.
