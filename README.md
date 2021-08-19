
# Acquiring-Stock-Prediction-Techniques
# Table of Contents
1.	Title of the Project
2.	Description
3.	Languages
4.	Packages
5.	Files to run
6.	Usage
7.	Support
8.	Future work and Enhancements
# Name of the Project
Acquiring Stock Prediction Techniques using Machine Learning Models.

# Description:
MachineLearningStocks is a template project that uses machine learning to provide stock forecasts and is designed to be intuitive and expandable. My objective is that this project will help you grasp the overall process of utilising machine learning to forecast stock movements, as well as appreciating some of the nuances.
I'll use pandas to clean and prepare a dataset of historical stock prices and fundamentals, and then we'll use a scikit-learn classifier to identify the association between stock fundamentals and stock prices. Before making predictions based on current data, I ran a basic backtest.
# Languages:
This project makes use of Python 3.6 and the pandas and scikit-learn data science modules.
I have used python Language for this project and to run the StockPrediction.py file you need to have a Python or Google colab account or jupyter notebook in your system.
For this project, we need three datasets:
Historical stock fundamentals
Historical stock prices
Historical S&P500 prices
# Packages:
pip install -r requirements.txt <br /> 
python download_historical_prices.py <br />
python parsing_keystats.py <br />
python backtesting.py <br />
python current_data.py <br />
pytest -v <br />
python stock_prediction.py <br />
# Files to run
The main file to run this project is available here StockPrediction.py and evaluation.py and the dataset required to run the code is available as forward_sample.csv, keystats.csv
The dataset used for this project can also be found here along with the original database file from where it originated.
Steps involved in running the project : <br />
Creating the training dataset , Preprocessing historical price data, Parsing, Backtesting.

# Usage
•	Open the Google colab or Jupyter notebook or pycharm <br />
•	Load the file StockPrediction.py and evaluation.py to the notebook. <br />
•	Download the dataset forward_sample.csv, keystats.csvfile to the notebook and adjust the current path of the dataset in the StockPrediction.py and evaluation.py files. <br />
•	Make sure all the resources and packages are included and run the code. <br />
•	Upon running the code, you will get output with the accuracy. <br />
I used the Random Forest Classifier  and SVM Model . Tested the stock prediction for both the models.Here you go the test results for both models:  <br />

# Support
You can reach out to me at one of the following places: 
•	Via Lakehead email : mgurijal@lakeheadu.ca  <br />
•	via Phone : +1 (807) 357 6308 <br />
# Future work and enhancements
•	There are still a few more areas to test in the dataset. <br />
•	Instead of using Random Forest Classifier I used Svm too, I can try other classifiers like linear regression to improve the accuracy of results. <br />



