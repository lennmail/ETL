# ETL

This program parses CSV files into a Matrix format that can be used for machine learning. The program is able to calculate mean and std of the data, rendering it able to normalize input data via the z-normalization. I have implemented support for splitting the data into training and testing sets. 

Note: The target variable should be the rightmost column of the CSV file. Otherwise the code has to be adjusted as necessary.
