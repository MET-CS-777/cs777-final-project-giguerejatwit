# final-project DPAC: Determining Player Archetypes using Clustering 
student: Jake Giguere
email: giguere@bu.edu

## Overview
Can we cluster players based on their performance data to predict the likelihood of certain expectations and game outcomes? 
This analysis can provide insights to which player types contribute most to the teams success and help teams identify performance trends. I have been working on an architecture for the past year called PLONN it is an MLP that will be transformed into a RL agent this MLB-offseason where I can learn trends in players that are more likely to have good game performances.

# How to run

To run the program, make sure the required pacakges are installed from ```requirements.txt```
From the PROJECT ROOT DIRECTORY ```CS777-Final-Project-Giguerejatwit/``` run ```spark-submit kmeans-lr-pyspark.py```

The data is already stored as a CSV file so there is no need to run ```mlb-statistics.py``` in the weird case where the data is missing you can easily run ```python3 mlb-statistics.py``` to fetch the data from baseball reference.

## ML Models
I will start by using K-Means Clustering to group players based on their statistics, then follow up with classification using logistic regression in pySpark to predict hits. Clustering can help identify player archetypes such as contact hitters, power hitters, etc. After clustering, classification models can assess how these archetypes correlate with outcomes.

## Expected Outcomes
I expect to be able to determine which cluster contributes more to a specific outcome such as hitting and determine which player types are associated with standout individual performance 

## Evaluation
The approach is the to use a within-cluster sum of squares (WCSS) for clustering quality and Accuracy score, precision recall, F1 score for classification performance. Depending on how I clean the data, I’m sure we can expect at least a reasonable accuracy, leveraging my familiarity with baseball metrics

## References
[Baseball Reference](https://www.baseball-reference.com/)