"""
Use PySparks K Means package to cluster players into archetypes 
e.g. Contact, Power, OB Specialists, Non-Performer
"""
import time


import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import pyspark.sql.functions as f

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType


data_path = 'Data/batting_2014_2024.csv'

def plot_roc_curve_spark(lr_model, test_data_scaled):
    
    # Evaluate the model on the test data
    test_summary = lr_model.evaluate(test_data_scaled)

    # Get ROC DataFrame and convert to Pandas
    roc_test_df = test_summary.roc
    roc_test_pd = roc_test_df.toPandas()

    # Plot ROC Curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.plot(roc_test_pd['FPR'], roc_test_pd['TPR'], label='Test ROC Curve')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Test Data)')
    plt.legend()
    plt.show()

    # Print Area Under ROC
    auc_test = test_summary.areaUnderROC
    print(f"Area Under ROC (Test Data): {auc_test}")


def plot_classification(precision, recall, f1_score):
    # Bar chart for precision, recall, and f1 score
    fig = go.figure()

    fig.add_trace(go.Bar(
        x=['Precision', 'Recall', 'f1-Score'],
        y=[precision, recall, f1_score],
        text=[f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}"],
        textposition='auto',
        name='Metrics',
        marker_color=['blue', 'green', 'red']
    ))

    fig.update_layout(
        title="Precision, Recall, f1-Score",
        xaxis_title="Metric",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )

    fig.show()

def plot_clustering(predictions):
    
    # Convert the Spark Dataframe to a Pandas Dataframe for plotting
    pandas_df = predictions.select(
        "Player", "OBP", "SLG", "cluster_label").toPandas()

    # Create a scatter plot
    fig = px.scatter(
        pandas_df,
        x="OBP",
        y="SLG",
        color="cluster_label",
        hover_name="Player",
        labels={"OBP": "On-Base Percentage", "SLG": "Slugging Percentage"},
        title="Player Archetypes Clustering",
        height=600,
        width=900
    )

    # Show the interactive plot
    fig.show()

def squared_distance_to_center(features, prediction):
    """
    Define a UDf to calculate the squared distance between a point and its cluster center 
    """
    center = cluster_centers[prediction]
    return float(np.sum((np.array(features) - center) ** 2))

if __name__ == "__main__":
    # Create Spark session
    spark = SparkSession.builder.appName(
        "Player Archetype Clustering").getOrCreate()

    # Read data from CSV file
    data = spark.read.csv(data_path, header=True, inferSchema=True)

    # Select features for clustering and classification
    feature_columns = ["BA", "OBP", "SLG", "HR", "SO", "RBI"]
    data = data.na.drop(subset=feature_columns)

    # Creating binary feature if a player has 100 hits or more
    data = data.withColumn("hit", f.when(f.col("H") > 100, 1).otherwise(0))

    assembler = VectorAssembler(
        inputCols=feature_columns, outputCol="features")

    # Transform data to include the features
    data_with_features = assembler.transform(data).cache()
    
    # Train-test split (e.g., 70% train, 30% test)
    train_data, test_data = data_with_features.randomSplit(
        [0.7, 0.3], seed=1234)
    train_data.cache()
    test_data.cache()
    
    # Apply K-Means clustering on the training data
    kmeans = KMeans(k=4, seed=1)
    kmeans_model = kmeans.fit(train_data)

    train_predictions = kmeans_model.transform(train_data)
    test_predictions = kmeans_model.transform(test_data)

    # Rename KMeans 'prediction' column to avoid conflicts with Logistic Regression
    train_predictions = train_predictions.withColumnRenamed(
        "prediction", "cluster_label").cache()
    test_predictions = test_predictions.withColumnRenamed(
        "prediction", "cluster_label").cache()

    # Add cluster labels as a feature
    assembler_with_cluster = VectorAssembler(
        inputCols=feature_columns + ["cluster_label"], outputCol="features_lr"
    )

    train_data_with_clusters = assembler_with_cluster.transform(
        train_predictions).cache()
    test_data_with_clusters = assembler_with_cluster.transform(
        test_predictions).cache()

    # Get the cluster centers
    cluster_centers = np.array(kmeans_model.clusterCenters())

    # UDf
    squared_distance_udf = f.udf(squared_distance_to_center, DoubleType())

    # Calculate the squared distances
    train_data_with_clusters = train_data_with_clusters.withColumn(
        "squared_distance", squared_distance_udf(
            f.col("features"), f.col("cluster_label"))
    )

    # Sum the squared distances to compute WCSS
    wcss = train_data_with_clusters.agg(
        {"squared_distance": "sum"}).collect()[0][0]
    print(f"Within-Cluster Sum of Squares (WCSS): {wcss}")

    # Train logistic regression
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(train_data_with_clusters)
    train_data_scaled = scaler_model.transform(train_data_with_clusters)
    test_data_scaled = scaler_model.transform(test_data_with_clusters)
    lr = LogisticRegression(labelCol="hit", featuresCol="features_lr")
    lr_model = lr.fit(train_data_with_clusters)

    # Make predictions on the test data
    lr_predictions = lr_model.transform(test_data_with_clusters)

    # Precision
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="hit", predictionCol="prediction", metricName="precisionByLabel")
    precision = precision_evaluator.evaluate(lr_predictions)
    print(f"Precision: {precision}")

    # Recall
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="hit", predictionCol="prediction", metricName="recallByLabel")
    recall = recall_evaluator.evaluate(lr_predictions)
    print(f"Recall: {recall}")

    # f1-Score
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="hit", predictionCol="prediction", metricName="f1")
    f1_score = f1_evaluator.evaluate(lr_predictions)
    print(f"f1-Score: {f1_score}")

    evaluator = BinaryClassificationEvaluator(
        labelCol="hit", metricName="areaUnderROC")
    roc_auc = evaluator.evaluate(lr_predictions)
    print(f"Test ROC AUC: {roc_auc}")

    # plot the KMeans
    plot_clustering(train_data_with_clusters)
    
    # plot metrics
    # plot_classification(precision, recall, f1_score)
    
    # plot curve
    plot_roc_curve_spark(lr_model, test_data_scaled)
