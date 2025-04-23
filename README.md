# Building a Feature Engineering Pipeline with Apache Airflow

This tutorial (**UNDER CONSTRUCTION**) walks you through the process of designing and implementing an ETL pipeline for feature engineering, leveraging weather data as a practical example. Using **Apache Airflow** for orchestration and `sklearn` for feature manipulation, the pipeline extracts data from an API, transforms it into meaningful features, and loads the results into **MongoDB**. An optional advanced section introduces **Feast** as a feature store for machine learning workflows.

---

## Introduction

The goal of this tutorial is to teach you how to build an automated ETL pipeline with a focus on feature engineering. By the end, you’ll know how to:
- Retrieve weather data from the **OpenWeatherMap API**.
- Transform raw data into engineered features using techniques like `sklearn`s polynomial features and one-hot encoding.
- Store the results in a database or feature store.
- Automate and schedule the pipeline using **Apache Airflow**.

**Tools You’ll Use**:
- **Apache Airflow**: Workflow orchestration tool to manage the pipeline.
- **scikit-learn**: Python library for feature engineering.
- **MongoDB**: Database to store the processed features.
- **Feast** (optional): Feature store for advanced ML applications.

**Prerequisites**:
- Familiarity with Python and basic machine learning concepts.
- Installed dependencies: Apache Airflow, MongoDB, and Python libraries (`requests`, `pandas`, `scikit-learn`, `pymongo`).
- An API key from [OpenWeatherMap](https://openweathermap.org/api).

---

## Tutorial Overview

The tutorial is structured around the three key stages of an ETL pipeline:

### 1. Extract
- Fetch real-time weather data using Airflow’s `SimpleHttpOperator` to query the OpenWeatherMap API.
- Convert the JSON response into a structured format (e.g., pandas DataFrame) for processing.

### 2. Transform
- Perform feature engineering on the weather data:
  - **Polynomial Features**: Create new features like temperature squared to capture non-linear relationships.
  - **One-Hot Encoding**: Encode categorical variables (e.g., weather conditions like "Rain" or "Sunny") into numerical values.
- Select the most informative features by filtering out those with low variance.

### 3. Load
- Save the engineered features into a MongoDB collection for downstream use.
- Optionally, explore storing features in **Feast** instead (see the optional extension below).

---

## Optional Extension: Integrating Feast

For learners interested in machine learning production systems, this section introduces **Feast**, an open-source feature store. You’ll:
- Set up a Feast feature view to define your engineered features.
- Load the transformed weather data into Feast instead of MongoDB.
- Understand how Feast enhances feature management for ML models by improving consistency and scalability.

This optional task is perfect if you want to take your pipeline to the next level and explore tools used in real-world ML deployments.

---

## Getting Started

To follow along with the tutorial:
1. **Install and Configure Airflow**:
   - Set up Apache Airflow and initialize its database.
   - Configure connections (e.g., OpenWeatherMap API key) in the Airflow UI.
2. **Run the Pipeline**:
   - Copy the provided DAG file to your Airflow `dags` folder.
   - Trigger the pipeline manually or schedule it to run automatically.
3. **Check the Output**:
   - Verify the engineered features in your MongoDB database.
   - Optionally, complete the Feast integration and explore the feature store.

Refer to the notebook files in this repository for complete code and step-by-step instructions.

---

## Share your thoughts

Create an issue to share any comments on how to improve this repository.