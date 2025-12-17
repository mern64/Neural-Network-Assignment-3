# Customer Lifestyle Segmentation Analysis — Assignment 3

This repository contains a comprehensive analysis of customer lifestyle data using **K-Means clustering** and **Self-Organizing Maps (SOM)** techniques. The project includes data preprocessing, exploratory data analysis, clustering algorithms, visualization, and performance metrics.

---

## 📁 Project Structure
```text
ASSIGNMENT 3/
├── .venv/                         # Virtual environment
├── 📄 k-Means_analysis.py          # K-Means clustering implementation
├── 📄 k-SOM_Analysis.py            # Self-Organizing Map (SOM) analysis
├── 📄 kMeansClusteringVisualization.py # Visualization script for results
├── 📊 STINK3014_Assignment03_Customer_Lifestyle.csv  # Main dataset
├── 📈 correlation_heatmap.png      # Correlation matrix visualization
├── 📈 k_means_elbow_method.png     # Elbow method plot (optimal K)
├── 📈 k_means_elbow_plot.png       # Alternative elbow plot
├── 📋 kmeans_centroids.csv         # K-Means cluster centroids
├── 📋 kmeans_centroids_results.csv # Detailed clustering results
├── 📋 ksom_performance_metrics.csv # SOM performance metrics
├── 📑 ASSIGNMENT 3 INSTRUCTION.pdf # Assignment instructions
├── 📑 STINK3014-A251-Assignment-#3.docx # Assignment report
└── 📖 README.md                    # Project documentation

## 🎯 Project Overview

### Objective
Segment customers based on lifestyle behaviors and characteristics using unsupervised machine learning techniques to identify distinct customer groups for targeted marketing strategies.

### Methodology
1. **Data Preprocessing** — Data cleaning, normalization, and feature engineering
2. **Exploratory Data Analysis (EDA)** — Statistical summaries and correlation analysis
3. **K-Means Clustering** — Traditional clustering with elbow method for optimal K
4. **Self-Organizing Maps (SOM)** — Neural network-based clustering approach
5. **Evaluation & Visualization** — Performance metrics and comparative analysis

## 🛠️ Technologies & Dependencies

### Python Version
- Python 3.14.0

### Required Packages
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

- **IDE**: PyCharm 2025.3
- **OS**: macOS 26.1 (ARM64)
- **Environment Manager**: virtualenv

---

## 🚀 Getting Started

### 1. Clone the Repository

#### K-Means Analysis

- Load and preprocess the dataset
- Generate correlation heatmap
- Run elbow method analysis
- Perform K-Means clustering
- Export centroids and results to CSV

#### SOM Analysis

- Train the Self-Organizing Map
- Calculate performance metrics
- Export results to CSV

#### Visualization

---

## 📊 Output Files

### Visualizations
- **`correlation_heatmap.png`** — Shows feature correlations
- **`k_means_elbow_method.png`** — Helps determine optimal number of clusters
- **`k_means_elbow_plot.png`** — Alternative elbow visualization

### Data Exports
- **`kmeans_centroids.csv`** — Final cluster centroids coordinates
- **`kmeans_centroids_results.csv`** — Detailed clustering results with assignments
- **`ksom_performance_metrics.csv`** — SOM algorithm performance metrics

---

## 📈 Key Features

- ✅ **Data Preprocessing** — Handles missing values, outliers, and normalization
- ✅ **Feature Correlation Analysis** — Identifies relationships between variables
- ✅ **Elbow Method** — Determines optimal number of clusters
- ✅ **K-Means Clustering** — Fast and efficient segmentation
- ✅ **Self-Organizing Maps** — Advanced neural network clustering
- ✅ **Performance Metrics** — Quantitative evaluation of clustering quality
- ✅ **Export Results** — CSV files for further analysis

---

## 📝 Documentation

For detailed analysis methodology, findings, and interpretations, please refer to:
- **`STINK3014-A251-Assignment-#3.docx`** — Full assignment report

---

## 👨‍💻 Author

**Imran Mansor**  
Course: STINK3014  
Assignment: #3 — Customer Lifestyle Segmentation  
Date: December 2025

---

## 📧 Contact

For questions or feedback, please contact: **[m_imran_mohamad@soc.uum.edu.my]**

---

## 📄 License

This project is submitted as part of academic coursework. All rights reserved.

---

## 🙏 Acknowledgments

- Course Instructor and Teaching Assistants
- Dataset provided by STINK3014 Course
- Python scientific computing community

---

**⭐ If you find this project useful, please give it a star!**