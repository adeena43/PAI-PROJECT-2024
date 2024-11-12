import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'Boston Dataset.csv'
data = pd.read_csv(file_path)

# Step 1: Preview the Dataset
# Display the first few rows and basic information about the dataset
print("First few rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())
print("\nDataset Summary Statistics:")
print(data.describe())

# Step 2: Outlier Detection using Boxplots
# Visualize outliers across each feature in the dataset

sns.set(style="whitegrid")
plt.figure(figsize=(16, 12))

plot_number = 1
for column in data.columns[1:]:  # Skipping the first column if it's an index
    plt.subplot(5, 3, plot_number)  # Arrange in a 5x3 grid
    sns.boxplot(x=data[column])
    plt.title(f'Boxplot of {column}')
    plot_number += 1  # Increment the plot counter

plt.tight_layout()
plt.show()

# Step 3: Distribution Analysis using Histograms
# Check the distribution of each feature to understand their spread and skewness

plt.figure(figsize=(16, 12))

plot_number = 1
for column in data.columns[1:]:  # Skipping the first column if it's an index
    plt.subplot(5, 3, plot_number)  # Arrange in a 5x3 grid
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plot_number += 1  # Increment the plot counter

plt.tight_layout()
plt.show()

# Step 4: Correlation Analysis using a Heatmap
# Analyze relationships between variables with a correlation heatmap

plt.figure(figsize=(14, 12))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Boston Housing Features")
plt.show()

# Display correlations with target 'medv' sorted in descending order
correlation_with_target = correlation_matrix['medv'].sort_values(ascending=False)
print("\nCorrelations with 'medv':\n", correlation_with_target)

# Step 5: Further Analysis of Significant Feature Relationships
# Focus on the most significant features (highest correlations with 'medv') and explore their relationships with scatter plots

# Plot the relationship between LSTAT (percentage of lower status population) and MEDV (median home value)
plt.figure(figsize=(12, 5))
sns.scatterplot(x='lstat', y='medv', data=data)
plt.title("Relationship between LSTAT (Lower Status) and MEDV (Median Value)")
plt.xlabel("LSTAT - % Lower Status Population")
plt.ylabel("MEDV - Median Value ($1000's)")
plt.show()

# Plot the relationship between RM (average number of rooms) and MEDV (median home value)
plt.figure(figsize=(12, 5))
sns.scatterplot(x='rm', y='medv', data=data)
plt.title("Relationship between RM (Average Rooms) and MEDV (Median Value)")
plt.xlabel("RM - Average Number of Rooms")
plt.ylabel("MEDV - Median Value ($1000's)")
plt.show()
