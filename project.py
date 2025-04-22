# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from scipy.stats import skew
from scipy import stats  # Added for regression
import warnings
warnings.filterwarnings('ignore')

# Set Seaborn theme and custom palette
sns.set_theme(style="whitegrid")
custom_palette = sns.color_palette("Set2")  # Unified color scheme
sns.set_palette(custom_palette)

# Create a directory to save plots
if not os.path.exists("eda_plots"):
    os.makedirs("eda_plots")

# Function to detect and handle outliers using IQR
def handle_outliers(df, column, method='remove'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    print(f"\n🔹 Outliers in {column}: {len(outliers)}")
    
    if method == 'remove':
        df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'cap':
        df_clean = df.copy()
        df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
    else:
        df_clean = df
    
    return df_clean, outliers

# Load the dataset with error handling
file_path = "C:/Users/hp/Desktop/python_dataset.csv"
try:
    df = pd.read_csv(file_path, encoding='latin1')
except FileNotFoundError:
    print(f"Error: Dataset file '{file_path}' not found. Please check the file path.")
    raise SystemExit
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise SystemExit

# Check if dataset is empty
if df.empty:
    print("Error: Dataset is empty.")
    raise SystemExit

# Basic info
print("🔹 Dataset Info:")
print(df.info())

# First few rows
print("\n🔹 First 5 Rows:")
print(df.head())

# Missing values analysis
print("\n🔹 Missing Values:")
missing_data = df.isnull().sum()
print(missing_data)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='Blues')  # Changed to Blues
plt.title("Missing Values Heatmap")
plt.savefig("eda_plots/missing_values_heatmap.png")
plt.show()

# Handle missing values (impute 'system' with mode, others as needed)
if df['system'].isnull().sum() > 0:
    df['system'].fillna(df['system'].mode()[0], inplace=True)
print("\n🔹 Missing Values After Imputation:")
print(df.isnull().sum())

# Summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe(include='all'))

# Column data types
print("\n🔹 Column Types:")
print(df.dtypes)

# Unique values for each categorical column
print("\n🔹 Unique Values in Categorical Columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col} ➤ Unique Values: {df[col].nunique()}")
    print(df[col].value_counts().head(5))

# Check for duplicate rows
print(f"\n🔹 Duplicate Rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"🔹 Duplicate Rows After Removal: {df.duplicated().sum()}")

# Skewness analysis for numeric columns
print("\n🔹 Skewness of Numeric Columns:")
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    skewness = skew(df[col].dropna())
    print(f"{col}: {skewness:.2f}")
    if abs(skewness) > 1:
        print(f"  ➤ High skewness detected. Consider log transformation for {col}.")
        df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

# Outlier detection and handling
for col in numeric_cols:
    df, outliers = handle_outliers(df, col, method='cap')  # Use 'cap' to avoid reducing dataset size
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col], color=custom_palette[1])  # Use Set2 color
    plt.title(f"Boxplot of {col} After Outlier Handling")
    plt.savefig(f"eda_plots/boxplot_{col}_after.png")
    plt.show()

# Correlation matrix (Pearson)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True, method='pearson'), annot=True, cmap='Blues', fmt=".2f")  # Changed to Blues
plt.title("Pearson Correlation Matrix")
plt.savefig("eda_plots/pearson_correlation_matrix.png")
plt.show()

# Highlight strong correlations
corr_matrix = df.corr(numeric_only=True)
strong_corrs = corr_matrix.where(np.triu(np.abs(corr_matrix) > 0.7, k=1)).stack()
print("\n🔹 Strong Correlations (|corr| > 0.7):")
print(strong_corrs)

# Histograms for numeric columns
df.select_dtypes(include='number').hist(figsize=(15, 10), bins=30, color=custom_palette[2], edgecolor='black')  # Use Set2 color
plt.suptitle("Distribution of Numeric Columns", fontsize=16)
plt.savefig("eda_plots/numeric_histograms.png")
plt.show()



# Feature engineering: Binning 'Tot_ED_NmbVsts' into categories
df['ED_Visits_Category'] = pd.qcut(df['Tot_ED_NmbVsts'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
print("\n🔹 New Feature 'ED_Visits_Category' Created:")
print(df['ED_Visits_Category'].value_counts())

# Count plot for each categorical column
for col in categorical_cols:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index[:10], palette='Set2')  # Use Set2
    plt.title(f"Top 10 Frequent Values in {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"eda_plots/countplot_{col}.png")
    plt.show()

# Stacked bar plot: HospitalOwnership vs UrbanRuralDesi
plt.figure(figsize=(10, 6))
pd.crosstab(df['HospitalOwnership'], df['UrbanRuralDesi']).plot(kind='bar', stacked=True, color=custom_palette)  # Use Set2
plt.title("Stacked Bar Plot: HospitalOwnership vs UrbanRuralDesi")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_plots/stacked_bar_hospital_urban.png")
plt.show()

# Interactive scatter plot: Tot_ED_NmbVsts vs EDStations
fig = px.scatter(df, x='Tot_ED_NmbVsts', y='EDStations', color='UrbanRuralDesi',
                 title="Interactive Scatter Plot: ED Visits vs ED Stations",
                 color_discrete_sequence=px.colors.qualitative.Set2)  # Use Set2
fig.write_html("eda_plots/interactive_scatter.html")  # Fixed method
fig.show()

# Regression line chart: Tot_ED_NmbVsts vs EDDXCount
print("\n🔹 Regression Analysis: Tot_ED_NmbVsts vs EDDXCount")
plt.figure(figsize=(10, 6))
sns.regplot(x='Tot_ED_NmbVsts', y='EDDXCount', data=df, scatter_kws={'alpha':0.5, 'color':custom_palette[0]},
            line_kws={'color':custom_palette[4]})  # Use Set2 colors
plt.title("Regression Line: Total ED Visits vs ED Diagnoses")
plt.xlabel("Total ED Visits")
plt.ylabel("ED Diagnoses")
plt.tight_layout()
plt.savefig("eda_plots/regression_ed_visits_vs_diagnoses.png")
plt.show()

# Calculate and display regression slope and intercept
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Tot_ED_NmbVsts'], df['EDDXCount'])
print(f"  ➤ Slope: {slope:.4f}, Intercept: {intercept:.4f}")
print(f"  ➤ R-squared: {r_value**2:.4f}, P-value: {p_value:.4f}")

# Optional: Regression plots for other numeric pairs
other_pairs = [('Tot_ED_NmbVsts', 'EDStations')]  # Swapped to include EDStations
for x_col, y_col in other_pairs:
    print(f"\n🔹 Regression Analysis: {x_col} vs {y_col}")
    plt.figure(figsize=(10, 6))
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'alpha':0.5, 'color':custom_palette[0]},
                line_kws={'color':custom_palette[4]})  # Use Set2 colors
    plt.title(f"Regression Line: {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(f"eda_plots/regression_{x_col}_vs_{y_col}.png")
    plt.show()
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
    print(f"  ➤ Slope: {slope:.4f}, Intercept: {intercept:.4f}")
    print(f"  ➤ R-squared: {r_value**2:.4f}, P-value: {p_value:.4f}")

# Generate a summary report
with open("eda_summary.txt", "w") as f:
    f.write("Exploratory Data Analysis Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"Numeric Columns: {list(numeric_cols)}\n")
    f.write(f"Categorical Columns: {list(categorical_cols)}\n")
    f.write("\nMissing Values After Imputation:\n")
    f.write(str(df.isnull().sum()) + "\n")
    f.write("\nStrong Correlations (|corr| > 0.7):\n")
    f.write(str(strong_corrs) + "\n")
    f.write("\nPlots Saved in: eda_plots/\n")
print("\n🔹 EDA Summary saved to 'eda_summary.txt'")

print("\n🔹 EDA Completed! All plots saved in 'eda_plots/' directory.")