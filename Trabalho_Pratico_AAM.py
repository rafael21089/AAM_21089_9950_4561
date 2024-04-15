import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from ipywidgets import widgets
from IPython.display import display, clear_output

# Load the injury data from the CSV file
data_file_path = 'injury_data.csv'
df = pd.read_csv(data_file_path)

# EDA
df['Player_Weight'] = df['Player_Weight'].round(2)
df['Player_Height'] = df['Player_Height'].round(2)
df['Training_Intensity'] = df['Training_Intensity'].round(2)

# Remove rows with null or empty values from the original DataFrame
df.dropna(inplace=True)

# Feature engineering
# Calculate the Body Mass Index (BMI)
df['BMI'] = df['Player_Weight'] / (df['Player_Height'] / 100) ** 2
df['Age_squared'] = df['Player_Age'] ** 2
df['BMI_squared'] = df['BMI'] ** 2
df['Training_Recovery_Ratio'] = df['Training_Intensity'] / df['Recovery_Time'].replace(0, np.nan)
# Derive features
df['Age_BMI_Interaction'] = df['Player_Age'] * df['BMI']

# Defining gaps for BMI classification
gaps = [-float('inf'), 18.5, 24.9, 29.9, 34.9, float('inf')]
categories = ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II']

# Create "BMI_Classification" column
df['BMI_Classification'] = pd.cut(df['BMI'], bins=gaps, labels=categories, right=False)

# Creating columns with grouping
df["Age_Group"] = pd.cut(
    df["Player_Age"],
    bins=[18, 22, 26, 30, 34, df["Player_Age"].max()],
    labels=["18-22", "23-26", "27-30", "31-34", "35+"],
    include_lowest=True,
)

# Set the style of seaborn
sns.set(style="whitegrid")

# Plot histogram of player ages
plt.figure(figsize=(10, 6))
sns.histplot(df['Player_Age'], bins=10, kde=False, color='skyblue')
plt.title('Histogram of Player Ages', fontsize=16, fontweight='bold')
plt.xlabel('Player Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot histograms of player weight and player height
plt.figure(figsize=(18, 6))

# Histogram of player weight
plt.subplot(1, 2, 1)
sns.histplot(df['Player_Weight'], bins=20, kde=False, color='skyblue')
plt.title('Histogram of Player Weight', fontsize=16, fontweight='bold')
plt.xlabel('Player Weight', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Histogram of player height
plt.subplot(1, 2, 2)
sns.histplot(df['Player_Height'], bins=20, kde=False, color='salmon')
plt.title('Histogram of Player Height', fontsize=16, fontweight='bold')
plt.xlabel('Player Height', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Subplot for distribution of likelihood of injury for each age group
color_palette = {'Not Injury Prone': '#1f77b4', 'Injury Prone': '#ff7f0e'}
count_plot = sns.countplot(data=df, x='Age_Group', 
                           hue='Likelihood_of_Injury', 
                           order=['18-22', '23-26', '27-30', '31-34', '35+'],
                           palette=color_palette.values()
                           )
plt.title('Distribution of Likelihood of Injury for Each Age Group', fontsize=20, fontweight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Count', fontsize=12)
handles, _ = count_plot.get_legend_handles_labels()
labels = ['Not Injury Prone', 'Injury Prone']
plt.legend(handles, labels, title='Likelihood of Injury', 
           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Subplot for distribution of previous injuries for each age group
color_palette = {'Had No Previous Injuries': '#1f77b4', 'Had Previous Injuries': '#ff7f0e'}
count_plot = sns.countplot(data=df, x='Age_Group', 
                           hue='Likelihood_of_Injury', 
                           order=['18-22', '23-26', '27-30', '31-34', '35+'],
                           palette=color_palette.values()
                           )
plt.title('Distribution of Previous Injuries for Each Age Group', fontsize=20, fontweight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Count', fontsize=12)
handles, _ = count_plot.get_legend_handles_labels()
labels = ['Had No Previous Injuries', 'Had Previous Injuries']
plt.legend(handles, labels, title='Previous Injuries', 
           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Define your color palette
color_palette = {'Had No Previous Injuries': '#1f77b4', 'Had Previous Injuries': '#ff7f0e'}

# Calculate the average likelihood of injury for each age group and previous injuries status
avg_likelihood_by_age_group_and_previous_injuries = df.groupby(['Age_Group', 'Previous_Injuries'], observed=True)['Likelihood_of_Injury'].mean().reset_index()

# Plot bar plot of average likelihood of injury for each age group and previous injuries status
plt.figure(figsize=(12, 6))
bar_plot = sns.barplot(x='Age_Group', y='Likelihood_of_Injury', 
            hue='Previous_Injuries', 
            data=avg_likelihood_by_age_group_and_previous_injuries,
            palette=color_palette.values()
            )
plt.title('Average Likelihood of Injury for Each Age Group', fontsize=14, fontweight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Average Likelihood of Injury', fontsize=12)
handles, _ = count_plot.get_legend_handles_labels()
labels = ['Had No Previous Injuries', 'Had Previous Injuries']
plt.legend(handles, labels, title='Previous Injuries', 
           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Plot scatter plot of Training_Intensity vs Likelihood_of_Injury with linear regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='Training_Intensity', y='Likelihood_of_Injury', data=df, color='skyblue', scatter_kws={'s': 50})
plt.title('Correlation between Training Intensity and Likelihood of Injury', fontsize=14, fontweight='bold')
plt.xlabel('Training Intensity', fontsize=12)
plt.ylabel('Likelihood of Injury', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Calculate the percentage of each BMI category
bmi_percentage = df['BMI_Classification'].value_counts(normalize=True) * 100

# Plot pie chart with legend showing colors
plt.figure(figsize=(10, 8))
pie = plt.pie(bmi_percentage, labels=None, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightblue', 'orange', 'salmon', 'red', 'darkred'], pctdistance=0.85)
plt.title('Percentage of BMI Categories ', fontsize=14, fontweight='bold')
plt.axis('equal')
# Add legend showing colors
legend_labels = bmi_percentage.index
colors = ['lightgreen', 'lightblue', 'orange', 'salmon', 'red', 'darkred']
legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
plt.legend(legend_handles, legend_labels, loc='lower right', fontsize=10)
plt.show()

# Scatter Plot to Examine Relationships
plt.figure(figsize=(14, 8))
sns.scatterplot(x='Player_Age', y='BMI', hue='Likelihood_of_Injury', data=df)
plt.title('Age vs. BMI Colored by Likelihood of Injury')
plt.xlabel('Player Age')
plt.ylabel('BMI')
plt.legend(title='Likelihood of Injury', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Apply OneHotEncoder to categorical columns
one_hot_cols = ["BMI_Classification", "Age_Group"]
encoder = OneHotEncoder()
df_encoded = pd.DataFrame(encoder.fit_transform(df[one_hot_cols]).toarray(), columns=encoder.get_feature_names_out(one_hot_cols))

# Concatenate encoded features with the original dataframe
df_final = pd.concat([df.drop(columns=one_hot_cols), df_encoded], axis=1)

# Calculate correlation matrix
correlation_matrix = df_final.corr()

# Plot heatmap of correlation matrix
plt.figure(figsize=(18, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Matrix', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Plot bar plot of correlation with Likelihood_of_Injury
plt.figure(figsize=(12, 6))
sns.barplot(x=correlation_matrix['Likelihood_of_Injury'].drop('Likelihood_of_Injury').index, 
            y=correlation_matrix['Likelihood_of_Injury'].drop('Likelihood_of_Injury').values, 
            hue=correlation_matrix['Likelihood_of_Injury'].drop('Likelihood_of_Injury').index)

plt.xticks(rotation=45, ha='right')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Correlation', fontsize=12)
plt.title('Correlation of Features with Likelihood_of_Injury', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Machine Learning Model

# Features and target
features = df.drop(columns=['Likelihood_of_Injury'])
target = df['Likelihood_of_Injury']

# Define numerical and categorical pipelines
numeric_features = ['Player_Age', 'Player_Weight', 'Player_Height', 'Training_Intensity', 'BMI']
categorical_features = ['BMI_Classification', 'Age_Group']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

# Function to evaluate models
def evaluate_models(models, X, y, folds=5):
    results = {}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        cv_results = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
        results[name] = np.mean(cv_results)
        print(f"{name}: Mean CV Accuracy = {np.mean(cv_results):.4f} (±{np.std(cv_results):.4f})")
    return results

# Evaluate all models
model_results = evaluate_models(classifiers, features, target)

# Plotting the results
plt.figure(figsize=(10, 5))
model_names = list(model_results.keys())
mean_scores = [np.mean(scores) for scores in model_results.values()]
errors = [np.std(scores) for scores in model_results.values()]

plt.bar(model_names, mean_scores, yerr=errors, capsize=5, color='skyblue')
plt.ylabel('Mean CV Accuracy')
plt.title('Comparison of Model Accuracies')
plt.show()

# Preprocessing
numeric_features = ['Player_Age', 'Player_Weight', 'Player_Height', 'Training_Intensity', 'BMI', 'Age_BMI_Interaction', 'Age_squared', 'BMI_squared', 'Training_Recovery_Ratio']
categorical_features = ['BMI_Classification', 'Age_Group']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Pipeline setup
logreg = LogisticRegression(max_iter=1000, random_state=42)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', logreg)
])

# Define the parameter grid
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

# GridSearchCV setup
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)

# Prepare data
features = df.drop(columns=['Likelihood_of_Injury'])
target = df['Likelihood_of_Injury']

# Train and evaluate
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
print("Best cross-validated accuracy: {:.3f}".format(grid_search.best_score_))

# Evaluate on test data
y_pred = grid_search.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Assuming df is your DataFrame and it already has the necessary columns computed.
# Here's an example of adding necessary calculations to the DataFrame:

# Sample data setup (for demonstration)
data = {
    'Likelihood_of_Injury': [0, 1, 1],
    'Previous_Injuries': [0, 0, 1],
    'Player_Age': [28, 36, 40],
    'Player_Weight': [70, 80, 90],
    'Player_Height': [180, 175, 165],
    'Training_Intensity': [8, 6, 5],
    'Recovery_Time': [2, 2, 3]
}
df = pd.DataFrame(data)

# Calculate BMI and Training Recovery Ratio
df['BMI'] = df['Player_Weight'] / (df['Player_Height'] / 100) ** 2
df['Training_Recovery_Ratio'] = df['Training_Intensity'] / df['Recovery_Time'].replace(0, np.nan)

def classify_investment(row):
    if (row['Likelihood_of_Injury'] == 0 and
        row['Previous_Injuries'] == 0 and
        row['Player_Age'] <= 30 and
        23.07 - 4.01 <= row['BMI'] <= 23.07 + 4.01 and
        row['Training_Recovery_Ratio'] > 0.249):
        return 'Excellent investment'
    elif (row['Likelihood_of_Injury'] == 0 and
          row['Previous_Injuries'] == 0 and
          row['Player_Age'] <= 35 and
          19.05 <= row['BMI'] <= 27.08 and
          0.141 <= row['Training_Recovery_Ratio'] <= 0.249):
        return 'Good investment'
    else:
        return 'Bad investment'
   
# Apply the classification function
df['Investment_Rating'] = df.apply(classify_investment, axis=1)

print(df[['Likelihood_of_Injury', 'Previous_Injuries', 'Player_Age', 'BMI', 'Training_Recovery_Ratio', 'Investment_Rating']])
