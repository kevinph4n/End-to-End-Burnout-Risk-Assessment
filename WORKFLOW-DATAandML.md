# **I/ Check the Data & EDA**

## **1/ Define Objectives & Review**
- Define your primary objective (e.g., Prediction or a specific Machine Learning task).
- Review the dataset manually rather than relying entirely on AI agents to maintain full control over feature selection.
- Calculate and determine the size of the sample.

## **2/ Explore Data & Detect Bias**
- libraries: pandas, matplotlib, seaborn
- Must check out the database by using visualizations to know more about the dataset (Histogram + KDE).
- Calculate the data distribution → by that, detect outliers.
- Check out if the data was biased or not. Deal with data bias by transforming the distributed data into more normalized data (log, log1p, boxcox).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read data
df = pd.read_csv('data.csv')

# overview
df.head()
df.info()
df.describe()

# detect outliers and bias using Histogram + KDE
sns.histplot(df['feature'], kde=True)
plt.show()

# transformative the distributed data into more normalized data
df['normalized_feature'] = np.log1p(df['feature'])
```

---

# **II/ Processing Data**

## **1/ Handling Missing Data & Outliers**
- Dealing with the outlier: Mask the data or Impute the data → replace with substitutes values.
- Filling the gaps, identify the type of the blanks.
- Numbers → can be filled by the mean() or med().
- libraries: sklearn.preprocessing, sklearn.impute

```python
from sklearn.impute import SimpleImputer

# check missing values
df.isnull().sum()

# replace numerical outliers or blanks with median
median_val = df['numeric_col'].median()
df['numeric_col'] = df['numeric_col'].fillna(median_val)

# alternative: use SimpleImputer for missing values
imputer = SimpleImputer(strategy='mean')
```

---

## **2/ Handling Categorical Data**
- Objects or String → locate the column we want to clean, identify the method, most common is fill those columns with “Unknown” or “Missing”.
- Step 1: Statistic to get the most - second - third common attribute.
- Step 2: Overwrite the rest of attributes by “Others” and fill the missing gap by “Unknown”. By that way, we aren't afraid of the high dimensionality.

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# The syntax: fill missing gap
df['column'] = df['column'].fillna('Unknown')

# encode categorical data into numerical formats using pandas
df = pd.get_dummies(df, columns=['category_col'])

# alternative: encode categorical data into numerical formats using sklearn
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder()
```

---

## **3/ Feature Scaling & Finalizing Variables**
- The feature method: encoding + scaling.
- Data scaling is crucial. Related features must be scaled together so they share the same magnitude.
- Normalize/standardize data scales (crucial for models like SVM, KNN, or Neural Networks).
- Integration Point: Your teammate must finalize and hand over these two crucial variables to you:
- X: The features (independent variables used for prediction).
- y: The target (the dependent variable you want to predict).

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# feature scaling
scaler = StandardScaler() # or MinMaxScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])
```

---

# **III/ Data Visualization (3 Stages)**

## **1/ Pre-training Stage (Data Exploration & Hypothesis)**
- Visualize the difference between the raw data and the refined data after cleaning.
- It is best to present key statistical metrics such as Standard Deviation, Mean, Median, Variance, etc.
- Formulate a hypothesis about the expected outcomes of the model, and visualize these predictions (e.g., Decision Boundary Plot for Classification or Partial Dependence Plot for Regression).

```python
# Key statistical metrics
print(df.describe())
print("Variance:", np.var(df['feature']))
```

---

## **2/ Post-training Stage (Model Evaluation)**
- Create a visualization to illustrate the relationship between your initial hypothesis and the model's actual performance to assess overfitting or underfitting.
- I strongly encourage using a Predicted vs. Actual Plot combined with the Identity Line (y = x).

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Predicted vs. Actual Plot with Identity Line
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual (Hypothesis Checking)')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Residual plot
sns.residplot(x=y_pred, y=y_test - y_pred)
plt.show()
```

---

## **3/ Post-tuning Stage (Final Assessment)**
- Once the model is tuned, you need a final visualization to demonstrate the improvements gained from your fine-tuning efforts.
- Naturally, the types of graphs used in this stage will be similar to those in Stage 2 to provide a clear before-and-after comparison.

---

# **IV/ Formulate Hypotheses & Advanced Concepts**

## **1/ Final Evaluation & Hypothesis Dashboard**
- Feature Importance: Analyze which individual variable has the greatest impact on your prediction, which has the least impact, and which seems to have absolutely no effect.
- Worst-Case Scenarios: Draw hypothetical conclusions for worst-case scenarios (e.g., extremely overpriced house).
- Performance Thresholds: Establish a Minimum Acceptable Accuracy for your model (e.g., using MAE - Mean Absolute Error) and aim for an R-squared score of over 0.8.

```python
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
    
print(f"MAE: {mae}")
print(f"R-squared: {r2}")
```

---

## **2/ Datatypes Categorization**
- Qualitative Data (Categorical):
- Nominal Data: No order (e.g., Gender, Employment_Status).
- Ordinal Data: Has clear ranking (e.g., Burnout_Risk, Sleep_Quality).
- Quantitative Data (Numerical):
- Discrete Data: Countable whole numbers (e.g., Coffee_Cups_Per_Day).
- Continuous Data: Measurements with decimals (e.g., Age, Screen_Time_Hours).

---

## **3/ Distribution Selection (Maximum Likelihood Estimation)**
- Domain Knowledge: Check data boundaries (e.g., cannot be negative -> drop Normal, use Log-Normal/Gamma; counting data -> use Poisson).
- Exploratory Data Analysis (EDA):
- Symmetrical peak -> Normal.
- Right-skewed tail -> Log-Normal or Gamma.
- Steep drop from zero -> Exponential.
- Statistical Testing: Use Q-Q Plot or AIC/BIC scores to evaluate the fit.

```python
import scipy.stats as stats

# Q-Q Plot to check distribution fit
stats.probplot(df['feature'], dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()
```

---

## **4/ Common Ways to Fine-Tune ML Models (Regularization)**
- In Machine Learning (especially for models like Ridge or Lasso Regression), `alpha` acts as a **Regularization parameter**. Its main job is to control the model's complexity by applying a "penalty" to prevent it from overcomplicating things.
- **When `alpha` is close to 0 (Very Low):**
  - The penalty is almost non-existent. The model is completely "unleashed" and will bend itself to perfectly connect every single dot in your training data, including random noise.
  - **The result:** Your training accuracy will look amazingly high, but the model will fail miserably on unseen test data. This is classic **Overfitting** (the model is just memorizing the data instead of understanding the true patterns).
- **When `alpha` increases (High):**
  - The penalty becomes strict. The model's coefficients are squeezed and forced to drop unnecessary complexity, keeping only the core trends.
  - **The result:** Increasing `alpha` is the ultimate medicine to cure Overfitting. However, if `alpha` is pushed *too high*, the model becomes overly rigid and fails to learn anything useful at all. This is called **Underfitting** (the model becomes too restricted to make good predictions).
- **The Bottom Line:** A low `alpha` makes the model try too hard to fit the data (causing Overfitting), while a higher `alpha` is used to restrict the model and prevent this memorization. Your ultimate goal is to find the perfect balance.
- Naturally, to evaluate whether the model truly performs well using this hyperparameter tuning method, I highly recommend using a Validation Curve for the most intuitive visualization and straightforward conclusions.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

# 1. Define the range of alpha values to test (from very low to very high)
param_range = np.logspace(-3, 3, 7) # e.g., [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# 2. Calculate accuracy on training and test sets using Cross-Validation
train_scores, test_scores = validation_curve(
    Ridge(), X_train, y_train, param_name="alpha", param_range=param_range, 
    cv=5, scoring="r2"
)

# 3. Calculate mean scores
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# 4. Plot the Validation Curve
plt.figure(figsize=(8, 5))
plt.plot(param_range, train_mean, label="Training Score (Overfits at low alpha)", color="blue", marker='o')
plt.plot(param_range, test_mean, label="Cross-Validation Score", color="red", marker='s')

plt.title("Validation Curve for Ridge Regression")
plt.xlabel("Alpha (Regularization Parameter)")
plt.ylabel("R^2 Score")
plt.xscale("log") # Use log scale to properly space out alpha values
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

---

# **V/ Model Training & Optimization**

## **1/ Train/Test Split**
- Use `train_test_split` to allocate portions of the dataset for training and testing purposes.

```python
from sklearn.model_selection import train_test_split

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## **2/ Initialize & Train the Model**
- Depending on whether your problem is Classification or Regression, you choose the appropriate algorithm. Below is an example using Random Forest (a robust algorithm that is less prone to overfitting).

```python
from sklearn.ensemble import RandomForestClassifier # Or RandomForestRegressor for continuous numerical predictions

# 1. Initialize the model (you can pre-set some hyperparameters here)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# 2. Train the model (Learn from the training data)
model.fit(X_train, y_train)
```

---

## **3/ Model Evaluation**
- You will use `sklearn.metrics` to calculate the metrics and `seaborn` to visualize the Confusion Matrix (which looks great in project presentations).

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Make predictions on the test set
y_pred = model.predict(X_test)

# --- 1. Calculate F1-Score ---
# Use average='binary' for 2 classes. If you have multiple classes (e.g., A, B, C), use average='weighted'
f1 = f1_score(y_test, y_pred, average='binary') 
print(f"F1-Score: {f1:.4f}")

# --- 2. Generate & Plot Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
print("Raw Confusion Matrix:\n", cm)

# Plotting the matrix for a visual report
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Model Confusion Matrix')
plt.xlabel('Prediction')
plt.ylabel('Actual Truth')
plt.show()

# You can still keep the classification report for a full summary
print("\nFull Classification Report:\n", classification_report(y_test, y_pred))
```

---

## **4/ How to Evaluate the Model Using These Two Metrics**

### **Confusion Matrix**
- The Confusion Matrix: Finding where the model is failing. A Confusion Matrix does not just give you a single score; it breaks down your model's predictions into four specific categories so you can see exactly what kind of mistakes it is making.
- Assuming a binary classification (e.g., predicting if a customer will buy a product or not):
- True Positives (TP - Bottom Right): The model predicted "Yes", and the actual answer was "Yes". (Correct)
- True Negatives (TN - Top Left): The model predicted "No", and the actual answer was "No". (Correct)
- False Positives (FP - Top Right / Type I Error): The model predicted "Yes", but the actual answer was "No". (Wrong - e.g., The model predicted a customer would buy, but they didn't).
- False Negatives (FN - Bottom Left / Type II Error): The model predicted "No", but the actual answer was "Yes". (Wrong - e.g., The model predicted the customer wouldn't buy, but they actually did).
- How to evaluate it: You want the numbers on the main diagonal (Top-Left and Bottom-Right) to be as high as possible. You want the numbers on the off-diagonal (Top-Right and Bottom-Left) to be as close to zero as possible. If your FP is very high, your model is too "trigger-happy" (guessing Yes too often). If your FN is very high, your model is too conservative (missing out on actual positive cases).

### **The F1-Score: The ultimate balance metric**
- Accuracy is often a misleading metric, especially if your dataset is imbalanced. For example, if 99% of your emails are normal and 1% is spam, a terrible model that simply predicts "Normal" every single time will still achieve 99% accuracy, but it is completely useless at catching spam. The F1-Score fixes this by combining two other metrics into one single score:
- Precision: Out of all the times the model predicted "Yes", how many were actually correct?
- Recall: Out of all the actual "Yes" cases in the dataset, how many did the model successfully find?
- How to evaluate it: The F1-Score ranges from 0.0 to 1.0 (or 0% to 100%). It will only give a high score if both Precision and Recall are high.
- 0.90 - 1.00: Excellent model. It rarely misses positive cases and rarely makes false positive claims.
- 0.70 - 0.89: Good model. This is a very realistic and acceptable range for most real-world ML projects.
- 0.50 - 0.69: Mediocre model. It is making a significant amount of mistakes and you likely need to ask your teammate to extract better features from the data, or you need to tune your hyperparameters.
- < 0.50: Poor model. It is performing barely better than random guessing.

---

## **5/ Hyperparameter Tuning**
- To turn a "good" model into an "excellent" one, you need to tweak its internal parameters. Instead of guessing manually, use automated search over specified parameter values.

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search through
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Run the grid search (using Cross-Validation)
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1, # Use all available CPU cores
                           verbose=2)

grid_search.fit(X_train, y_train)

# Extract the best performing model
best_model = grid_search.best_estimator_
print("Best hyperparameters:", grid_search.best_params_)
```

---

## **6/ Export the Model**
- Once you have the best_model, you must save it to a file so you can deploy it later. This prevents you from having to retrain the model from scratch every time you launch the app.

```python
import joblib

# Export the trained model to a file
joblib.dump(best_model, 'my_trained_model.pkl')

# (Optional) Export your teammate's scaler if StandardScaler was used
# joblib.dump(scaler, 'my_scaler.pkl')
```

---

# **VI/ App Deployment (Building the Final Demo)**
- To showcase your project effectively, you should build a Web App rather than asking judges or users to run Python scripts in a terminal. Streamlit is currently the top choice for ML demos because of its rapid development speed and clean UI.

```python
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the trained model
model = joblib.load('my_trained_model.pkl')
# scaler = joblib.load('my_scaler.pkl') # Load this if you used scaling

# 2. Design the Web UI
st.title("Ultimate Prediction Web App")
st.write("Enter the parameters below to get a prediction from the model.")

# 3. Create input forms for the user (in a Sidebar or the Main body)
# Assuming your data has 3 features: Age, Salary, Credit Score
age = st.slider("Enter Age:", min_value=18, max_value=80, value=25)
salary = st.number_input("Enter Salary ($):", min_value=0, value=50000)
credit_score = st.slider("Credit Score:", min_value=300, max_value=850, value=650)

# 4. Action button to trigger the model
if st.button("Predict Now"):
    # Package the input data into a 2D array (the format sklearn expects)
    input_data = np.array([[age, salary, credit_score]])
    
    # input_data_scaled = scaler.transform(input_data) # Normalize if necessary
    
    # Execute the prediction
    prediction = model.predict(input_data) # or input_data_scaled
    
    # 5. Display the results
    if prediction[0] == 1:
        st.success("Prediction: The customer is LIKELY to purchase!")
        st.balloons()
    else:
        st.error("Prediction: The customer will NOT purchase.")
```