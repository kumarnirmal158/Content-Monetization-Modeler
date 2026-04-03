# Content-Monetization-Modeler
# 📊 YouTube Revenue Prediction & Data Cleaning Pipeline

## 🚀 Project Overview

This project builds a complete **Machine Learning pipeline** to predict YouTube revenue using data preprocessing, feature engineering, and multiple regression models.

It includes:

* Data Cleaning
* Handling Missing Values
* Outlier Detection
* Encoding
* Feature Scaling
* Model Training & Evaluation

---

## 🧠 Problem Statement

Predict the **revenue of YouTube videos** based on features like views, likes, comments, category, device, country, etc.

---

## 📁 Project Structure

```
├── data_cleaning.ipynb   # Main notebook
├── dataset.csv           # Input dataset
├── model.pkl             # Saved trained model
├── scaler.pkl            # Saved scaler
├── README.md             # Project documentation
```

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## 🔄 Workflow

### 1️⃣ Data Loading

* Load dataset using `pandas`

### 2️⃣ Data Cleaning

* Remove unnecessary columns
* Handle missing values:

  * Mean → for normal distribution
  * Median → for skewed data

### 3️⃣ Outlier Detection

* Z-score method used
* Threshold: `|Z| > 3`

### 4️⃣ Encoding

* One-Hot Encoding using `pd.get_dummies()`
* Avoided dummy variable trap using `drop_first=True`

### 5️⃣ Feature Scaling

* Standardization using `StandardScaler`

### 6️⃣ Train-Test Split

* Split dataset into training and testing sets

### 7️⃣ Model Training

Models used:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

### 8️⃣ Model Evaluation

* R² Score
* Mean Squared Error (MSE)

### 9️⃣ Model Saving

* Saved model using `pickle`

# 💰 YouTube Content Monetization Modeler (Streamlit App)

## 🚀 Project Overview

This project is an **interactive Streamlit web application** that predicts **YouTube Ad Revenue** using a trained Machine Learning model.

It allows users to:

* 📊 Predict revenue based on video performance
* 🔍 Explore dataset insights (EDA)
* 🧠 Understand model behavior and feature importance

---

## 🧠 Objective

To build an end-to-end **ML-powered web app** that:

* Takes user input (views, likes, etc.)
* Applies preprocessing (encoding + scaling)
* Uses a trained model to predict revenue

---

## ⚙️ Technologies Used

* Python 🐍
* Streamlit
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Pickle (model persistence)

---

## 📁 Project Structure

```bash
├── app.py                # Streamlit application
├── model.pkl            # Trained ML model (Lasso)
├── scaler.pkl           # StandardScaler object
├── columns.pkl          # Model feature columns
├── youtube_dataset.csv  # Dataset (for insights)
```

---

# 🔄 Application Workflow

---

## 🏠 1️⃣ Page Configuration

```python
st.set_page_config(...)
```

**Purpose:**

* Sets page title, layout, and icon

---

## 📦 2️⃣ Load Model & Artifacts

```python
pickle.load(...)
```

**Files loaded:**

* `model.pkl` → trained ML model
* `scaler.pkl` → feature scaler
* `columns.pkl` → feature order

**Purpose:**

* Avoid retraining
* Ensure consistency during prediction

---

## 📊 3️⃣ Load Dataset

```python
@st.cache_data
def load_data():
```

**Functions used:**

* `pd.read_csv()` → load dataset
* `pd.to_datetime()` → convert date

**Purpose:**

* Used in **Data Insights page**

**Note:**

* `@st.cache_data` improves performance by caching data

---

## 📌 4️⃣ Sidebar Navigation

```python
st.sidebar.radio(...)
```

**Pages:**

* Home
* Prediction
* Data Insights
* Model Insights

---

# 🏠 HOME PAGE

**Purpose:**

* Introduces the application
* Explains features

---

# 🚀 PREDICTION PAGE

## 📥 User Inputs

```python
st.number_input()
st.selectbox()
```

**Collected Features:**

* Views, Likes, Comments
* Watch Time, Video Length
* Subscribers, Year
* Category, Device, Country

---

## ⚙️ Feature Engineering

```python
engagement_rate = (likes + comments) / views
```

**Purpose:**

* Creates new feature
* Improves model performance

---

## 🔄 Encoding

```python
pd.get_dummies(...)
```

**Purpose:**

* Convert categorical → numeric

---

## 📐 Column Alignment

```python
for col in model_columns:
```

**Purpose:**

* Match input features with training data
* Add missing columns with 0

---

## ⚖️ Scaling

```python
scaler.transform(input_encoded)
```

**Purpose:**

* Standardize features
* Maintain consistency with training

---

## 🤖 Prediction

```python
model.predict(input_scaled)
```

**Model Used:**

* Lasso Regression (L1 regularization)

**Purpose:**

* Predict YouTube ad revenue

---

# 📊 DATA INSIGHTS PAGE

## 🔥 Correlation Heatmap

```python
sns.heatmap(...)
```

**Purpose:**

* Shows relationships between variables

---

## 📈 Distribution Analysis

```python
sns.histplot()
sns.boxplot()
```

**Purpose:**

* Histogram → distribution
* Boxplot → outlier detection

---

## 📅 Trends Over Time

```python
sns.lineplot(...)
```

**Purpose:**

* Analyze performance trends

---

## 🏷️ Categorical Analysis

```python
groupby().mean()
sns.barplot()
sns.countplot()
```

**Purpose:**

* Compare performance across categories

---

# 🧠 MODEL INSIGHTS PAGE

## 🔍 Feature Importance

```python
model.coef_
```

**Purpose:**

* Identify most influential features

---

## 📊 Visualization

```python
sns.barplot(...)
```

**Purpose:**

* Rank features by importance

---

## 📌 Interpretation

* High importance → strong influence
* Low importance → minimal effect

---

# 🔑 Key Concepts Used

| Concept             | Purpose                  |
| ------------------- | ------------------------ |
| Pickle              | Save/load model          |
| One-Hot Encoding    | Convert categorical data |
| StandardScaler      | Normalize features       |
| Feature Engineering | Improve predictions      |
| Lasso Regression    | Feature selection        |
| EDA                 | Data understanding       |

---

# ⚠️ Important Highlights

### ✅ Column Matching

Ensures input matches training features

### ✅ Scaling Consistency

Uses same scaler as training

### ✅ Caching

Improves performance

---

# 🎯 Final Workflow

```text
User Input
   ↓
Feature Engineering
   ↓
Encoding
   ↓
Column Alignment
   ↓
Scaling
   ↓
Model Prediction
   ↓
Display Result
```

---

Give it a ⭐ on GitHub!

