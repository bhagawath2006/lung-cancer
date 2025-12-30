# lung-cancer
https://colab.research.google.com/notebooks/intro.ipynb

🩺 Lung Cancer Prediction Using Machine Learning

Lung cancer is one of the most serious and life-threatening diseases worldwide. Early prediction of lung cancer risk can significantly improve treatment outcomes and patient survival rates.
This project uses **machine learning algorithms** to predict the likelihood of lung cancer based on **patient medical, lifestyle, and demographic data**.

The project includes **data preprocessing, model training, evaluation, and model saving**, all implemented in **Python using Scikit-learn** and executed in **Jupyter Notebook / Google Colab**.

🎯 Objectives

* To analyze medical and lifestyle data of patients
* To preprocess and transform raw data into a usable format
* To train multiple machine learning classification models
* To compare model performance and select the best one
* To save the trained model for future predictions or deployment

📂 Dataset

File name: `dataset_med.csv`
Size: ~890,000 records
Type: Structured medical dataset
Features include:

  * Age, Gender, Country
  * Cancer Stage
  * Family History
  * Smoking Status
  * BMI and Cholesterol Level
  * Medical conditions (Hypertension, Asthma, Cirrhosis, etc.)
  * Treatment Type
  Target column:** `survived` / lung cancer risk label

🧠 Machine Learning Algorithms Used

* **Logistic Regression**
* **Random Forest Classifier**
* **Gradient Boosting Classifier**
* **Support Vector Machine (SVM)**

Each model is trained and evaluated to identify the best-performing classifier.

 🔄 Project Workflow

1. Load and inspect the dataset
2. Data cleaning and preprocessing
3. Feature encoding and scaling
4. Train–test split
5. Model training using multiple algorithms
6. Model evaluation using standard metrics
7. Save the best-performing model

 ⚙️ Preprocessing Techniques

* Handling missing values and duplicates
* Encoding categorical features using **OneHotEncoder**
* Scaling numerical features using **StandardScaler**
* Using **ColumnTransformer** and **Pipeline** for clean and efficient preprocessing

 📊 Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics help compare model performance and reliability.

 🛠️ Tech Stack

Programming Language

* Python

Libraries Used

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib

Environment

* Jupyter Notebook
* Google Colab

▶️ How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/your-username/Lung_Cancer_Prediction.git
```

2. Navigate to the project folder:

```bash
cd Lung_Cancer_Prediction
```

3. Open the notebook:

```bash
jupyter notebook Lung_Cancer_Prediction.ipynb
```

4. Run all cells in sequence

 📈 Output

* Dataset summary and statistics
* Model comparison results
* Evaluation metrics
* Saved trained model file (`.pkl`)







