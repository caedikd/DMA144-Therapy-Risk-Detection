
Link to the presentation here:
https://drive.google.com/file/d/1XXh9EoL10mIGqqWBvDwk20kyYx9guukF/view?usp=sharing

# Therapy Risk Detection Using DASS-21 Survey Data

## Project Overview

This project explores whether early post-session survey responses can be used to predict if a patient will require **30 or more therapy sessions**. The goal is to identify patients who may need extended care earlier in treatment so that therapists can review care plans or allocate additional resources sooner.

The dataset consists of responses to the **DASS-21 (Depression, Anxiety, Stress Scale)** survey completed by patients after therapy sessions. Using these responses, we built machine learning models to determine whether patterns in early sessions could predict long-term therapy needs.

This project was completed as part of a data mining course and focuses on **handling imbalanced datasets, model comparison, and evaluation using appropriate metrics**.

---

# Problem Statement

The objective is to predict whether a patient will eventually reach **30+ therapy sessions** based on information collected during **early sessions**.

**Target Variable**

`thirtyplus`
Binary label indicating whether a patient ultimately attends 30 or more therapy sessions.

**Motivation**

Patients requiring long-term care may benefit from earlier intervention or reevaluation of treatment plans. Predicting long-term therapy needs earlier in the process can help providers allocate resources more effectively.

---

# Dataset

The dataset contains survey responses collected via Google Forms after therapy sessions. Each survey includes:

* DASS-21 question responses (0–3 frequency scale)
* Session number
* Therapist identifier
* Timestamp and anonymized patient identifiers

The **DASS-21 instrument** measures three emotional states:

* Depression
* Anxiety
* Stress

Each category is calculated using a subset of survey questions and converted into severity levels.

Because this dataset contains sensitive information, the raw data is **not included in this repository**.

---

# Feature Engineering

## DASS Score Calculation

Survey responses are converted into three standardized metrics:

* **Depression score**
* **Anxiety score**
* **Stress score**

Each score is calculated by summing specific survey questions and converting the total into severity categories ranging from **0 (normal) to 4 (extremely severe)**.

## Target Creation

A new target variable (`thirtyplus`) was created by identifying patients who eventually reached the **30+ session category**.

## Additional Processing

Additional preprocessing steps included:

* Cleaning inconsistent identifiers
* Converting session ranges (e.g., 13–20, 21–30, 30+) into numerical proxies
* One-hot encoding therapist identifiers
* Removing unused columns
* Creating a filtered dataset using only **early sessions (≤10)** to simulate early prediction

---

# Exploratory Data Analysis

Initial exploration revealed two key insights:

### Severe Class Imbalance

Only about **10% of patients** fall into the `30+ sessions` category.

Because of this imbalance, **accuracy is not an appropriate evaluation metric**. A naive model predicting "not 30+" for every patient already achieves roughly **90% accuracy**.

### Therapist Influence

Feature correlation analysis suggested that **therapist identity had a measurable effect** on patient session duration. This may reflect differences in patient assignment, treatment approach, or patient needs.

---

# Evaluation Strategy

To properly evaluate models on this imbalanced dataset, we focused on:

* **Recall** – minimizing false negatives (missing patients who need extended care)
* **Precision**
* **F1 Score**
* **Confusion Matrix**

Recall is particularly important in this context because failing to identify a patient who needs additional care is worse than incorrectly flagging someone who does not.

---

# Handling Class Imbalance

Because the dataset is highly skewed, multiple strategies were explored:

### Upsampling

The minority class (30+ patients) was duplicated through sampling with replacement until it matched the size of the majority class.

### Downsampling

The majority class was reduced to match the minority class.

### Class Weighting

Some models were trained using balanced class weights to penalize misclassification of the minority class more heavily.

Upsampling generally produced better results because the dataset was relatively small.

---

# Models Tested

Several models were trained and evaluated.

## Logistic Regression (Baseline)

A logistic regression model using the DASS scores was used as a baseline.

However, due to the class imbalance and nonlinear relationships in the data, the model struggled to identify positive cases and often predicted only the majority class.

---

## Random Forest

Random Forest models performed significantly better by capturing nonlinear relationships and interactions between features.

After tuning hyperparameters and using resampling techniques, the model achieved:

* Strong recall
* Improved F1 scores
* Useful feature importance insights

Feature importance analysis revealed that **DASS scores and therapist identifiers** were the most influential predictors.

---

## K-Nearest Neighbors (KNN)

KNN was tested after scaling the data and selecting the optimal value of **K using the elbow method**.

While KNN performed reasonably well on the balanced dataset, its performance dropped on the original imbalanced data.

---

## CART (Decision Tree)

Decision trees were trained using cross-validation and pruning.

These models provided useful interpretability but had lower predictive stability compared to ensemble methods.

---

## XGBoost

Extreme Gradient Boosting (XGBoost) was used as an advanced ensemble method.

Randomized search was used to tune hyperparameters such as:

* learning rate
* tree depth
* gamma
* minimum child weight

XGBoost performed well on the resampled dataset but showed reduced performance when tested on the original imbalanced data.

---

# Key Results

The strongest models were tree-based ensemble methods.

| Model               | Strength                                               |
| ------------------- | ------------------------------------------------------ |
| Random Forest       | Best balance of recall, F1 score, and interpretability |
| XGBoost             | Strong recall when trained on balanced data            |
| Decision Tree       | High interpretability                                  |
| Logistic Regression | Useful baseline                                        |

Overall, **Random Forest performed best in balancing recall and precision**, while **XGBoost produced strong recall when optimized for detecting minority cases**.

---

# Feature Importance

Random Forest analysis showed that the most important features included:

* Depression score
* Anxiety score
* Stress score
* Therapist identifier

Interestingly, therapist identity appeared to influence session length almost as much as the DASS severity scores.

This suggests that treatment style or patient assignment may influence therapy duration.

---

# Limitations

Several limitations affect this project:

* **Small dataset size**
* **Strong class imbalance**
* **Self-reported survey data**, which can be subjective
* Limited contextual features about patients or therapists

Additionally, therapist identity appearing as a predictive feature may reflect underlying biases or data artifacts rather than causal relationships.

---

# Future Improvements

Possible improvements include:

* Collecting more patient data, particularly 30+ session cases
* Incorporating therapist metadata or treatment types
* Including therapist-reported progress metrics
* Designing richer surveys to better capture patient progress
* Using temporal models to track symptom changes over time

---

# Repository Structure

```
project/
│
├── Final_Vers_FinalProjectDMA144.ipynb   # Main analysis notebook
├── README.md                            # Project documentation
└── data/                                # Data folder (not included)
```

---

# How to Run

## Option 1 — Google Colab

1. Upload the notebook to Google Colab
2. Upload the dataset CSV
3. Update the `pd.read_csv()` path
4. Run the notebook from top to bottom

## Option 2 — Run Locally

Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

Then run the notebook using Jupyter.

---

# Contributors

This project was completed as a group project for a data mining course.

My contributions focused on:

* Feature engineering improvements
* Therapist encoding for modeling
* Random Forest modeling and hyperparameter tuning
* Model evaluation and interpretation
* Feature importance analysis

---

# Conclusion

This project demonstrates the challenges of predictive modeling in healthcare-related datasets, particularly when dealing with **class imbalance and subjective survey data**.

While ensemble models showed promising results on balanced data, performance on the original dataset suggests that **additional data and features are needed before deploying such a model in real-world settings**.
