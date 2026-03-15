# Cricket Match Outcome & Score Predictor

An end-to-end machine learning pipeline designed to predict cricket match results and total scores for **ODI** and **T20** formats. The project features custom-built ensemble models and an interactive Streamlit dashboard.

---

##  Overview
Predicting cricket outcomes is challenging due to the high number of variables involved (pitch, weather, team form). This project addresses this by using a dual-model approach:
1.  **Classification:** To determine the winner and categorize pitch conditions.
2.  **Regression:** To estimate the final total score of the match.

##  Tech Stack
* **Language:** Python
* **Modeling:** Custom Decision Trees, Random Forest, and Gradient Boosting.
* **API:** OpenWeather API (for atmospheric match context).
* **UI/UX:** Streamlit (interactive dashboard).
* **Data Handling:** Pandas, NumPy, Scikit-learn.

---

##  Model Architectures

### 1. BetterDecisionTreeClassifier
The foundation of our classification system. 
* **Logic:** Uses **Gini Impurity** to create binary splits in the data.
* **Splitting Criterion:**
  $$Gini = 1 - \sum p_{i}^2$$
* **Purpose:** Serves as the base learner for the ensemble forest.

### 2. SimpleRandomForest (Classification)
An ensemble of multiple Decision Trees designed to handle the variance of sports data.
* **Mechanism:** Uses **Bootstrapping** and **Feature Subsampling** to ensure tree diversity.
* **Voting:** Final prediction is determined by the **Majority Vote (Mode)** of all trees.
* **Benefit:** Reduces overfitting compared to a single decision tree.

### 3. SimpleGradientBoostingRegressor (Regression)
A sequential model used to predict the numerical **Total Score**.
* **Approach:** It fits a sequence of shallow trees to the **residuals** (errors) of previous trees.
* **Optimization:** Minimizes the **Sum of Squared Errors (SSE)** incrementally.
* **Update Rule:**
  $$F_{m}(X) = F_{m-1}(X) + \eta \times T_{m}(X)$$
  *(Where $\eta$ is the learning rate)*

---

##  Evaluation & Metrics
The project evaluates model health through several diagnostic visualizations:
* **R² Score:** Our ODI regression model achieves high variance explanation ($R^2 \approx 1.00$).
* **Residual Analysis:** Errors are verified to be normally distributed and centered around zero (Mean $\approx -0.2$).
* **Feature Importance:** Identification of which factors (e.g., Toss, Venue, Temperature) influence the score most.

##  How to Run
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/varunnavie/cryc-lytics.git](https://github.com/varunnavie/cric-lytics.git)
   cd cricket-prediction
