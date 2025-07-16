# 🧮 Integrating Agentic AI and Machine Learning

## 🌟 Overview

[![Watch the video](https://img.youtube.com/vi/tozReWteUAQ/hqdefault.jpg)](https://www.youtube.com/watch?v=tozReWteUAQ)

Click the thumbnail above to watch the video on YouTube.

🧠 Coal Production Prediction: Mastering the Depths with AutoML and Agent AI
🌟 Overview
Welcome to a groundbreaking exploration in coal production forecasting, where the rugged expertise of mining meets the precision of advanced machine learning. Leveraging data from cleaned_coalpublic2015.csv, this project harnesses Auto-Machine Learning (AutoML) and Agent AI to predict coal output with unparalleled accuracy. Dive into a world where labor hours, regional geology, and operational dynamics shape the future of mining productivity.
🌍 Background
In the mining industry, coal production is a complex interplay of human effort, geological variance, and operational efficiency. Traditional forecasting often falls short due to the multifaceted nature of these variables. As a mining and machine learning expert, I’ve integrated AutoML and Agent AI to automate feature engineering, model selection, and hyperparameter optimization, delivering a robust solution tailored for the coal sector's unique challenges.
🎯 Objectives

Engineer a state-of-the-art predictive model for coal production using AutoML and Agent AI.
Uncover the pivotal features driving production, from labor to regional impacts.
Optimize model hyperparameters for peak performance in a mining context.
Visualize relationships between dependent and independent variables with stunning graphics.

📊 Methods
Adopting the STAR framework with a mining and ML lens:

Situation: The dataset comprises 853 records with 16 initial features, expanded to 199 post-preprocessing, reflecting mining intricacies.
Task: Process raw data, split into 70% training and 30% test sets, benchmark multiple ML models, tune the top performer, and evaluate with mining-relevant metrics.
Action: 
Preprocessed data by removing non-predictive fields and encoding categorical variables like mine type and region.
Benchmarked models (Gradient Boosting, Random Forest, etc.) using 5-fold cross-validation.
Tuned the Gradient Boosting model with grid search for optimal mining predictions.
Evaluated with R², MAE, MSE, and RMSE, tailored to mining output accuracy.


Results: Gradient Boosting emerged with a test R² of 0.8511, highlighting labor hours and average employees as key drivers.

Mathematical Foundation
The prediction model employs Gradient Boosting, where the output ( \hat{y} ) for input ( x ) is:
[ \hat{y} = \sum_{m=1}^{M} f_m(x), ]
with ( f_m(x) ) as individual decision trees and ( M ) as the total trees. The loss function optimized is:
[ L = \sum_{i=1}^{n} l(y_i, \hat{y}i) + \sum{m=1}^{M} \Omega(f_m), ]
where ( l ) is the mean squared error, and ( \Omega ) regularizes to prevent overfitting—critical for mining data variability.
Visualizing the Model
Behold a dynamic graph illustrating dependent (log_production) and independent variables (e.g., Labor_Hours, Average_Employees):

🌟 Support This Project
Follow my mining ML journey on GitHub: 
Star this repo: 
Connect on LinkedIn: 
Show your support by clicking above!
Pipeline started at: 2025-07-16 07:42:00
Loading coal production data...Data loaded from: data/cleaned_coalpublic2015.csvDataset shape: (853, 16)Columns: ['Year', 'Mine_Name', 'Mine_State', 'Mine_County', 'Mine_Status', 'Mine_Type', 'Company_Type', 'Operation_Type', 'Operating_Company', 'Operating_Company_Address', 'Union_Code', 'Coal_Supply_Region', 'Production_(short_tons)', 'Average_Employees', 'Labor_Hours', 'log_production'][2025-07-16 07:42:00] Data Loading completed in 0.02 secondsPreprocessing data...Dropped columns: ['Year', 'Mine_Name', 'Operating_Company', 'Operating_Company_Address', 'Production_(short_tons)']Features after preprocessing: 199Sample features: ['Average_Employees', 'Labor_Hours', 'Mine_State_Alaska', 'Mine_State_Arizona', 'Mine_State_Arkansas', 'Mine_State_Colorado', 'Mine_State_Illinois', 'Mine_State_Indiana', 'Mine_State_Kansas', 'Mine_State_Kentucky (East)'][2025-07-16 07:42:00] Data Preprocessing completed in 0.02 secondsSplitting data into 70% train and 30% test...Training set: (597, 199)Test set: (256, 199)[2025-07-16 07:42:00] Data Splitting completed in 0.00 secondsComparing models with 5-fold cross-validation...Training Random Forest...Training Gradient Boosting...Training Extra Trees...Training Linear Regression...Training Ridge Regression...Training Lasso Regression...Training Elastic Net...Training Decision Tree...Training K-Nearest Neighbors...Training Support Vector Regression...
Model Comparison Results:
                Model  Mean_R2  Std_R2  Min_R2  Max_R2
    Gradient Boosting   0.8871  0.0097  0.8792  0.9056
        Random Forest   0.8783  0.0152  0.8576  0.9035
          Extra Trees   0.8720  0.0115  0.8542  0.8889
  K-Nearest Neighbors   0.8549  0.0211  0.8239  0.8826
        Decision Tree   0.7922  0.0320  0.7317  0.8262

Support Vector Regression   0.7410  0.0341  0.6838  0.7732         Ridge Regression   0.6219  0.0289  0.5885  0.6666        Linear Regression   0.5328  0.0525  0.4561  0.6197         Lasso Regression   0.4157  0.0119  0.4005  0.4312              Elastic Net   0.4140  0.0107  0.3992  0.4256[2025-07-16 07:42:07] Model Comparison completed in 7.00 seconds
✓ Best performing model: Gradient BoostingTuning hyperparameters for Gradient Boosting...Best parameters: {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.9}Best CV R2 score: 0.8901[2025-07-16 07:42:23] Model Hyperparameter Tuning completed in 15.84 secondsTraining final model...[2025-07-16 07:42:23] Final Model Training completed in 0.09 secondsEvaluating model performance...
Test Set Performance:
R² Score: 0.8511MAE: 0.6477MSE: 0.8873RMSE: 0.9420
Training Set Performance:
R² Score: 0.9495MAE: 0.3889
Evaluation plots saved as 'model_evaluation.png'

---

<div align="center">


*Remember: Every expert was once a beginner. Your programming journey is unique, and we're here to support you every step of the way.*

## 🌟 Support This Project
**Follow me on GitHub**: [![GitHub Follow](https://img.shields.io/github/followers/Harrypatria?style=social)](https://github.com/Harrypatria?tab=followers)
**Star this repository**: [![GitHub Star](https://img.shields.io/github/stars/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab?style=social)](https://github.com/Harrypatria/SQLite_Advanced_Tutorial_Google_Colab/stargazers)
**Connect on LinkedIn**: [![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harry-patria/)

Click the buttons above to show your support!

</div>
