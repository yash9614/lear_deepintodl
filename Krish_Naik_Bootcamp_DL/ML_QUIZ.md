# Machine Learning Concepts Quiz

40 multiple-choice questions covering core ML topics (hyperparameter tuning, metrics, Random Forest, DBSCAN, VIF, XGBoost, bagging, boosting, evaluation, feature engineering, etc.).

**Use this for quick self-review.** Correct answers are marked below each question.

---

**Q1. How does GridSearchCV determine the best hyperparameters for a model?**
- By consulting expert recommendations
- By using default parameters only
- By selecting random combinations to minimize time
- By testing every possible combination in the specified parameter grid

**Answer:** ✅ By testing every possible combination in the specified parameter grid

---

**Q2. You are an environmental data scientist building a machine learning model to predict the tsunami in a specific region. Which model metric would you give the highest importance?**
- Precision
- Recall
- Accuracy
- Number of Likes on Your LinkedIn Post

**Answer:** ✅ Recall  
*(Missing a tsunami is catastrophic; false negatives are extremely costly, so high recall is prioritized.)*

---

**Q3. Which metric on the ROC curve is directly affected by changing the threshold for classifying a positive class?**
- The precision of the model
- Only the area under the curve
- The fate of the Humans
- Both true positive rate and false positive rate

**Answer:** ✅ Both true positive rate and false positive rate

---

**Q4. What type of machine learning algorithm is a random forest?**
- Clustering algorithm
- Regression and classification algorithm
- Reinforcement learning algorithm
- Random Rocker Algorithm

**Answer:** ✅ Regression and classification algorithm

---

**Q5. What parameter in DBSCAN determines the maximum distance between two points for them to be considered as part of the same neighbourhood?**
- minPts
- eps
- corePts
- distance

**Answer:** ✅ eps

---

**Q6. Which technique is commonly used to normalize different scales of feature values in machine learning preprocessing?**
- Clustering
- Min-Max Scaling
- One-Hot Encoding
- Data Dance

**Answer:** ✅ Min-Max Scaling

---

**Q7. Which linkage method in hierarchical clustering considers the shortest distance between any two points in different clusters?**
- Average Linkage
- Single Linkage
- Complete Linkage
- Love Linkage

**Answer:** ✅ Single Linkage

---

**Q8. Why might one choose to use a random forest over a single decision tree?**
- For a simpler explanation of the model’s decisions
- To decrease the accuracy for more stable predictions
- Because he/she hates being single
- To reduce the risk of overfitting associated with single decision trees

**Answer:** ✅ To reduce the risk of overfitting associated with single decision trees

---

**Q9. What is the main benefit of Stratified K-Fold Cross-Validation?**
- Increases computational speed
- Ensures representative folds by class percentage
- Allows more folds
- It crosses the speed limit every time

**Answer:** ✅ Ensures representative folds by class percentage

---

**Q10. Why is a high VIF problematic in a regression model?**
- It indicates that the feature is less important to the model
- It implies that the model will be extremely accurate
- It shows that the data has no interest in being trained
- It indicates multicollinearity which will distort the interpretation of coefficients and make them unstable

**Answer:** ✅ It indicates multicollinearity which will distort the interpretation of coefficients and make them unstable

---

**Q11. How does bagging help in handling outliers in the dataset?**
- By focusing more on outsiders in the park
- By ignoring outliers completely
- By amplifying the effect of outliers
- By reducing the impact of outliers through aggregation

**Answer:** ✅ By reducing the impact of outliers through aggregation

---

**Q12. Which of the following is a key characteristic of a good loss function for gradient descent optimization?**
- It should be convex and differentiable
- It should be non-convex
- It should be non-differentiable
- It should make you lose 50% of your money on Adani stock

**Answer:** ✅ It should be convex and differentiable

---

**Q13. Given the dataset below, where '1' indicates that an individual owns a car and '0' indicates they do not, calculate the precision of the model’s predictions for car ownership. Select the correct precision percentage from the options provided:**
- 83%
- 70%
- 75%
- 0.008%

**Answer:** ✅ 75%  
*(Typical precision calculation from a confusion matrix scenario; confirm with the original table if available.)*

---

**Q14. Why might a data scientist choose to use RandomizedSearchCV for optimizing hyperparameters?**
- To ensure faster execution by exploring fewer parameter combinations
- To achieve the highest possible accuracy without regard to time
- To avoid cross-validation completely
- When they get bribed

**Answer:** ✅ To ensure faster execution by exploring fewer parameter combinations

---

**Q15. You can use this technique to detect outliers**
- IQR
- Standard Deviation
- Clustering
- All of the Above

**Answer:** ✅ All of the Above

---

**Q16. What is an acceptable practice when managing features with high VIF?**
- Ignore the VIF values as they do not impact predictions
- Keep increasing the dataset size until VIF values drop
- Continuously eliminate high VIF features (one at a time) and recompute until VIFs are acceptable
- Use only categorical variables to reduce VIF

**Answer:** ✅ Continuously eliminate high VIF features (one at a time) and recompute until VIFs are acceptable

---

**Q17. Why might a model with a very high R-squared value on training data not perform well on unseen testing data?**
- The model is likely underfitting the training data
- Because it is like that student who doesn't like participating in tests and exams!
- The model training did not involve enough epochs
- The model is likely overfitting the training data

**Answer:** ✅ The model is likely overfitting the training data

---

**Q18. In the context of machine learning and statistics, what does "bootstrapping" refer to?**
- A technique for dimensionality reduction
- A method of sampling with replacement
- A type of cross-validation
- Getting blessings from your parents before starting a new project

**Answer:** ✅ A method of sampling with replacement

---

**Q19. What aspect of XGBoost allows it to process large datasets quickly?**
- The absence of tree pruning
- Serial processing of each tree
- It's hunger to succeed
- Parallel processing and block structure to optimize computation

**Answer:** ✅ Parallel processing and block structure to optimize computation

---

**Q20. Which loss function is typically used in gradient boosting for classification tasks?**
- Logistic Loss
- Mean Squared Error
- Cross-Entropy Loss
- Latest Loss

**Answer:** ✅ Logistic Loss / Cross-Entropy Loss  
*(Both are closely related; logistic loss is commonly used for binary classification in gradient boosting.)*

---

**Q21. For text classification (such as spam classification), this model gets a preference -**
- XGBoost
- FBI Agent
- Logistic Regression
- Naive Bayes

**Answer:** ✅ Naive Bayes

---

**Q22. Which algorithm is better when there are complex, non-linear relationships in your data?**
- Logistic Regression
- XGBoost
- Linear Regression
- A marriage counselor

**Answer:** ✅ XGBoost

---

**Q23. Which technique combines the predictions of several base estimators (in parallel fashion) to improve model's robustness?**
- Stacking
- Bagging
- Boosting
- Calling Astrologer

**Answer:** ✅ Bagging

---

**Q24. What is the primary purpose of a confusion matrix in machine learning?**
- To predict future values
- To confuse new data scientists
- To calculate data speed
- To visualize the performance of a classification model

**Answer:** ✅ To visualize the performance of a classification model

---

**Q25. How does DBSCAN determine if a cluster has been fully formed?**
- When all points are at least eps distance apart
- After it receives a confirmation mail
- Once all points in the dataset have been visited
- When no new points can be added to any cluster according to the eps and min_samples criteria

**Answer:** ✅ When no new points can be added to any cluster according to the eps and min_samples criteria

---

**Q26. What is a key benefit of using XGBoost for machine learning projects?**
- It significantly reduces the need for data preprocessing
- It offers high performance and scalability, particularly suitable for large datasets
- It simplifies model interpretations by creating fewer trees
- It automates the feature selection process

**Answer:** ✅ It offers high performance and scalability, particularly suitable for large datasets

---

**Q27. What does the term 'epochs' mean in machine learning training?**
- Total Number of Balls in a over
- Number of Inputs
- Total Errors
- Complete passes through the training dataset

**Answer:** ✅ Complete passes through the training dataset

---

**Q28. What does the variable pi represent in the Log Loss formula?**
- Actual value
- Probability of a good joke
- Number of records
- Predicted probability

**Answer:** ✅ Predicted probability

---

**Q29. These two algorithms are used widely in the industry due to their robust performance and their ability to generalize well.**
- XGBoost and Random Forest
- Logistic and Linear Regression
- Cricket and Tennis
- Decision Tree and Naive Bayes

**Answer:** ✅ XGBoost and Random Forest

---

**Q30. In what scenario might you prefer using the F1 score over accuracy for evaluating a classification model?**
- When the classes are balanced
- When the dataset is highly imbalanced
- When there are high costs associated with false positives
- In F1 car racing game

**Answer:** ✅ When the dataset is highly imbalanced

---

**Q31. What does the F1 Score provide in terms of model evaluation?**
- It emphasizes the model's sensitivity to outliers
- It exclusively measures the model’s accuracy
- It combines both precision and recall into a single measure, balancing the two metrics
- Top speed of the F1 race cars

**Answer:** ✅ It combines both precision and recall into a single measure, balancing the two metrics

---

**Q32. What does a dendrogram show in hierarchical clustering?**
- Your Future
- Cluster merging sequence
- Data geography
- Complexity level

**Answer:** ✅ Cluster merging sequence

---

**Q33. Which evaluation metric could be misleading if used alone for a model that is overfitting?**
- Training accuracy
- Testing accuracy
- Cross-validated accuracy
- AUC-ROC

**Answer:** ✅ Training accuracy

---

**Q34. What role does the learning rate play in gradient boosting?**
- It controls how quickly the model learns, affecting how each tree influences the overall prediction
- It specifies the depth of each tree
- It determines the number of trees in the model
- Amount which a person needs to pay to learn

**Answer:** ✅ It controls how quickly the model learns, affecting how each tree influences the overall prediction

---

**Q35. What is the main purpose of the learning rate in gradient descent?**
- To control the number of iterations
- To add noise to the gradient updates
- To determine the size of the steps taken during optimization
- To decide how often the model should take coffee breaks

**Answer:** ✅ To determine the size of the steps taken during optimization

---

**Q36. What does the term "bias" refer to in machine learning?**
- The variance in the model's predictions
- The error due to overly simplistic assumptions in the learning algorithm
- The error on the training set
- The model's performance in a fashion show

**Answer:** ✅ The error due to overly simplistic assumptions in the learning algorithm

---

**Q37. Which method is used to evaluate a classification model's performance?**
- Mean Squared Error
- Root Mean Squared Error
- Precision and Recall
- Asking your cat for feedback

**Answer:** ✅ Precision and Recall

---

**Q38. What is the purpose of feature creation in feature engineering?**
- To reduce the overall data available for modeling
- To generate new features from existing data to enhance predictive power
- To simplify the model by eliminating most features
- To have a safe Future for your child

**Answer:** ✅ To generate new features from existing data to enhance predictive power

---

**Q39. What is the purpose of using streamlit framework in ML Projects?**
- To enhance back-end processing speed
- For detailed 3D modeling
- For rapid front-end design without deep front-end knowledge
- To restrict user access to applications

**Answer:** ✅ For rapid front-end design without deep front-end knowledge

---

**Q40. You are a data scientist working for a digital marketing agency building a model that shows ads to the correct target audience. To ensure that ads are shown to users who are most likely to be interested and to minimize wasted ad spend, which model metric should you focus on the most?**
- Precision
- Recall
- F1 Race
- Accuracy

**Answer:** ✅ Precision  
*(High precision minimizes wasted ad spend by ensuring shown ads are relevant to interested users.)*

---

## Quick Review Notes
- Cross-reference with:
  - [NOTES.md](./NOTES.md)
  - [MOST_IMPORTANT_CONCEPTS.md](./MOST_IMPORTANT_CONCEPTS.md)
  - [INTERVIEW_QUESTIONS.md](./INTERVIEW_QUESTIONS.md)
  - [QUIZ.md](./QUIZ.md) (Deep Learning quiz)
- Key topics: GridSearchCV vs RandomizedSearchCV, Recall vs Precision, ROC, Random Forest, DBSCAN (eps), Min-Max Scaling, Hierarchical Clustering linkages, VIF/multicollinearity, Bagging, XGBoost, Stratified K-Fold, F1 Score, Bias-Variance, Streamlit, Feature Engineering.

Last updated: 2026-11-23