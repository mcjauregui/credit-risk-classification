# Challenge 20 - Credit Risk Classification

## Overview of the Analysis

The purpose of this analysis is to evaluate how well a logistic regression model performs on a portfolio of loans. This is an instance of supervised learning, because we know which loans are actually healthy and which are high-risk. We test the model on two classes of loans: healthy and high-risk. We evaluate model performance with respect to two classes, individually. In other words, we subject the model to two tests. The first test is how well does it predict healthy loans. The second test is how well does it predict high-risk loans. The model could potentially perform well on one or the other class, or neither, or both. 
  
The independent variables (features) we have to work with are loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, and total_debt. 
There are 77,536 observations or rows in the dataset.   
The dataset can be described as follows:  

          loan_size  interest_rate  borrower_income  debt_to_income  \  
count  77536.000000   77536.000000     77536.000000    77536.000000     
mean    9805.562577       7.292333     49221.949804        0.377318    
std     2093.223153       0.889495      8371.635077        0.081519    
min     5000.000000       5.250000     30000.000000        0.000000   
25%     8700.000000       6.825000     44800.000000        0.330357    
50%     9500.000000       7.172000     48100.000000        0.376299   
75%    10400.000000       7.528000     51400.000000        0.416342    
max    23800.000000      13.235000    105200.000000        0.714829    
  
       num_of_accounts  derogatory_marks    total_debt   loan_status    
count     77536.000000      77536.000000  77536.000000  77536.000000   
mean          3.826610          0.392308  19221.949804      0.032243    
std           1.904426          0.582086   8371.635077      0.176646    
min           0.000000          0.000000      0.000000      0.000000    
25%           3.000000          0.000000  14800.000000      0.000000    
50%           4.000000          0.000000  18100.000000      0.000000    
75%           4.000000          1.000000  21400.000000      0.000000   
max          16.000000          3.000000  75200.000000      1.000000    
  
The dependent variable (outcomes or responses) is loan_status. A value of 0 indicates a healthy loan while a value of 1 indicates a high-risk loan. A high-risk loan implies a loss for the lender while a healthy loan implies a profit. Lenders seek to maximize the number of healthy loans in their portfolios and minimize the number of high-risk loans.   

Machine Learning Process:  
a. Read the CSV file representing the loan portfolio into a Pandas DataFrame  
b. Get descriptive statistics for the DataFrame  
c. Separate the data into labels and features by spliting the data into X (features) and y (target). The features included all columns of the data frame except for loan_status, our y variable.  
d. Split the data into train and test set using train_test_split. The result is four data sets: X_train, X_test, y_train, y_test. We assign a value of 1 to the random_state function. We stratify y because it's a means of maintaining the distribution of classes (0 and 1) in both sets (training and testing). We do this in an effort to ensure the proportion of both classes in the original dataset is maintained in both the training and testing sets. This helps us avoid overfitting or underfitting to one class, which can hinder unbiased model performance evaluation.
e. Define a logistic regression model with a random state of 1 and fit the model with the training data (X_train and y_train). 
f. Instantiate the Logistic Regression model with the LogisticRegression method, and fit the logistic regression model with the training data.  
g. Generate testing predictions with the logistic regression model on the test data (X_test).   
h. The first step in evaluating the model results is generating a confusion matrix out of the y_test data and testing predictions. The confusion matrix takes the following shape, and the actual ouput is included below:    

                          Predicted Positive            Predicted Negative  
                                0                               1  
Actual Positive   0       True Positive (14,951)      False Negative (57)  
#Actual Negative  1       False Positive (59)         True Negative (441) 

True Positive (TP): The model correctly predicted instances of the positive class.  
True Negative (TN): The model correctly predicted instances of the negative class. 
False Positive (FP): The model incorrectly predicted instances of the positive class when they were actually negative  
False Negative (FN): The model incorrectly predicted instances of the negative class when they were actually positive   

A well-performing model will maximize the True Negative and True Positive values and minimize the False Negative and False Positive values.  

i. The second evaluation step is generating a classification report, which - in part - uses the confusion matrix values to calculate the model's precision and recall. Precision = True Positives / (True Positives + False Positives) and Recall = True Positives / (True Positives + False Negatives). 

              precision    recall  f1-score   support  

           0       1.00      1.00      1.00     15008  
           1       0.89      0.88      0.88       500  
 
    accuracy                           0.99     15508 
   macro avg       0.94      0.94      0.94     15508 
weighted avg       0.99      0.99      0.99     15508  

## Results
* The model's accuracy is 99%. This means it correctly classified the vast majority of the the loan applications.  

* For the healthy loan predictions (row 0), the model has a 100% score in both precision and recall, which is the highest possible.   
    * The 100% precision score means the model classified 100% of the healthy loans as healthy and had no false positives (misclassification of high-risk loans as healthy).   
    * The 100% recall score means the model correctly predicted all truly healthy loans as healthy. It made no false negatives, which means it did not classify any of the healthy loans as high-risk.   
  
* The model was less precise and had poorer recall when it came to predicting high-risk loans, with scores of 89% and 88%, respectively.   
    * This precision score means it correctly predicted only 89% of the loans it predicted to be high-risk. In other words, the model mis-classified 11% of its high-risk predictions as high risk when they were actuall healthy (false positives).   
    * The recall score of 88% means the model correctly identified 88% of all the truly high-risk loans as high-risk (true positives). The other 12% it mis-classified as healthy (false negatives).  

## Summary  
* This logistic regression model does an excellent job at predicting the healthy loans and a decent job at predicting the high-risk loans. As we can see from the precision and recall scores, it performs better at predicting healthy loans than high-risk loans.   
* I recommend the model to lenders who prioritize writing good loans to minimizing bad loans.   
* For lenders who are extremely risk averse, I would recommend finding another model that has higher presicion and recall on the high-risk class.    
