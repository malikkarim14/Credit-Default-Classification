# Credit Default Classification

## Description

I compared the classification algorithm with a case study of default credit payment prediction in Taiwanese society. Random Forest, Naive Bayes, and Support Vector Machine are the algorithms being compared. The dataset contains 23 independent variables, but after going through the variable selection process with random forest, I only used 16 variables.

## Data Source
Data was gained from UCI Machine Learning Repository that you can look in [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Atributte Information
This study employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
- X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
- X2: Gender (1 = male; 2 = female).
- X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
- X4: Marital status (1 = married; 2 = single; 3 = others).
- X5: Age (year).
- X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
- X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005.
- X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

## Source Code

You can see my code in [here](https://github.com/malikkarim14/Credit-Default-Classification/blob/6c132cc7d26e6ab18c27be3b9f37b104d7ca84f1/Deteksi%20Kredit%20Macet.py)

## Result

To see the paper of my study you can click [here](https://github.com/malikkarim14/Credit-Default-Classification/blob/6c132cc7d26e6ab18c27be3b9f37b104d7ca84f1/Deteksi%20Kredit%20Macet.pdf)




#### Creation Date

June 25th, 2020
