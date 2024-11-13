## **Overview**

This is a multiple linear regression performed on the average rating of all Serie A goalkeepers from a [fantasy football online game](https://www.fantacalcio.it/). 

## **Technologies**

Used python libraries such as scikit-learn for the regression, pandas, matplotlib and seaborn for visualizations.

## **Objectives**

The main goal of this project was to build a robust regression and identify which goalkeeper stats give the best explanation for their average rating in a Serie A season in a fantasy football game. This way if a new goalkeeper starts playing for a Serie A team we can predict his expected average rating given his previous season stats.

The dependent variable is the average rating of the goalkeeper for a full season, for seasons from 2019 to 2024. These ratings are provided by a group of journalists selected by the game provider.

The independent variables are advanced goalkeeper stats provided by [fbref.com](http://fbref.com).

## Analysis

From our initial dataset preview, FM column is our dependent value and the rest are independent:

![image](https://github.com/user-attachments/assets/9ae07b38-e14c-4c94-bfe1-9e3b8e7dfb50)

Pandas DF preview

Sequential Feature Selector is used to select the most important features in the dataset for the predictive model. This selector adds features step by step based on their contribution to the model's performance. It begins with an empty feature set, and iteratively adds one feature at a time which best improves the R-squared score. It stops when adding a feature does not improve the metric anymore.

After this we calculate a correlation matrix, and then remove iteratively each variable with a VIF of more than 10 to account for multicollinearity:

![image](https://github.com/user-attachments/assets/38f04e96-326c-46fc-bcce-0037b2a21ffd)

Correlation Matrix

![image](https://github.com/user-attachments/assets/44bce233-a263-4296-8da1-6ade6be25268)

Multicollinearity corrected variables

### Model Summary

![image](https://github.com/user-attachments/assets/693eb977-4c69-4533-a2c7-6febee37fe34)

With an adjusted R-square for the regression of 0.735, there are 4 statistically significant variables at 95% confidence level:

- W (Wins) - it is expected that a win in most cases means a good performance from the goalkeeper
- PKA (Penalty Kicks Allowed) - we expect a negative coefficient for this variable since allowing a goal from a penalty kick will decrease the goalkeeper’s rating
- Save%.1 (Penalty Kicks Allowed/ Penalty Kick attempts) - a higher save percentage indicates a goalkeeper of a greater ability
- /90 (Post-Shot Expected Goals minus Goals Allowed per 90 minutes) - a positive number suggests an above average ability to stop shots on target

---

Doing a train-test split a value of 0.71 is observed for the R-squared:

![image](https://github.com/user-attachments/assets/bafcf082-cb86-4bc6-bd4a-31c2ac2a06a8)

And in a cross-validation test a score of 0.83, with a standard deviation of 0.117:

![image](https://github.com/user-attachments/assets/d97ddaea-08dc-44d6-8016-15a2140c8e46)

---
