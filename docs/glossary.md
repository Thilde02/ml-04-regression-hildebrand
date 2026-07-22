# Glossary

Use this page to record terms and ideas that help you understand
professional analytics projects.

This project covers regression: building and evaluating models
that predict a continuous numeric value from input features.

Pro-tip: Expand the VS Code **Outline** view (below the navigator on the right)
to see this file organization at-a-glance.

## Regression Evaluation

### residual

A residual is the difference between a model's predicted value and the true value.
Examining residuals reveals where a model is consistently over- or under-predicting.

### MAE (Mean Absolute Error)

MAE is the average of the absolute differences between predicted and true values.
It is easy to interpret: an MAE of 5 means predictions are off by 5 units on average.

### RMSE (Root Mean Squared Error)

RMSE is the square root of the average squared differences between predicted and true values.
It penalizes large errors more heavily than MAE.
A lower RMSE means predictions are closer to the true values.

### R-squared (R²)

R-squared measures how much of the variance in the target the model explains.
A value of 1.0 means perfect predictions; a value of 0.0 means the model
does no better than predicting the mean every time.
Negative values are possible when the model performs worse than the mean.

### baseline model

A baseline model is the simplest possible prediction strategy,
such as always predicting the mean of the training target.
A good model should outperform the baseline; if it does not,
the features or approach need revisiting.

## Regression Models

### linear regression

Linear regression fits a straight line (or hyperplane) through the data
to predict a continuous target.
It is fast and interpretable but assumes a linear relationship between features and target.

### decision tree regressor

A decision tree regressor predicts a continuous value by partitioning
feature space into regions and predicting the mean of each region.
It can capture nonlinear relationships but may overfit.
