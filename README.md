# CFO-forcasts

<<<<<<< HEAD

## Report of the project

 

### Approach and strategy
 
The model we build is based on the RNN network and LSTM layers.Since we deal with a problem of time series with many variables, the approach adopeted is to develop a model to forecast a multivariate time series.

This will allow to use many inputs (all the variables in question in case it is expected to improve the predictions) and enventually other external factors in the future.
    
For churn and leavers variables : each target is predicted alone based only on its own history and price increases' history.
    
For other four variables: each one of them are considered as a multivariate time series.Each variable is predicted using it's own history and the history of others alongside with price increases' record.

 
### Spliting the data set into train and validation :
   
   we take the data prior to 2018-09-01 to train our model.This choice is justified by first by the small size of dataset as well as the objective to prevent on-off shock price in 2018-09-01 to impact and bias our model.
       The task of this challenge consits of prediction of the next 12 month.Which means if we have a large data set, we will use 12-month pediction values and 12-month real values to compute the loss function (rmse and mape) for validation set.But in our case, the size of validation set allows to construct a target future whith a size of 6-months, in addition to this, 

 
### Quantify the impact to not impact our future prediction:

Since our model can handel multivariate series, we add a column `['increase_impact']` to our dataset in which we map the dates with increased price by 1 and others with 0.

when we compute the loss for validation set we don't include the range that hase undergone the impact of on-off shock .


### On-off shock impact on future prediction :

The test split was performed without including this priode between september and december in the training set

In addition to not bias the evaluation of the model, the period that overlaps with the impact of this on-off shock is excluded from the data validation set when computing losses.

We remove periode of the on-off shock that is conidered as anomalie (will not be repeated in the future) which mean that only the last 3 month of the validation will be used to compute the rmse of the model on the validation set.
=======
>>>>>>> a83e9ab98dda3123eceb238675c3b590a30fdd8d
