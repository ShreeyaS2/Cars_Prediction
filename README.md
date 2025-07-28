# Cars_Prediction

"""Random forest regression is a model that uses 
multiple decision trees to make predictions. It 
calculates the mean of all the predictions of the 
individual decision trees to produce a final 
prediction. This approach helps to reduce 
overfitting and improve the accuracy of predictions 
compared to a single decision tree."""

Used libraries- panda, skit.learn
Tools used- RandomForestRegressor, mean_absolute_error, train_test_split

The following program uses the RandomForestRegressor model from skit.learn to analyse a dataset with various features of cars. The featured data(mileage per gas, cylinders used) is trained with the target(horsepower) to predict the same for testing data.
The predictions are then printed in tabular form, and the error margin is tested with the actual data as well. 
