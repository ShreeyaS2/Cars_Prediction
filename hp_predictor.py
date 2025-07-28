from sklearn.ensemble import RandomForestRegressor #library for random forest regression
import pandas as pd #library for data manipulation and analysis
from sklearn.tree import DecisionTreeRegressor #library for decision tree regression
from sklearn.metrics import mean_absolute_error #library for calculating mean absolute error    
from sklearn.model_selection import train_test_split #library for splitting datasets into training and testing sets


file_path = 'https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv'
data= pd.read_csv(file_path) #read the csv file from the URL

features=['mpg', 'cyl'] #list of features to be used
x= data[features] #select the features from the dataframe
y=data.hp #select the target variable

# Using the prediction model over split datasets(training and testing)

# the random_state argument guarantees we get the same split every time we
# run this script.
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state = 1)

model = RandomForestRegressor(max_leaf_nodes=100, random_state=1) #create a random forest regression model
model.fit(train_x, train_y) #fit the model to the training data
val_predictions = model.predict(test_x) #make predictions using the model on the test data
print("Predictions for horsepower:", val_predictions) #display the predictions made by the model
print(mean_absolute_error(test_y, val_predictions)) #calculate and display the mean absolute error of the predictions
