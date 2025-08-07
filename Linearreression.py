#Importing libraries and uploading the data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from google.colab import files
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
uploaded = files.upload()
file_name = next(iter(uploaded))
data = pd.read_csv(file_name)

#Encoding defining features and target , split the dataset , create and train the model , making prediction and evaluating the model 
data_encoded = pd.get_dummies (data , drop_first = True)
x = data_encoded.drop('price' , axis = 1)
y = data_encoded['price']
xtrain , xtest , ytrain , ytest = train_test_split (x , y ,test_size = 0.2 , random_state = 42)
model = LinearRegression ()
model.fit(xtrain , ytrain)
y_pred = model.predict (xtest)
mae = mean_absolute_error (ytest , y_pred)
mse = mean_squared_error = (ytest , y_pred)
r2 = r2_score (ytest , y_pred)
print (f'Mean Absolute error : {mae}')
print (f'Mean Squared error : {mse}')
print (f'R2 score : {r2}')

##coifficent and intercept
print ('Intercept (bo)' , model.intercept_)
coiffiecents = pd.Series(model.coef_ , index = x.columns)
print ('Coifficent value : ' , coiffiecents.sort_values(ascending = False))

#Visulaizing coifficient as bar charts
coiffiecents.sort_values().plot(kind = 'barh' , figsize = (10 , 8) , title = 'Feature importance in price predictions')
plt.xlabel ('Effect on price')
plt.grid(True)
plt.tight_layout ()
plt.show ()