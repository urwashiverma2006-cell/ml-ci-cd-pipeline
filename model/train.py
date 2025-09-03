import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

#Load dataset
data= pd.read_csv('data/iris.csv')

#preprocess the dataset
X = data.drop('Species' , axis=1)
y = data['Species']

#split the data into trainig and test sets
X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.2 , random_state=42)

#train a random forest model 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

#save the model
joblib.dump(model, 'model/iris_model.pkl')