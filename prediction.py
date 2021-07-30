import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

advertising = pd.read_csv('company.csv')

x=advertising.iloc[:,:-1].values
y=advertising.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, test_size=0.3, random_state=100)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

import pickle
pickle.dump(regressor,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))