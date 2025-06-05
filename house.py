import pandas as pd
from sklearn.linear_model import LinearRegression  # For regression modeling

train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
# Select features and target
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X_train = train_df[features]
y_train = train_df['SalePrice']

#training the model
model=LinearRegression()
model.fit(X_train,y_train)

#predicting
X_test=test_df[features]
predict =model.predict(X_test)

#creating dataframe with prediction
result=pd.DataFrame({
    'Id':test_df['Id'],
    'predicted_saleprice' :predict
})

#Display the predicted values
print(result)


