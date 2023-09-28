# Import lib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
class Model1():

    def __init__(self) -> None:
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        pass

    # Partitioning into features and target variable
    def feature_target_split(self):
        # Loading the dataset
        raw_df = pd.read_csv("data/dataTrain_carListings.csv")

        X = raw_df.drop(columns=["Price"],axis=1)
        y = raw_df["Price"]
        return X,y
    

    # Training and test data split
    def split_train_test(self,X,y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train = pd.concat([X_train,pd.get_dummies(X_train["Make"]),pd.get_dummies(X_train["Model"]),pd.get_dummies(X_train["State"])],axis=1)
        X_train.drop(columns=["State","Make","Model"],inplace=True,axis=1)
        X_test = pd.concat([X_test,pd.get_dummies(X_test["Make"]),pd.get_dummies(X_test["Model"]),pd.get_dummies(X_test["State"])],axis=1)
        X_test.drop(columns=["State","Make","Model"],inplace=True,axis=1)
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        return X_train, X_test, y_train, y_test


    # Train the model
    def fit_random_forest(self):
        # Initializing randomforest regressor
        model = RandomForestRegressor(n_estimators=10)

        model.fit(self.X_train, self.y_train)
        self.model = model
        pass


    # Predicting on test data
    def predict(self,X_test):

        y_pred = self.model.predict(X_test)
        return y_pred


    # Accuracy of the model
    def MAE_percent_error(self,y_test,y_pred):
        accuracy = mean_absolute_percentage_error(y_test, y_pred)
        return accuracy