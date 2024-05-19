import numpy as np
from Test import t
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, \
    StackingRegressor, HistGradientBoostingRegressor,VotingRegressor


class Surrogate:

    def __init__(self,x_train,y_train,n_estimators):
        self.x_train = x_train
        self.y_train = y_train
        self.svr = self.fit(n_estimators)

    def fit(self,n_estimators):
        y_train = np.array(self.y_train).reshape(-1, 1)
        y_train = MinMaxScaler().fit_transform(y_train).ravel()
        svr = GradientBoostingRegressor(n_estimators=n_estimators)
        svr.fit(self.x_train,y_train)
        return svr

    def _predict(self,configurations):
        svr_predict = self.svr.predict(configurations)
        return svr_predict



s = 0
t(s)
print(s)