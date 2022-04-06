import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# read data
breast_cancer_datasets = sklearn.datasets.load_breast_cancer()

data_frame = pd.DataFrame(breast_cancer_datasets.data, columns = breast_cancer_datasets.feature_names)

data_frame['label'] = breast_cancer_datasets.target

data_frame.shape
data_frame.info()
data_frame.isnull().sum()
data_frame.describe()
data_frame['label'].value_counts()
data_frame.groupby('label').mean()

X = data_frame.drop(columns='label', axis=1)
y = data_frame['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# train model
model = LogisticRegression()
model.fit(X_train, y_train)

# accuracy train of model
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)
training_data_accuracy

# accuracy test of model
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)
test_data_accuracy

# input data for classification
input_data = (20.57,17.77,132.9,1326,0.08474,0.07864,0.0869,0.07017,0.1812,0.05667,0.5435,0.7339,3.398,74.08,0.005225,0.01308,0.0186,0.0134,0.01389,0.003532,24.99,23.41,158.8,1956,0.1238,0.1866,0.2416,0.186,0.275,0.08902)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

# prediction[0] = 0 thì ác tính
# prediction[0] = 1 thì lành tính

if (prediction[0] == 0) :
    print("kết quả: ung thư vú thuộc loại ác tính")
else:
    print("kết quả: ung thư vú thuộc loại lành tính")