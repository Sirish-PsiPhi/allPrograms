import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

diabetes = pd.read_csv("./datasets/P6.csv")
number = LabelEncoder()


def encode(f):
    diabetes[f] = number.fit_transform(diabetes[f])


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction',
            'Age']
target = ['Outcome']
for feature in features:
    encode(feature)
encode(target[0])
features_train, features_test, target_train, target_test = train_test_split(diabetes[features], diabetes[target],
                                                                            test_size=0.3, random_state=54)

model = GaussianNB()
model.fit(features_train, target_train)

prediction = model.predict(features_test)
accuracy = accuracy_score(target_test, prediction)
print(f"Accuracy of the prediction is {accuracy}")
