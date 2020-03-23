import json
import requests
import time
import numpy as np
from keras import models, layers
from sklearn.model_selection import train_test_split


BASE_URL = "http://www.nlotto.co.kr/common.do?method=getLottoNumber&drwNo="
DATA_PATH = "./data/data.json"

with open(DATA_PATH) as json_file:
    data = json.load(json_file)

drwNo = 1
while True:
    if not str(drwNo) in data:
        res = requests.get(BASE_URL + str(drwNo)).json()
        if res["returnValue"] == "fail":
            print("Next drwNo: " + str(drwNo))
            break

        print("Save drwNo=" + str(drwNo) + " data..")
        data[str(drwNo)] = res
        with open(DATA_PATH, 'w') as json_file:
            json.dump(data, json_file)
    
    drwNo = drwNo + 1
    time.sleep(1)

print("Train with " + str(len(data)) + " data..")

x = []
y = []
for i in range(1, drwNo):
    x.append(i)
    
    refined = []
    for drwt in range(1, 7):
        refined.append(data[str(i)]["drwtNo" + str(drwt)])
    print(refined)
    y.append(refined)
x = np.array(x)
y = np.array(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

model = models.Sequential()
model.add(layers.Dense(6, input_shape=(1,)))
model.compile(optimizer='SGD', loss='mse', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=30, verbose=1, validation_split=0.1)
[loss, accuracy] = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy: " + str(accuracy*100) + "%")

result = model.predict([drwNo])
print("=== " + str(drwNo) + "st Numbers Prediction ===")
print(result)
