import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# set system encoding to utf-8
sys.stdout.reconfigure(encoding='utf-8')

dataset = pd.read_csv('cancer.csv')

x = dataset.drop("diagnosis(1=m, 0=b)", axis=1)
y = dataset["diagnosis(1=m, 0=b)"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation="sigmoid"))
model.add(tf.keras.layers.Dense(256, activation="sigmoid"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)


model.evaluate(x_test, y_test) #testing the test data