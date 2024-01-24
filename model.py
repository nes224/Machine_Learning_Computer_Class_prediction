import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("Computer_class _training.csv")

print(df.head())

X = df[["Computer_length"]]
y = df["Class"]

X_train,_, y_train,_ = train_test_split(X,y) 

# Feature scaling
sc = StandardScaler() # Sc will be an object of this class 

# To develop our machine learning model so first I will instantiate the model.
# Make one object classifier
classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

pickle.dump(classifier, open("model.pkl", "wb"))