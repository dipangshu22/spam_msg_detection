import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

data = data[['v1','v2']]
data.columns = ['label','message']

# convert labels
data['label'] = data['label'].map({'ham':0,'spam':1})

# text vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(data["message"])

y = data["label"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# train model
model = LogisticRegression()
model.fit(X_train,y_train)

# save model
pickle.dump(model, open("spam_model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Model saved successfully")