import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


trainingdata = pd.read_json('Training.json')
trainingdata['SentimentText'] = trainingdata['SentimentText'].fillna('missing')
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(trainingdata['Sentiment1'])
X = trainingdata['SentimentText']
testid = trainingdata['TonyID']

vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state=42)

model = make_pipeline(vectorizer, MultinomialNB())
# Train the model
model.fit(X_train, y_train)
 
# Predict the labels for the test set
y_pred = model.predict(X_test)
 
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


