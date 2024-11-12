import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

label_encoder = LabelEncoder()

trainingdata = pd.read_json('Training.json')
trainingdata['SentimentText'] = trainingdata['SentimentText'].fillna('missing')

y = label_encoder.fit_transform(trainingdata['Sentiment1'])
X = trainingdata['SentimentText']
testid = trainingdata['TonyID']

vectorizer = TfidfVectorizer(lowercase=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state=42)

naivebayesmodel = make_pipeline(vectorizer, MultinomialNB())

# Train the model
naivebayesmodel.fit(X_train, y_train)

# Predict the labels for the test set
nbpred = naivebayesmodel.predict(X_test)

 
# Calculate the accuracy
nbaccuracy = accuracy_score(y_test, nbpred)
print(f'Accuracy: {nbaccuracy}')


svmmodel = make_pipeline(vectorizer, svm.SVC(kernel='linear'))
svmmodel.fit(X_train, y_train)
svmpred = svmmodel.predict(X_test)
svmaccuracy = accuracy_score(y_test, svmpred)
print(f'Accuracy: {svmaccuracy}')


