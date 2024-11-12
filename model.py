import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

label_encoder = LabelEncoder()

# import training data
trainingdata = pd.read_json('Training.json')
trainingdata['SentimentText'] = trainingdata['SentimentText'].fillna('missing')

# import test data
test1 = pd.read_json('test1_public.json')
test1['SentimentText'] = test1['SentimentText'].fillna('missing')
test1Ids = test1['TonyID']

test2 = pd.read_json('test2_public.json')
test2['SentimentText'] = test1['SentimentText'].fillna('missing')
test2Ids = test2['TonyID']

y = label_encoder.fit_transform(trainingdata['Sentiment1'])
X = trainingdata['SentimentText']
testid = trainingdata['TonyID']

vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,4))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state=42)

naivebayesmodel = make_pipeline(vectorizer, MultinomialNB())

# Train the model
naivebayesmodel.fit(X_train, y_train)

# Predict the labels for the test set
nbpred = naivebayesmodel.predict(X_test)

# make predictions
test1preds = naivebayesmodel.predict(test1['SentimentText'])
output = []

for x in range (len(test1preds)):
    if test1preds[x] == 0:
        sentiment = "+"
    else:
        sentiment = "-"
    output.append(str(test1Ids[x]) + "\t" + sentiment + "\n")

file = open('chrisduym_file1_entry1.txt', "w")
for x in output:
    file.write(x)
file.close()

test2preds = naivebayesmodel.predict(test2['SentimentText'])
output = []

for x in range (len(test2preds)):
    if test2preds[x] == 0:
        sentiment = "+"
    else:
        sentiment = "-"
    output.append(str(test2Ids[x]) + "\t" + sentiment + "\n")

file = open('chrisduym_file2_entry1.txt', "w")
for x in output:
    file.write(x)
file.close()

# Calculate the accuracy
nbaccuracy = accuracy_score(y_test, nbpred)
print(f'Accuracy: {nbaccuracy}')

svmmodel = make_pipeline(vectorizer, LinearSVC())
svmmodel.fit(X_train, y_train)
svmpred = svmmodel.predict(X_test)
svmaccuracy = accuracy_score(y_test, svmpred)
print(f'Accuracy: {svmaccuracy}')

# make predictions
test1preds = svmmodel.predict(test1['SentimentText'])
output = []

for x in range (len(test1preds)):
    if test1preds[x] == 0:
        sentiment = "+"
    else:
        sentiment = "-"
    output.append(str(test1Ids[x]) + "\t" + sentiment + "\n")

file = open('chrisduym_file1_entry2.txt', "w")
for x in output:
    file.write(x)
file.close()

test2preds = svmmodel.predict(test2['SentimentText'])
output = []

for x in range (len(test2preds)):
    if test2preds[x] == 0:
        sentiment = "+"
    else:
        sentiment = "-"
    output.append(str(test2Ids[x]) + "\t" + sentiment + "\n")

file = open('chrisduym_file2_entry2.txt', "w")
for x in output:
    file.write(x)
file.close()


