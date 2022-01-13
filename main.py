import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('dataset/train.csv')

def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 칼럼은: {col} 입니다.')
            print(f'해당 칼럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다.')
    return missing_col

missing_col = check_missing_col(data)

X = data.document
y = data.label

vectorizer = CountVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

model = LogisticRegression()
model.fit(X, y)

X_pred = vectorizer.transform(["영화 그렇게 재밌지는 않음"])
y_pred = model.predict(X_pred)
print(y_pred)

test = pd.read_csv("dataset/test.csv")

test_X = test.document
test_X_vect = vectorizer.transform(test_X)

pred = model.predict(test_X_vect)
print(pred)

submission = pd.read_csv("dataset/sample_submission.csv")

submission["label"] = pred
submission.to_csv("dataset/submission.csv", index=False)