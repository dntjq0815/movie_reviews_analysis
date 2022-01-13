# movie_reviews_analysis
Analyzing movie reviews

## train.csv
Column | Description
---| ---|
id | id value
document | review contents
label | reviews type (pos:1, neg:0)

## test.csv
Column | Description
---| ---|
id | id value
document | review contents

### import data
```python
data = pd.read_csv("dataset/train.csv")
```
<br/>

### Checking missing columns
```python
def check_missing_col(dataframe):
    missing_col = []
    for col in dataframe.columns:
        missing_values = sum(dataframe[col].isna())
        is_missing = True if missing_values >= 1 else False
        if is_missing:
            print(f'결측치가 있는 컬럼은: {col} 입니다')
            print(f'해당 컬럼에 총 {missing_values} 개의 결측치가 존재합니다.')
            missing_col.append([col, dataframe[col].dtype])
    if missing_col == []:
        print('결측치가 존재하지 않습니다')
    return missing_col

missing_col = check_missing_col(data)
```
```python
결측치가 존재하지 않습니다.
```
Find out there's no missing values in the data. <br/><br/>

## Seperating documents & labels
```python
X = data.document
y = data.label
```
Put documents on the variable 'X', labels on the varialble 'y' for building a model. <br/><br/>

## import CountVectorizer for word embedding
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
```

## training
```python
vectorizer.fit(X)       # X = data.document
X = vectorizer.transform(X)
```


## import LogisticRegression & training
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)
```

<br/>


## Run model
```python
X_pred = vectorizer.transform(["너무 재밌어요!"])
y_pred = model.predict(X_pred)      # pos:1, neg:0
```
```python
[1]
```
