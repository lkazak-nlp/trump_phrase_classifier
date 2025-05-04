from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

# Предобработка
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return X.apply(lambda s: len(s.split())).values.reshape(-1, 1)

# Загрузка
df = pd.read_csv("result.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Пайплайн
pipeline = Pipeline([
    ("features", FeatureUnion([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            tokenizer=word_tokenize,
            token_pattern=None,
            ngram_range=(1,2),
            max_df=0.9,
            min_df=2,
            max_features=5000
        )),
        ("length", TextLengthExtractor())
    ])),
    ("clf", LogisticRegression(solver='saga', max_iter=1000, class_weight='balanced'))
])

# Сетка гиперпараметров
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ['l1', 'l2'],
}

search = GridSearchCV(
    pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
)

# Обучение
search.fit(X_train, y_train)
print("Лучшие параметры:", search.best_params_)

# Оценка
y_pred = search.predict(X_test)
print(classification_report(y_test, y_pred))

# Функция предсказания
def predict_trump_quote(phrase):
    cleaned = clean_text(phrase)
    vec = search.best_estimator_.named_steps['features'].transform(pd.Series([cleaned]))
    pred = search.best_estimator_.named_steps['clf'].predict(vec)[0]
    proba = search.best_estimator_.named_steps['clf'].predict_proba(vec)[0][1]
    print(f"«{phrase}» → {'Трамп' if pred==1 else 'Не Трамп'} (уверенность {proba:.2f})")
    return pred, proba

# Примеры
predict_trump_quote("We will make America great again.")
predict_trump_quote("Fuck Hillary")
predict_trump_quote("Fuck Kamala")
predict_trump_quote("We have the most smart people.")

predict_trump_quote("To be or not to be.")
predict_trump_quote("He who has a why to live can bear almost any how.")
predict_trump_quote("Russia has no borders, only interests.")
