import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# 1) Загрузка модели токенизации
nltk.download('punkt_tab')

# 2) Очистка текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# 3) Преобразователь чистого текста
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return X.apply(clean_text)

# 4) Экстрактор длины текста (число слов)
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        return X.apply(lambda s: len(s.split())).values.reshape(-1, 1)

# 5) Загружаем данные из result.csv
df = pd.read_csv("result.csv")

# 6) Разбиваем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 7) Строим пайплайн
pipeline = Pipeline([
    ("features", FeatureUnion([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            tokenizer=word_tokenize,
            token_pattern=None,
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=2,
            max_features=3000
        )),
        ("length", TextLengthExtractor())
    ])),
    ("classifier", RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

# 8) Сетка гиперпараметров
param_grid = {
    "classifier__n_estimators": [100, 300],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_leaf": [1, 3, 5],
    "classifier__max_features": ["sqrt", 0.3]
}

search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

# 9) Обучаем
search.fit(X_train, y_train)
print("Лучшие параметры:", search.best_params_)

# 10) Оценка
y_pred = search.predict(X_test)
print("\nОтчёт на тестовой выборке:")
print(classification_report(y_test, y_pred))

# 11) Функция для проверки новых фраз
def predict_trump_quote(phrase):
    clean = clean_text(phrase)
    tokens = word_tokenize(clean)
    joined = ' '.join(tokens)
    vec = search.best_estimator_.named_steps['features'].transform(pd.Series([joined]))
    pred = search.best_estimator_.named_steps['classifier'].predict(vec)[0]
    proba = search.best_estimator_.named_steps['classifier'].predict_proba(vec)[0][1]
    label = 'Трамп' if pred == 1 else 'Не Трамп'
    print(f"Фраза: «{phrase}»\nВердикт: {label} (уверенность: {proba:.2f})")
    return pred, proba

# 12) Примеры использования
predict_trump_quote("We will make America great again.")
predict_trump_quote("Fuck Hillary")
predict_trump_quote("Fuck Kamala")
predict_trump_quote("We have the most smart people.")

predict_trump_quote("To be or not to be.")
predict_trump_quote("He who has a why to live can bear almost any how.")
predict_trump_quote("Russia has no borders, only interests.")
