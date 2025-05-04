import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Загрузка данных
df = pd.read_csv("result.csv")

# Токенизация
df['tokens'] = df['text'].apply(word_tokenize)
df['joined'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))  # для TF-IDF нужен текст

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(df['joined'], df['label'], test_size=0.2, random_state=42)

# Векторизация
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Оценка
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))


def predict_trump_quote(phrase):
    # Токенизация и подготовка
    tokens = word_tokenize(phrase)
    joined = ' '.join(tokens)
    vec = vectorizer.transform([joined])

    # Предсказание
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]  # вероятность "фразы Трампа"

    print(f"Вердикт: {'Трамп' if prediction == 1 else 'Не Трамп'} (уверенность: {proba:.2f})")


# Пример
predict_trump_quote("We will make America great again.")
predict_trump_quote("Fuck Hillary")
predict_trump_quote("Fuck Kamala")
predict_trump_quote("We have the most smart people.")

predict_trump_quote("To be or not to be.")
predict_trump_quote("He who has a why to live can bear almost any how.")
predict_trump_quote("Russia has no borders, only interests.")

