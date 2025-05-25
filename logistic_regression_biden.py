import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Загрузка данных
df = pd.read_csv("trump_biden_balanced.csv")

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
predict_trump_quote("We will make America great again!")
predict_trump_quote("Fuck Hillary")
predict_trump_quote("Putin said I'm a genius!!!")

predict_trump_quote("We lead not by the example of our power, but by the power of our example.")
predict_trump_quote("Folks, this is not who we are as Americans.")
predict_trump_quote("We’re in a battle for the soul of this nation.")



import numpy as np

# Получаем коэффициенты модели (shape: [1, n_features] для бинарной классификации)
coef = model.coef_[0]

# Имена признаков из TF-IDF
feature_names = vectorizer.get_feature_names_out()

# Индексы сортировки по абсолютному значению коэффициентов (убывание)
top_n = 20
top_indices = np.argsort(np.abs(coef))[-top_n:][::-1]

print(f"Топ {top_n} важных признаков:")

for i in top_indices:
    print(f"{feature_names[i]}: coef = {coef[i]:.4f}")


import shap
import numpy as np

# Берём небольшой сэмпл теста для удобства
X_test_sample = X_test_vec[:100].toarray()

# Создаём explainer для LogisticRegression (линейной модели)
explainer = shap.LinearExplainer(model, X_train_vec, feature_perturbation="interventional")

# Вычисляем SHAP значения
shap_values = explainer.shap_values(X_test_sample)

# Визуализация: важность признаков в среднем по сэмплу
shap.summary_plot(shap_values, X_test_sample, feature_names=vectorizer.get_feature_names_out())

# Для визуализации конкретного предсказания (например, первого)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], feature_names=vectorizer.get_feature_names_out())
