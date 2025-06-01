import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack

# Загрузка данных
df = pd.read_csv("trump_biden_balanced.csv")

# Делим
X_train_text, X_test_text, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# === TF-IDF ===
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)

# === SBERT ===
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
X_train_sbert = sbert_model.encode(X_train_text.tolist(), show_progress_bar=True)
X_test_sbert = sbert_model.encode(X_test_text.tolist(), show_progress_bar=True)

# Превращаем SBERT в sparse формат, чтобы объединить
from scipy.sparse import csr_matrix
X_train_sbert_sparse = csr_matrix(X_train_sbert)
X_test_sbert_sparse = csr_matrix(X_test_sbert)

# === Объединение ===
X_train_combined = hstack([X_train_tfidf, X_train_sbert_sparse])
X_test_combined = hstack([X_test_tfidf, X_test_sbert_sparse])

# === Обучение модели ===
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_combined, y_train)

# === Предсказание ===
y_pred = model.predict(X_test_combined)
print(classification_report(y_test, y_pred))


def predict_trump_quote_combined(phrase):
    tfidf_vec = tfidf.transform([phrase])
    sbert_vec = sbert_model.encode([phrase])
    sbert_sparse = csr_matrix(sbert_vec)
    combined = hstack([tfidf_vec, sbert_sparse])

    prediction = model.predict(combined)[0]
    proba = model.predict_proba(combined)[0][1]

    print(f"Вердикт: {'Трамп' if prediction == 1 else 'Не Трамп'} (уверенность: {proba:.2f})")


# Примеры
predict_trump_quote_combined("We will make USA great again!")
predict_trump_quote_combined("We will make America great again!")
predict_trump_quote_combined("We will make yankees cool!")

predict_trump_quote_combined("Fuck Hillary")
predict_trump_quote_combined("Hillary is a good girl")

predict_trump_quote_combined("Hillary is a bad girl")
predict_trump_quote_combined("Hillary is a bad president")

predict_trump_quote_combined("Putin said I'm a genius!!!")

predict_trump_quote_combined("We lead not by the example of our power, but by the power of our example.")
predict_trump_quote_combined("Folks, this is not who we are as Americans.")
predict_trump_quote_combined("We’re in a battle for the soul of this nation.")
