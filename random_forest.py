import pandas as pd
import re
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from pandarallel import pandarallel
from scipy.sparse import vstack
import os


# Загрузка ресурсов
nltk.download('punkt_tab')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# Загрузка данных
df = pd.read_csv("trump_biden_balanced.csv")

# Трамповская лексика и местоимения
trump_words = set([
    "tremendous", "fake", "china", "great", "believe", "wall", "america", "again",
    "media", "nobody", "genius", "winning", "hillary", "jobs", "big", "bad",
    "deal", "kamala", "disaster", "strong", "success", "loser", "enemy", "sad", "historic", "obama", "she"
])
trump_pronouns = set(["i", "me", "we", "nobody"])

# Лемматизация и предобработка
def preprocess_text(text):
    # Удаляем ссылки
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # NLP-пайплайн
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and token.text.lower() not in stop_words
    ]
    return ' '.join(tokens)

# Ручные признаки
def extract_trump_features(text):
    tokens = word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha()]
    word_count = len(words)
    sent_count = len(sent_tokenize(text))

    trump_word_count = sum(1 for w in words if w in trump_words)
    trump_pronoun_count = sum(1 for w in words if w in trump_pronouns)

    uppercase_sequences = re.findall(r'[A-Z]{2,}', text)
    max_uppercase_seq_len = max((len(seq) for seq in uppercase_sequences), default=0)

    exclam_count = text.count('!')
    question_count = text.count('?')
    short_word_ratio = sum(len(w) <= 4 for w in words) / (word_count + 1e-6)
    repeats = sum(words[i] == words[i+1] for i in range(len(words)-1))
    negative_words = set(["fake", "bad", "loser", "sad", "disaster", "enemy"])
    negative_ratio = sum(w in negative_words for w in words) / (word_count + 1e-6)

    doc = nlp(text)
    adv_count = sum(1 for token in doc if token.pos_ == 'ADV')
    adv_ratio = adv_count / (word_count + 1e-6)

    first_person_pronouns = set(['i', 'me', 'we', 'us', 'my', 'mine', 'our', 'ours'])
    first_person_ratio = sum(w in first_person_pronouns for w in words) / (word_count + 1e-6)

    uppercase_words_count = sum(w.isupper() for w in tokens)
    uppercase_word_ratio = uppercase_words_count / (len(tokens) + 1e-6)

    avg_sent_len = word_count / (sent_count + 1e-6)

    return [
        trump_word_count / (word_count + 1e-6),
        trump_pronoun_count / (word_count + 1e-6),
        max_uppercase_seq_len,
        exclam_count,
        # question_count,
        short_word_ratio,
        # repeats,
        negative_ratio,
        # adv_ratio,
        first_person_ratio,
        # uppercase_word_ratio,
        avg_sent_len
    ]

# Применяем предобработку и извлекаем признаки
pandarallel.initialize()

# if os.path.exists("processed_data.csv"):
#     df_full = pd.read_csv("processed_data.csv")
# else:
pandarallel.initialize()
df['processed'] = df['text'].parallel_apply(preprocess_text)
manual_features = df['text'].apply(extract_trump_features).tolist()
manual_features_df = pd.DataFrame(manual_features, columns=[
    'trump_word_ratio', # important
    'trump_pronoun_ratio', # important
    'max_uppercase_seq_len', # important
    'exclam_count',  # important
    # 'question_count',
    'short_word_ratio', # important
    # 'repeats',
    'negative_ratio', # important
    # 'adv_ratio',
    'first_person_ratio', # important
    # 'uppercase_word_ratio',
    'avg_sent_len' # important
])

df_full = df[['processed', 'label']].copy()
df_full[['trump_word_ratio',
         'trump_pronoun_ratio',
         'max_uppercase_seq_len',
         'exclam_count',
         # 'question_count',
         'short_word_ratio',
         # 'repeats',
         'negative_ratio',
         # 'adv_ratio',
         'first_person_ratio',
         # 'uppercase_word_ratio',
         'avg_sent_len']] = manual_features_df
# df_full[['char_length', 'sent_count', 'trump_word_ratio', 'trump_pronoun_ratio', 'max_uppercase_seq_len']] = manual_features_df
df_full.to_csv("processed_data.csv", index=False)

# Делим на train/test
train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42)

X_train_text = train_df['processed']
X_test_text = test_df['processed']
y_train = train_df['label']
y_test = test_df['label']

X_train_feat = train_df[['trump_word_ratio',
                         'trump_pronoun_ratio',
                         'max_uppercase_seq_len',
                         'exclam_count',
                         # 'question_count',
                         'short_word_ratio',
                         # 'repeats',
                         'negative_ratio',
                         # 'adv_ratio',
                         'first_person_ratio',
                         # 'uppercase_word_ratio',
                         'avg_sent_len']]
X_test_feat = test_df[['trump_word_ratio',
                       'trump_pronoun_ratio',
                       'max_uppercase_seq_len',
                       'exclam_count',
                       # 'question_count',
                       'short_word_ratio',
                       # 'repeats',
                       'negative_ratio',
                       # 'adv_ratio',
                       'first_person_ratio',
                       # 'uppercase_word_ratio',
                       'avg_sent_len']]
# X_train_feat = train_df[['char_length', 'sent_count', 'trump_word_ratio', 'trump_pronoun_ratio', 'max_uppercase_seq_len']]
# X_test_feat = test_df[['char_length', 'sent_count', 'trump_word_ratio', 'trump_pronoun_ratio', 'max_uppercase_seq_len']]

# TF-IDF векторизация
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# Объединение TF-IDF и ручных признаков
X_train_combined = hstack([X_train_vec, csr_matrix(X_train_feat.values)])
X_test_combined = hstack([X_test_vec, csr_matrix(X_test_feat.values)])

# RandomForest + GridSearch
param_grid = {
    'n_estimators': [200],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_combined, y_train)

print("Best parameters found: ", grid_search.best_params_)
model = grid_search.best_estimator_

# Оценка
y_pred = model.predict(X_test_combined)
print(classification_report(y_test, y_pred))

# Предсказание новой фразы
def predict_trump_quote(phrase):
    processed = preprocess_text(phrase)
    tfidf = vectorizer.transform([processed])
    manual = extract_trump_features(phrase)
    combined = hstack([tfidf, csr_matrix([manual])])

    prediction = model.predict(combined)[0]
    proba = model.predict_proba(combined)[0][1]
    print(f"Вердикт: {'Трамп' if prediction == 1 else 'Не Трамп'} (уверенность: {proba:.2f})")


# 12) Примеры использования
predict_trump_quote("We will make America great again!")
predict_trump_quote("Fuck Hillary")
predict_trump_quote("Putin said I'm a genius!!!")

predict_trump_quote("We lead not by the example of our power, but by the power of our example.")
predict_trump_quote("Folks, this is not who we are as Americans.")
predict_trump_quote("We’re in a battle for the soul of this nation.")

# Отфильтруем тестовые данные с label=1
test_trump_mask = (y_test == 1)
X_test_trump_combined = X_test_combined[test_trump_mask]
y_test_trump = y_test[test_trump_mask]

# Предсказания модели на этих данных
y_pred_trump = model.predict(X_test_trump_combined)

# Важность признаков — глобальная, берем из модели
importances = model.feature_importances_

feature_names = vectorizer.get_feature_names_out().tolist() + list(X_train_feat.columns)

# Берём топ-N важных признаков
N = 20
indices = np.argsort(importances)[-N:][::-1]

plt.figure(figsize=(10, 6))
plt.title(f"Топ {N} важных признаков для фраз Трампа (label=1)")
plt.barh(range(N), importances[indices], align="center")
plt.yticks(range(N), [feature_names[i] for i in indices])
plt.gca().invert_yaxis()
plt.xlabel("Важность признака")
plt.tight_layout()
plt.show()