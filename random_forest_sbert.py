import pandas as pd
import re
import nltk
import spacy
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from pandarallel import pandarallel

# Загрузка ресурсов
nltk.download('punkt_tab')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

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
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in stop_words
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
    short_word_ratio = sum(len(w) <= 4 for w in words) / (word_count + 1e-6)
    negative_words = set(["fake", "bad", "loser", "sad", "disaster", "enemy"])
    negative_ratio = sum(w in negative_words for w in words) / (word_count + 1e-6)
    first_person_pronouns = set(['i', 'me', 'we', 'us', 'my', 'mine', 'our', 'ours'])
    first_person_ratio = sum(w in first_person_pronouns for w in words) / (word_count + 1e-6)
    avg_sent_len = word_count / (sent_count + 1e-6)

    return [
        trump_word_count / (word_count + 1e-6),
        trump_pronoun_count / (word_count + 1e-6),
        max_uppercase_seq_len,
        exclam_count,
        short_word_ratio,
        negative_ratio,
        first_person_ratio,
        avg_sent_len
    ]

# Предобработка
tqdm_kwargs = {'desc': 'Preprocessing', 'leave': False}
pandarallel.initialize()
df['processed'] = df['text'].parallel_apply(preprocess_text)
df['sbert'] = sbert_model.encode(df['processed'].tolist(), convert_to_numpy=True).tolist()
manual_features = df['text'].apply(extract_trump_features).tolist()
manual_features_df = pd.DataFrame(manual_features, columns=[
    'trump_word_ratio',
    'trump_pronoun_ratio',
    'max_uppercase_seq_len',
    'exclam_count',
    'short_word_ratio',
    'negative_ratio',
    'first_person_ratio',
    'avg_sent_len'
])

# Объединение в один DataFrame
df_full = df[['processed', 'label', 'sbert']].copy()
df_full[manual_features_df.columns] = manual_features_df

# Train/Test Split
train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42)

X_train_text = train_df['processed']
X_test_text = test_df['processed']
y_train = train_df['label']
y_test = test_df['label']
X_train_feat = train_df[manual_features_df.columns]
X_test_feat = test_df[manual_features_df.columns]
X_train_sbert = np.array(train_df['sbert'].tolist())
X_test_sbert = np.array(test_df['sbert'].tolist())

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train_text)
X_test_vec = vectorizer.transform(X_test_text)

# Комбинируем признаки
X_train_combined = hstack([X_train_vec, csr_matrix(X_train_feat.values), csr_matrix(X_train_sbert)])
X_test_combined = hstack([X_test_vec, csr_matrix(X_test_feat.values), csr_matrix(X_test_sbert)])

# Обучение модели
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
model = grid_search.best_estimator_

# Оценка
y_pred = model.predict(X_test_combined)
print(classification_report(y_test, y_pred))

# Предсказание новой фразы
def predict_trump_quote_combined(phrase):
    processed = preprocess_text(phrase)
    tfidf_vec = vectorizer.transform([processed])
    manual_vec = extract_trump_features(phrase)
    sbert_vec = sbert_model.encode([processed], convert_to_numpy=True)

    combined = hstack([
        tfidf_vec,
        csr_matrix([manual_vec]),
        csr_matrix(sbert_vec)
    ])

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
