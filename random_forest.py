import pandas as pd
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# Загрузка ресурсов
nltk.download('punkt_tab')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')
stop_words = set(stopwords.words('english'))

# Загрузка данных
df = pd.read_csv("result.csv")

# Трамповская лексика и местоимения
trump_words = set([
    "tremendous", "fake", "china", "great", "believe", "wall", "america", "again",
    "media", "nobody", "genius", "winning", "hillary", "jobs", "big", "bad",
    "deal", "kamala", "disaster", "strong", "success", "loser", "enemy", "sad", "historic", "obama", "she"
])
trump_pronouns = set(["i", "me", "we", "nobody"])

# Лемматизация и предобработка
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    return ' '.join(tokens)

# Ручные признаки
def extract_trump_features(text):
    tokens = word_tokenize(text.lower())
    words = [w for w in tokens if w.isalpha()]
    word_count = len(words)
    sent_count = len(sent_tokenize(text))

    trump_word_count = sum(1 for w in words if w in trump_words)
    trump_pronoun_count = sum(1 for w in words if w in trump_pronouns)

    return [
        len(text),                   # длина текста (в символах)
        sent_count,                  # количество предложений
        trump_word_count / (word_count + 1e-6),  # доля "трамповских" слов
        trump_pronoun_count / (word_count + 1e-6) # доля местоимений
    ]

# Применяем предобработку и извлекаем признаки
df['processed'] = df['text'].apply(preprocess_text)
manual_features = df['text'].apply(extract_trump_features).tolist()
manual_features_df = pd.DataFrame(manual_features, columns=[
    'char_length', 'sent_count', 'trump_word_ratio', 'trump_pronoun_ratio'
])

# Объединяем в один DataFrame
df_full = df[['processed', 'label']].copy()
df_full[['char_length', 'sent_count', 'trump_word_ratio', 'trump_pronoun_ratio']] = manual_features_df

# Делим на train/test
train_df, test_df = train_test_split(df_full, test_size=0.2, random_state=42)

X_train_text = train_df['processed']
X_test_text = test_df['processed']
y_train = train_df['label']
y_test = test_df['label']

X_train_feat = train_df[['char_length', 'sent_count', 'trump_word_ratio', 'trump_pronoun_ratio']]
X_test_feat = test_df[['char_length', 'sent_count', 'trump_word_ratio', 'trump_pronoun_ratio']]

# TF-IDF векторизация
vectorizer = TfidfVectorizer()
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
predict_trump_quote("We will make America great again.")
predict_trump_quote("Fuck Hillary")
predict_trump_quote("She is a bitch")
predict_trump_quote("Fuck Obama")
predict_trump_quote("We have the most smart people.")

predict_trump_quote("To be or not to be.")
predict_trump_quote("He who has a why to live can bear almost any how.")
predict_trump_quote("Russia has no borders, only interests.")
