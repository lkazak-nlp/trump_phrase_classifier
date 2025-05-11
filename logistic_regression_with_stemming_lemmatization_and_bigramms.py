import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet

# Загрузка ресурсов NLTK
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Инициализация инструментов
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Преобразование POS-тегов для лемматизатора
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Предобработка текста: токенизация, лемматизация, стемминг
def preprocess(text):
    tokens = word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    normalized = []
    for word, tag in tagged:
        lemma = lemmatizer.lemmatize(word.lower(), get_wordnet_pos(tag))
        # stemmed = stemmer.stem(lemma)
        # normalized.append(stemmed)
        normalized.append(lemma)
    return ' '.join(normalized)

# Загрузка данных
df = pd.read_csv("result.csv")

# Применение обработки
df['joined'] = df['text'].apply(preprocess)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df['joined'], df['label'], test_size=0.2, random_state=42)

# TF-IDF векторизация
# vectorizer = TfidfVectorizer(stop_words='english')
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Модель логистической регрессии
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

# Оценка модели
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Предсказание фраз
def predict_trump_quote(phrase):
    joined = preprocess(phrase)
    vec = vectorizer.transform([joined])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]
    print(f"Вердикт: {'Трамп' if prediction == 1 else 'Не Трамп'} (уверенность: {proba:.2f})")

# Пример
predict_trump_quote("We will make America great again.") # 0.82 < 0.90
predict_trump_quote("Fuck Hillary") # 0.59 < 0.71
predict_trump_quote("We have the most smart people.") # 0.69 < 0.77

predict_trump_quote("To be or not to be.") # 0.18 < 0.22
predict_trump_quote("He who has a why to live can bear almost any how.") # 0.29 > 0.26
predict_trump_quote("Russia has no borders, only interests.") # 0.42 > 0.32