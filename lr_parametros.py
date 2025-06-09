import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1) Carrega e pré-processa como você já faz
def limpar_prefixo_agencia(texto):
    return re.sub(r"^(.*?\(Reuters\)[\s\-–]*)", "", texto)

df_fake = pd.read_csv("Data/Fake.csv")
df_true = pd.read_csv("Data/True.csv")
df_fake['label'] = 0
df_true['label'] = 1
for df in (df_fake, df_true):
    df.drop(columns=["title","date","subject"], errors='ignore', inplace=True)
df_true['text'] = df_true['text'].apply(limpar_prefixo_agencia)
df = pd.concat([df_fake, df_true], ignore_index=True).drop_duplicates('text').sample(frac=1, random_state=42)

# 2) Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
)

# 3) Monta um pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf',  LogisticRegression(max_iter=1000, random_state=42)),
])

# 4) Define a grade de parâmetros a explorar
param_grid = {
    # parâmetros do TfidfVectorizer
    'tfidf__stop_words':       ['english', None],
    'tfidf__max_df':           [0.75, 0.85, 1.0],
    'tfidf__min_df':           [1, 2, 5],
    'tfidf__ngram_range':      [(1,1), (1,2)],
    'tfidf__max_features':     [5000, 10000, 20000],
    # parâmetros do LogisticRegression
    'clf__penalty':            ['l2'],              # l1 só funciona com solver liblinear
    'clf__C':                  [0.01, 0.1, 1, 10, 100],
    'clf__solver':             ['liblinear','saga'],
    'clf__class_weight':       [None, 'balanced'],
}

# 5) GridSearchCV com 5-fold
grid = GridSearchCV(
    pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1'
)

# 6) Executa a busca
grid.fit(X_train, y_train)

print("Melhores parâmetros:\n", grid.best_params_)
print(f"Melhor F1 (validação): {grid.best_score_:.4f}")

# 7) Avalia no conjunto de teste
y_pred = grid.predict(X_test)
print(classification_report(y_test, y_pred, digits=4))