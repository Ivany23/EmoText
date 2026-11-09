import json
import os
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline, make_union

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DADOS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAIZ, 'dados', 'afrisenti')
DESTINO = os.path.join(RAIZ, 'modelos', 'polaridade_xi_tfidf')
SENTIMENTOS = ['negative', 'neutral', 'positive']


def limpar_texto(texto):
    if not isinstance(texto, str):
        return ''
    texto = re.sub(r'http\S+|www\.\S+', '', texto)
    texto = re.sub(r'@\w+', '', texto)
    texto = re.sub(r'<[^>]+>', '', texto)
    texto = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', texto)
    return re.sub(r'\s+', ' ', texto).strip()


def carregar(split):
    df = pd.read_csv(os.path.join(PASTA_DADOS, f'afrisenti_tso_{split}.tsv'), sep='\t', dtype=str)
    df['texto'] = df['tweet'].apply(limpar_texto)
    df['y'] = df['label'].map({s: i for i, s in enumerate(SENTIMENTOS)})
    return df


treino, teste = carregar('train'), carregar('test')
vectorizador = make_union(TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2, sublinear_tf=True))
modelo = make_pipeline(vectorizador, LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42))
modelo.fit(treino['texto'], treino['y'])
mf1 = f1_score(teste['y'], modelo.predict(teste['texto']), average='macro')
print(f'Macro-F1 no teste (xitsonga): {100 * mf1:.1f}%')

os.makedirs(DESTINO, exist_ok=True)
joblib.dump(modelo, os.path.join(DESTINO, 'modelo.joblib'))
with open(os.path.join(DESTINO, 'emotext.json'), 'w', encoding='utf-8') as f:
    json.dump({'tarefa': 'polaridade', 'lingua': 'xitsonga', 'modelo': 'TF-IDF (palavras 1-2, caracteres 2-5) + Regressão Logística',
               'classes': SENTIMENTOS, 'macro_f1_teste': round(float(mf1), 4)}, f, ensure_ascii=False, indent=2)
print('Guardado em', DESTINO)
