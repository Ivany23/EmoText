import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
DetectorFactory.seed = 42

def carregar_lexico(caminho: str='data/lexicon/lexico_xichangana_set.txt') -> set:
    with open(caminho, 'r', encoding='utf-8') as f:
        lexico = set((line.strip().lower() for line in f if line.strip()))
    print(f'[LangDetector] Léxico carregado: {len(lexico)} termos Xichangana')
    return lexico

def calcular_score_xi(texto: str, lexico: set) -> float:
    tokens = re.findall("\\b[\\w']+\\b", texto.lower())
    if not tokens:
        return 0.0
    matches = sum((1 for t in tokens if t in lexico))
    return matches / len(tokens)

def detectar_lingua_base(texto: str) -> str:
    try:
        return detect(texto)
    except LangDetectException:
        return 'unknown'

def classificar_lingua(texto: str, lexico: set) -> str:
    score = calcular_score_xi(texto, lexico)
    lang_base = detectar_lingua_base(texto)
    if score >= 0.8:
        return 'xi'
    elif score >= 0.3:
        return 'misto'
    elif score < 0.05 and lang_base == 'pt':
        return 'pt'
    else:
        return 'misto'

def validar_modulo(df_sample, col_texto: str, col_label_humano: str):
    from sklearn.metrics import classification_report
    import pandas as pd
    LEXICO = carregar_lexico()
    df_sample = df_sample.copy()
    df_sample['pred_lang'] = df_sample[col_texto].apply(lambda t: classificar_lingua(t, LEXICO))
    print('\n=== VALIDAÇÃO DO MÓDULO DE DETECÇÃO LINGUÍSTICA ===')
    print(classification_report(df_sample[col_label_humano], df_sample['pred_lang'], labels=['pt', 'misto', 'xi'], target_names=['Português', 'Código-Misto', 'Xichangana']))
    from sklearn.metrics import f1_score
    f1_por_classe = f1_score(df_sample[col_label_humano], df_sample['pred_lang'], labels=['pt', 'misto', 'xi'], average=None)
    nomes = ['Português', 'Código-Misto', 'Xichangana']
    for nome, f1 in zip(nomes, f1_por_classe):
        status = 'OK' if f1 >= 0.8 else 'ABAIXO DO LIMIAR'
        print(f'  F1 [{nome}]: {f1:.4f} → {status}')
if __name__ == '__main__':
    LEXICO = carregar_lexico()
    exemplos = [('Ndza tenda swinene pela vossa ajuda', 'misto'), ('O governo anunciou novas medidas hoje', 'pt'), ('Xikwembu wa pfuna hinkwavu va munhu', 'xi'), ("Muito obrigado n'wina hinkwenu", 'misto'), ('Reunião aconteceu em Maputo ontem', 'pt')]
    print('\n=== TESTE DO MÓDULO DE DETECÇÃO ===')
    print(f'{'Texto':<45} {'Esperado':<10} {'Previsto':<10} {'Score Xi':<10}')
    print('-' * 80)
    for texto, esperado in exemplos:
        previsto = classificar_lingua(texto, LEXICO)
        score = calcular_score_xi(texto, LEXICO)
        status = 'OK' if previsto == esperado else 'ERRO'
        print(f'{texto:<45} {esperado:<10} {previsto:<10} {score:<10.3f} {status}')