import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
CLASSES = ['positive', 'negative', 'neutral']
NOMES_PT = ['Positivo', 'Negativo', 'Neutro']
SEED = 42

def calcular_metricas(y_true, y_pred, nome_modelo: str='Modelo') -> dict:
    f1_macro = f1_score(y_true, y_pred, average='macro', labels=CLASSES)
    accuracy = accuracy_score(y_true, y_pred)
    f1_pos = f1_score(y_true, y_pred, labels=['positive'], average='macro')
    f1_neg = f1_score(y_true, y_pred, labels=['negative'], average='macro')
    f1_neu = f1_score(y_true, y_pred, labels=['neutral'], average='macro')
    resultados = {'modelo': nome_modelo, 'f1_macro': round(f1_macro, 4), 'accuracy': round(accuracy, 4), 'f1_pos': round(f1_pos, 4), 'f1_neg': round(f1_neg, 4), 'f1_neu': round(f1_neu, 4)}
    print(f'\n{'=' * 50}')
    print(f'  RESULTADOS — {nome_modelo}')
    print(f'{'=' * 50}')
    print(f'  F1-Macro  (métrica principal): {f1_macro:.4f}')
    print(f'  Accuracy                     : {accuracy:.4f}')
    print(f'  F1 Positivo                  : {f1_pos:.4f}')
    print(f'  F1 Negativo                  : {f1_neg:.4f}')
    print(f'  F1 Neutro                    : {f1_neu:.4f}')
    print(f'\n{classification_report(y_true, y_pred, labels=CLASSES, target_names=NOMES_PT)}')
    return resultados

def tabela_comparativa(lista_resultados: list) -> pd.DataFrame:
    df = pd.DataFrame(lista_resultados)
    df = df.set_index('modelo')
    df.columns = ['F1-Macro', 'Accuracy', 'F1(pos)', 'F1(neg)', 'F1(neu)']
    print('\n=== TABELA COMPARATIVA DE MODELOS ===')
    print(df.to_string())
    melhor = df['F1-Macro'].idxmax()
    print(f'\n Melhor modelo por F1-Macro: {melhor} ({df.loc[melhor, 'F1-Macro']:.4f})')
    return df

def plotar_matriz_confusao(y_true, y_pred, nome_modelo: str, guardar_em: str='results/confusion_matrices/'):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=NOMES_PT, yticklabels=NOMES_PT, linewidths=0.5)
    plt.title(f'Matriz de Confusão — {nome_modelo}', fontsize=13, pad=12)
    plt.ylabel('Label Real', fontsize=11)
    plt.xlabel('Label Previsto', fontsize=11)
    plt.tight_layout()
    nome_ficheiro = f'{guardar_em}{nome_modelo.replace(' ', '_')}.png'
    import os
    os.makedirs(guardar_em, exist_ok=True)
    plt.savefig(nome_ficheiro, dpi=200)
    plt.show()
    print(f'Matriz guardada: {nome_ficheiro}')

def calcular_kappa(df_anotacoes: pd.DataFrame, col_a1: str='anotador_1', col_a2: str='anotador_2') -> dict:
    kappa = cohen_kappa_score(df_anotacoes[col_a1], df_anotacoes[col_a2], labels=CLASSES)
    concordancia = (df_anotacoes[col_a1] == df_anotacoes[col_a2]).mean()
    print('\n=== CONCORDÂNCIA INTER-ANOTADORES ===')
    print(f'  Kappa de Cohen          : {kappa:.4f}')
    print(f'  Concordância simples    : {concordancia:.2%}')
    print(f'  Comentários anotados    : {len(df_anotacoes)}')
    if kappa >= 0.7:
        print('  RESULTADO: KAPPA ACEITAVEL — corpus aprovado para treino')
    elif kappa >= 0.6:
        print('  RESULTADO: KAPPA MODERADO — rever casos de desacordo')
    else:
        print('  RESULTADO: KAPPA INSUFICIENTE — rever guia de anotacao')
    return {'kappa': round(kappa, 4), 'concordancia': round(concordancia, 4)}

def auditar_equidade(y_true, y_pred, linguas, min_exemplos: int=20) -> dict:
    print('\n=== AUDITORIA DE EQUIDADE POR GRUPO LINGUÍSTICO ===')
    resultados = {}
    for lingua in ['pt', 'misto', 'xi']:
        idx = [i for i, l in enumerate(linguas) if l == lingua]
        if len(idx) < min_exemplos:
            print(f'  [{lingua}] Poucos exemplos ({len(idx)}) — ignorado')
            continue
        y_t = [y_true[i] for i in idx]
        y_p = [y_pred[i] for i in idx]
        f1 = f1_score(y_t, y_p, average='macro', labels=CLASSES, zero_division=0)
        resultados[lingua] = round(f1, 4)
        print(f'  F1-Macro [{lingua:5s}]: {f1:.4f}  (n={len(idx)})')
    if len(resultados) >= 2:
        diff = max(resultados.values()) - min(resultados.values())
        print(f'\n  Diferença máxima entre grupos: {diff:.4f}')
        if diff > 0.15:
            print('  ATENCAO: diferenca significativa — possivel vies algoritmico!')
        else:
            print('  Equidade aceitavel entre grupos linguisticos')
    return resultados
if __name__ == '__main__':
    np.random.seed(SEED)
    n = 60
    y_true = np.random.choice(CLASSES, n).tolist()
    y_pred = np.random.choice(CLASSES, n).tolist()
    linguas = np.random.choice(['pt', 'misto', 'xi'], n).tolist()
    res = calcular_metricas(y_true, y_pred, nome_modelo='Exemplo Aleatório')
    df_kappa = pd.DataFrame({'anotador_1': np.random.choice(CLASSES, 50), 'anotador_2': np.random.choice(CLASSES, 50)})
    calcular_kappa(df_kappa)
    auditar_equidade(y_true, y_pred, linguas)