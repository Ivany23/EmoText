import re
import hashlib
import pandas as pd

def limpar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ''
    texto = re.sub('http\\S+|www\\.\\S+', '', texto)
    texto = re.sub('@\\w+', '', texto)
    texto = re.sub('<[^>]+>', '', texto)
    texto = re.sub('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]', '', texto)
    texto = re.sub('\\s+', ' ', texto).strip()
    return texto

def anonimizar_autor(autor_id: str) -> str:
    if not autor_id or pd.isna(autor_id):
        return 'anon_desconhecido'
    return hashlib.sha256(str(autor_id).encode('utf-8')).hexdigest()[:16]

def deduplicate(df: pd.DataFrame, col_texto: str='texto') -> pd.DataFrame:
    antes = len(df)
    df = df.drop_duplicates(subset=[col_texto], keep='first').reset_index(drop=True)
    depois = len(df)
    print(f'[Cleaner] Deduplicação: {antes} → {depois} comentários ({antes - depois} removidos)')
    return df

def filtrar_invalidos(df: pd.DataFrame, col_texto: str='texto', min_tokens: int=3) -> pd.DataFrame:
    antes = len(df)
    df = df[df[col_texto].str.split().str.len() >= min_tokens].reset_index(drop=True)
    depois = len(df)
    print(f'[Cleaner] Filtro min_tokens={min_tokens}: {antes} → {depois} ({antes - depois} removidos)')
    return df

def pipeline_limpeza(df: pd.DataFrame, col_texto: str='texto', col_autor: str='autor_id') -> pd.DataFrame:
    print('\n=== PIPELINE DE LIMPEZA TEXTUAL ===')
    print(f'Comentários iniciais: {len(df)}')
    df = df.copy()
    df['texto_limpo'] = df[col_texto].apply(limpar_texto)
    if col_autor in df.columns:
        df['autor_hash'] = df[col_autor].apply(anonimizar_autor)
        df.drop(columns=[col_autor], inplace=True)
    df = deduplicate(df, col_texto='texto_limpo')
    df = filtrar_invalidos(df, col_texto='texto_limpo', min_tokens=3)
    print(f'Comentários finais: {len(df)}')
    print('=== LIMPEZA CONCLUIDA ===\n')
    return df
if __name__ == '__main__':
    dados_exemplo = {'texto': ['Ndza tenda swinene! https://t.co/abc123 @utilizador', 'O governo <b>anunciou</b> novas medidas hoje', 'ok', 'Xikwembu wa pfuna hinkwavu va munhu misava', 'Ndza tenda swinene! https://t.co/abc123 @utilizador'], 'autor_id': ['user_123', 'user_456', 'user_789', 'user_123', 'user_999']}
    df = pd.DataFrame(dados_exemplo)
    df_limpo = pipeline_limpeza(df, col_texto='texto', col_autor='autor_id')
    print(df_limpo[['texto_limpo', 'autor_hash']].to_string())