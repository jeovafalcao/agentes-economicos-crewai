import requests
import pandas as pd
from datetime import datetime

indicadores = {
    "IPCA": 433,
    "SELIC": 432,
    "PIB": 4380,
    "DOLAR": 1,
    "COMMODITIES": 22795,
    "IGP-M": 189
}

def coletar_indicadores_bacen(indicadores: dict, n_ultimos: int = 20) -> pd.DataFrame:
    todos_dados = []
    
    for nome, codigo in indicadores.items():
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados/ultimos/{n_ultimos}?formato=json"
        response = requests.get(url)
        
        if response.status_code == 200:
            dados = response.json()
            df = pd.DataFrame(dados)
            df['valor'] = df["valor"].str.replace(",",".").astype(float)
            df['indicador'] = nome
            df['data_coleta'] = datetime.now().date()
            todos_dados.append(df)
        else:
            print(f"Erro ao buscar {nome} (codigo {codigo}). Status: {response.status_code}")
 
    df_final = pd.concat(todos_dados, ignore_index=True)
    df_final.rename(columns={'data': 'data', 'valor': 'valor'}, inplace=True)
    return df_final      

df_indicadores = coletar_indicadores_bacen(indicadores)
df_indicadores.to_csv("../data/indicadores_economicos.csv", index=False, encoding='utf-8-sig')
print('Arquivo salvo com sucesso.')

