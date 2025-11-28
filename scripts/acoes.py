import requests
import pandas as pd
from time import sleep
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

top_10_acoes = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "B3SA3", "WEGE3", "RENT3", "MGLU3"]


def buscar_dados_acao_alpha_vantage(ticker_b3, api_key, num_registros=10):
    ticker = ticker_b3 + ".SA"
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}&outputsize=compact"
    response = requests.get(url)
    if response.status_code == 200:
        # Adiciona um tratamento para o erro 503, comum quando a API esta ocupada
        if response.status_code == 503:
            print(f"[{ticker_b3}] Servidor Alpha Vantage indisponivel (503). Tentando novamente em alguns segundos...")
            sleep(30)
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f'[{ticker_b3}] Erro {response.status_code} apos nova tentativa.')
            else:
                raise Exception(f'[{ticker_b3}] Erro {response.status_code}')
            
    data = response.json()
    # Verifica se a chave Time series (Daily) existe e tambem se ha alguma mensagem de erro/limite
    if "Note" in data or "Information" in data:
        print(f"[{ticker_b3}] Nota da API: {data.get('Note', data.get("Information", 'Limite de API provavelmente atingido.'))}")
        return None
    if "Time Series (Daily)" not in data:
        print(f"[{ticker_b3}] Sem dados 'Time Series (Daily)' na resposta. Resposta completa: {data}")
        return None
    
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df.columns = ["abertura", "alta", "baixa", "fechamento", "volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True)
    df["ticker"] = ticker_b3
    
    # Seleciona os ultimos num_registros
    df = df.tail(num_registros)
    
    return df

df_total = pd.DataFrame()

for ativo in top_10_acoes:
    try:
        print(f"Coletando {ativo}...")
        # Agora a funcao buscar_dados_acao_alpha_vantage buscara 20 registros por padrao
        df = buscar_dados_acao_alpha_vantage(ativo, api_key)
        
        if df is not None and not df.empty:
            df_total = pd.concat([df_total, df])
            print(f"{ativo} adicionado com {len(df)} registros.")
        elif df is not None and df.empty:
            print(f"{ativo} retornou um DataFrame vazio apos o processamento (pode ser que nao haja 20 dias de dados apos o filtro).")
            
        sleep(15) # mantem o sleep para respeitar os limites da API
    except Exception as e:
        print(f"Erro com {ativo}: {e}")

if not df_total.empty:
    df_total.to_csv("../data/top_10_acoes_csv", index=True, encoding='utf-8-sig')
    print('Arquivo salvo com sucesso.')
else:
    print("Nenhum dado foi coletado para salvar no arquivo CSV.")

