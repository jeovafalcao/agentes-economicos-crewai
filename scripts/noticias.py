import requests
from bs4 import BeautifulSoup
import pandas as pd

palavras_chaves = [
    "ipca", "inflação", "selic", "juros", "bovespa", "ações", "investimentos",
    "bolsa", "ibovespa", "economia", "mercado", "taxa básica", "taxa de juros"
]

headers = {"User-Agent": "Mozilla/5.0"}

sites = {
    "CNN Brasil": "https://www.cnnbrasil.com.br/economia/",
    "G1 Economia": "https://g1.globo.com/economia/",
    "InfoMoney": "https://www.infomoney.com.br/mercados/",
    "Exame Economia": "https://www.exame.com/economia/",

}
    
noticias = []

def filtrar_noticias(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    encontrados = []
    for a in soup.find_all('a', href=True):
        titulo = a.get_text().strip().lower()
        link  = a['href']
        if any(p in titulo for p in palavras_chaves):
            if titulo and link.startswith('http'):
                encontrados.append({'titulo': titulo.title(), 'link': link})
            elif titulo and link.startswith('/'):
                encontrados.append({'titulo': titulo.title(), 'link': base_url + link})
    return encontrados

for nome_site, url in sites.items():
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            base_url = "/".join(url.split("/")[:-3])
            noticias += filtrar_noticias(resp.text, base_url)
        else:
            print(f"[!] Erro ao acessar {nome_site}: Status {resp.status_code}")
    except Exception as e:
        print(f"[!] Erro ao acessar {nome_site}: {e}")

# Remover duplicadas
noticias_unicas = list({n["titulo"]: n for n in noticias}.values())
# Salvar CSV
df = pd.DataFrame(noticias_unicas)
                
print(f"CSV gerado com {len(df)} noticias: noticias_investimentos_csv")

df.to_csv("../data/noticias_investimentos_csv", index=False, encoding='utf-8-sig')
