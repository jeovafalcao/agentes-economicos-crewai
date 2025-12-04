#%%
import os
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import AzureChatOpenAI
from crewai_tools import SerperDevTool

#%%
load_dotenv()

tool = SerperDevTool()
# %%
llm = AzureChatOpenAI(
    model = "azure/" + os.getenv("AZURE_OPENAI_DEPLOYMENT_LLM"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = os.getenv("AZURE_API_VERSION"),
    temperature=0.3
)
#%%
try:
    df_top_10_acoes = pd.read_csv("../data/top_10_acoes.csv")
    df_noticias_investimento = pd.read_csv("../data/noticias_investimentos.csv")
    df_indices = pd.read_csv("../data/indicadores_economicos.csv")
except FileNotFoundError as e:
    print(f'Erro: Arquivo CSV nao encontrado. Verifique os nomes e caminhos dos arquivos: {e}')
    print(f'Certifique-se que "top_10_acoes.csv", "noticias_investimentos.csv" e "indicadores_economicos.csv" estao na pasta "data".')
    exit()

contexto_top_10_acoes = df_top_10_acoes.to_markdown(index=False)
contexto_indices = df_indices.to_markdown(index=False)

# Assumindo que df_noticias_investimenttos tem coluna 'titulo', 'resumo', 'link'
# Ajuste se os nomes das colunas forem diferentes
contexto_noticias_investimentos = "\n".join([
    f'Titulo: {row["titulo"]}\nLink: {row["link"]}'
    for _, row in df_noticias_investimentos.iterrows()
]) if not df_noticias_investimentos.empty else "Nenhuma noticia de investimento carregada do CSV."

# == Juntar todo o contexto BASE ==
contexto_geral_csv = f"""
=== Dados historicos de indices economicos ===
{contexto_indices}

=== Noticias de investimento recentes ===
{contexto_noticias_investimentos}

=== Top 10 acoes ===
{contexto_top_10_acoes}
"""

azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_LLM")
#%%
# Criar a orquestracao dos agentes
analista_macroeconomico = Agent(
    role='Analista Macroeconomico Senior',
    goal='Analisar mercados, avaliar comportamentos de consumidores e empresas, estudar oferta, demanda e preços, examinar custos e estruturas de mercado, modelar cenários econômicos, interpretar dados e fornecer subsídios para decisões estratégicas.',
    backstory='Economista com vasta experiencia na analise da conjuntura brasileira, indicadores economicos e seus efeitos sobre o mercado',
    verbose=True,
    allow_delegation=False,
    tools=[tool],
    llm=llm,
    model_name=f"azure/{azure_deployment_name}"
)

especialista_em_acoes = Agent(
    role='Especialista em Analise de Acoes da Bovespa',
    goal='Analisar empresas e setores, avaliar fundamentos e indicadores financeiros, estudar tendências de mercado, monitorar preços e volumes negociados, interpretar notícias e eventos econômicos, projetar cenários e valuations, recomendar compra, venda ou manutenção de ações, gerenciar riscos, acompanhar portfólios e comunicar insights e estratégias a clientes ou gestores.',
    backstory='Economista com vasta experiencia em analise do mercado de acoes, com expertise em valuation de empresas e estrategias de investimentos',
    verbose=True,
    allow_delegation=False,
    tools=[tool],
    llm=llm,
    model_name=f"azure/{azure_deployment_name}"
)

redator_de_relatorios_de_investimento = Agent(
    role='Redator de Relatorios de Investimentos',
    goal='Pesquisar informações de mercado e dados financeiros, analisar cenários econômicos e ativos, interpretar indicadores e tendências, organizar informações de forma clara, escrever relatórios técnicos e narrativas estratégicas, sintetizar insights complexos, estruturar recomendações de investimento conforme diretrizes dos analistas, revisar conteúdos para precisão e conformidade, e comunicar análises de forma objetiva e acessível ao público-alvo.',
    backstory='Profissional de comunicacao com foco no mercado financeiro, especializado em transformar analises tecnicas complexas em relatorios e informacoes',
    verbose=True,
    allow_delegation=False,
    tools=[tool],
    llm=llm,
    model_name=f"azure/{azure_deployment_name}"
)

#%%
# Definir as tasks
tarefa_analise_cenario = Task(
    description=("prompt" f"{contexto_geral_csv}"),
    expected_output=("Um relatorio conciso sobre o cenario macroeconomico brasileiro, destacando: \n"
                     "- Analise da trajetoria recente dos indices obtidos e suas perspectivas. \n"
                     "- Principais noticias e eventos de investimentos relevantes (do CSV e da pesquisa online). \n"
                     "- impactos esperados desse cenario no mercado de acoes brasileiro em geral."),
    agent= analista_macroeconomico
)

tarefa_indicacao_acoes = Task(
    description=("prompt" f"{contexto_top_10_acoes}"),
    expected_output=("Um relatorio de indicacoes de acoes contendo: \n"
                     "- Recomendacoes claras de COMPRA, VENDA ou MANTER para 3 a 5 acoes da Bovespa. \n"
                     "- Justificativa detalhada para cada recomendacao, explicando os fatores considerados. \n"
                     "- Priorizar as acoes do 'top_10_acoes.csv' na analise, mas incluir outras se forem identificadas oportunidades ou riscos relevantes."),
    agent= especialista_em_acoes,
    context=[tarefa_analise_cenario]
)

tarefa_compilacao_relatorio_final = Task(
    description=("prompt" f"{contexto_top_10_acoes}"),
    expected_output=("O Texto completo e final de um relatorio de investimento em formato Markdown na lingua portuguesa do Brasil: \n"
                     ),
    agent= redator_de_relatorios_de_investimento,
    context=[tarefa_analise_cenario, tarefa_indicacao_acoes]
)

crew_recomendacao_de_acoes = Crew(
    agents=[analista_macroeconomico, especialista_em_acoes, redator_de_relatorios_de_investimento],
    tasks=[tarefa_analise_cenario, tarefa_indicacao_acoes, tarefa_compilacao_relatorio_final],
    verbose=True,
    manager_llm=llm,
    #process=Process.hierarchical # Habilita o "gerente" para orquestrar com mais raciocinio
)