from fastapi import FastAPI
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import os

app = FastAPI(title="Agentes de Mercado Financeiro")

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

analista = Agent(
    role="Analista Sênior de Mercado Financeiro",
    goal="Fazer análises técnicas e fundamentalistas precisas de ações, criptomoedas, índices e economia",
    backstory="Você é um dos melhores analistas do mercado brasileiro e global. Tem anos de experiência em B3, NYSE, criptomoedas e macroeconomia. Sempre baseia suas análises em dados e lógica.",
    verbose=True,
    llm=llm
)

pesquisador = Agent(
    role="Pesquisador de Dados e Notícias Financeiras",
    goal="Buscar e organizar as informações mais relevantes e atualizadas sobre o mercado financeiro",
    backstory="Especialista em coletar e filtrar informações de alta qualidade sobre economia, empresas e tendências de mercado.",
    verbose=True,
    llm=llm
)

estrategista = Agent(
    role="Estrategista e Gestor de Portfólio",
    goal="Criar estratégias claras e recomendações práticas baseadas nas análises",
    backstory="Você é um gestor experiente que transforma análises em planos de ação objetivos e realistas para o investidor.",
    verbose=True,
    llm=llm
)

class Ordem(BaseModel):
    pedido: str

@app.get("/")
def home():
    return {
        "status": "online",
        "mensagem": "Agentes de Mercado Financeiro prontos para receber ordens",
        "agentes": ["Analista", "Pesquisador", "Estrategista"]
    }

@app.post("/ordem")
def executar_ordem(ordem: Ordem):
    tarefa = Task(
        description=f"""
        O usuário deu a seguinte ordem: {ordem.pedido}
        
        Trabalhem juntos para responder da melhor forma possível.
        Seja claro, objetivo e profissional.
        """,
        expected_output="Uma resposta completa, bem estruturada e útil para o usuário.",
        agent=estrategista
    )

    crew = Crew(
        agents=[pesquisador, analista, estrategista],
        tasks=[tarefa],
        process=Process.sequential,
        verbose=True
    )

    resultado = crew.kickoff()
    return {"resposta": str(resultado)}
