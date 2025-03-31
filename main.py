from fastapi import FastAPI, Query
from multiagent_rag import MultiAgentRAG

# Initialize FastAPI app
app = FastAPI()
chatbot = MultiAgentRAG()

@app.get("/ask")
def ask_question(query: str = Query(..., description="User question about HR")):
    response = chatbot.get_response(query)
    return {"answer": response, "CODE": "200"}

@app.get("/agents")
def list_agents():
    return {"agents": chatbot.get_agents()}
