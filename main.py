# main.py
import os
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from dotenv import load_dotenv

# Importa a função de stream que criamos acima
from src.research_analyzer import analisar_artigo_stream

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

PASTA_DATA = "data"
os.makedirs(PASTA_DATA, exist_ok=True)

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def enviar_progresso(self, client_id: str, mensagem: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json({"status": "progresso", "mensagem": mensagem})

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Passando o request de forma explícita por nome
    return templates.TemplateResponse(request=request, name="index.html")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(client_id)

@app.post("/upload/{client_id}")
async def processar_artigo_api(client_id: str, file: UploadFile = File(...)):
    caminho_temporario = os.path.join(PASTA_DATA, f"temp_{client_id}.pdf")
    try:
        with open(caminho_temporario, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Roda a função de análise enviando as notificações via WebSocket
        relatorio_md = await analisar_artigo_stream(caminho_temporario, manager, client_id)
        
        relatorio_html = relatorio_md.replace("\n", "<br>")

        if client_id in manager.active_connections:
            await manager.active_connections[client_id].send_json({
                "status": "concluido",
                "relatorio": relatorio_html
            })
    except Exception as e:
        if client_id in manager.active_connections:
            await manager.active_connections[client_id].send_json({"status": "erro", "mensagem": str(e)})
    finally:
        if os.path.exists(caminho_temporario):
            os.remove(caminho_temporario)

    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)