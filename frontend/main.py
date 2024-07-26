# pgvector-ann/frontend/main.py
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import json
import websockets
import asyncio

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

BACKEND_URL = os.getenv("BACKEND_URL", "ws://backend:8001")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    backend_ws_url = f"{BACKEND_URL}/ws"

    try:
        async with websockets.connect(backend_ws_url) as backend_ws:
            await asyncio.gather(
                forward_to_backend(websocket, backend_ws),
                forward_to_client(websocket, backend_ws)
            )
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {str(e)}")

async def forward_to_backend(client_ws: WebSocket, backend_ws: websockets.WebSocketClientProtocol):
    try:
        while True:
            data = await client_ws.receive_text()
            await backend_ws.send(data)
    except WebSocketDisconnect:
        await backend_ws.close()

async def forward_to_client(client_ws: WebSocket, backend_ws: websockets.WebSocketClientProtocol):
    try:
        while True:
            response = await backend_ws.recv()
            response_data = json.loads(response)
            if "error" in response_data:
                await client_ws.send_json({"error": response_data["error"]})
            else:
                await client_ws.send_json(response_data)
    except WebSocketDisconnect:
        await client_ws.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
