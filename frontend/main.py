# pgvector-ann/frontend/main.py
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import os
import json
import websockets
import asyncio
import httpx

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

BACKEND_URL = os.getenv("BACKEND_URL", "ws://backend:8001")
BACKEND_HTTP_URL = os.getenv("BACKEND_HTTP_URL", "http://backend:8001")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("/app/static/fastapi-1.svg", media_type="image/svg+xml")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/pdf/{file_name}")
async def stream_pdf(file_name: str, page: int = None):
    url = f"{BACKEND_HTTP_URL}/pdf/{file_name}"
    if page is not None:
        url += f"?page={page}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, stream=True)
            response.raise_for_status()

            return StreamingResponse(
                response.iter_bytes(),
                media_type="application/pdf",
                headers={"Content-Disposition": f'inline; filename="{file_name}"'}
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))

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
