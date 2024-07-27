from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketState, WebSocketDisconnect
import uvicorn
import logging

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except Exception as e:
        if isinstance(e, WebSocketDisconnect):
            if e.code == 1000:
                logging.info(f"Connection closed with code: {e.code}")
            else:
                logging.error(f"Error: {e}")
        else:
            logging.error(f"Unexpected error: {e}")
    finally:
        if websocket.client_state in {WebSocketState.CONNECTED, WebSocketState.CONNECTING}:
            try:
                await websocket.close()
            except Exception as e:
                logging.error(f"Error while closing websocket: {e}")
        logging.info("Connection closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
