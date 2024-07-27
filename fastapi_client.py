import asyncio
import websockets

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello, WebSocket!")
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(test_websocket())
