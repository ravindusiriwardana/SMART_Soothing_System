import asyncio
import websockets
import json
from threading import Thread

class WebSocketServer:
    def __init__(self, host="0.0.0.0", port=5001):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self._thread = None

    async def _handler(self, websocket):
        self.clients.add(websocket)
        print(f"📡 Client Connected! Total: {len(self.clients)}")
        try:
            async for msg in websocket:
                # Handle incoming requests if needed
                data = json.loads(msg)
                print(f"📩 Client Request: {data}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print("❌ Client Disconnected")

    async def _run_server(self):
        self.loop = asyncio.get_running_loop()
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future() 

    def start(self):
        self._thread = Thread(target=lambda: asyncio.run(self._run_server()), daemon=True)
        self._thread.start()
        print(f"🚀 Server started on ws://{self.host}:{self.port}")

    async def _broadcast(self, data):
        if not self.clients: return
        message = json.dumps(data)
        await asyncio.gather(*[client.send(message) for client in self.clients], return_exceptions=True)

    def broadcast_data(self, data):
        """
        Call this from your AI Loop:
        ws_server.broadcast_data({
            "emotion": "hungry", 
            "confidence": 0.95, 
            "posture": "lying", 
            "posture_confidence": 0.88
        })
        """
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(data), self.loop)