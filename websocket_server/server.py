import asyncio
import websockets
import json
from threading import Thread

class WebSocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None
        self._thread = None

    async def _handler(self, websocket):
        self.clients.add(websocket)
        print(f"📡 New Client Connected! (Total: {len(self.clients)})")
        try:
            async for msg in websocket:
                print(f"📨 Received: {msg}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print("❌ Client Disconnected")

    async def _run_server(self):
        """Main coroutine to run the server."""
        print(f"⏳ Starting WebSocket on ws://{self.host}:{self.port}...")
        # Capture the loop so we can use it for threadsafe broadcasting later
        self.loop = asyncio.get_running_loop()
        
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"✅ WebSocket Server active")
            await asyncio.Future()  # Run forever

    def _start_thread(self):
        """Entry point for the thread."""
        try:
            asyncio.run(self._run_server())
        except Exception as e:
            print(f"❌ WebSocket Thread Error: {e}")

    def start(self):
        """Starts the server in a separate daemon thread."""
        self._thread = Thread(target=self._start_thread, daemon=True)
        self._thread.start()

    async def _broadcast(self, data):
        if not self.clients:
            return
        message = json.dumps(data)
        # Create tasks for all sends
        await asyncio.gather(*[client.send(message) for client in self.clients], return_exceptions=True)

    def broadcast_data(self, data):
        """Thread-safe method to send data to all connected clients."""
        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(data), self.loop)