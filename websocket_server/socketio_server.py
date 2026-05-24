import asyncio
import socketio
from aiohttp import web
from threading import Thread


class SocketIOServer:
    """Simple Socket.IO server running in a background thread.

    The class exposes a subset of the previous WebSocketServer API so that
    switching in ``system_controller`` is straightforward: ``start()`` and
    ``broadcast_data``.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sio = socketio.AsyncServer(
            async_mode="aiohttp", cors_allowed_origins="*"
        )
        self.app = web.Application()
        self.sio.attach(self.app)
        self._thread = None

        # register a couple of simple events for logging
        @self.sio.event
        async def connect(sid, environ):
            print(f"📡 socket.io client connected: {sid}")

        @self.sio.event
        async def disconnect(sid):
            print(f"❌ socket.io client disconnected: {sid}")

    def _run_app(self):
        # When running in a background thread, aiohttp.web.run_app tries to
        # install signal handlers which only works in the main thread.  That
        # was causing ``RuntimeError: set_wakeup_fd only works in main thread``
        # during startup.
        #
        # Instead we create a dedicated event loop for the thread and start the
        # app using AppRunner/TCPSite manually.  This avoids signal handling and
        # behaves well inside a non-main thread.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def start_server():
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, host=self.host, port=self.port)
            await site.start()
            print(f"✅ Socket.IO Server active on http://{self.host}:{self.port}")
            # keep the loop running forever
            await asyncio.Event().wait()

        try:
            loop.run_until_complete(start_server())
        except Exception as e:
            print(f"❌ Socket.IO thread error: {e}")
        finally:
            loop.close()

    def start(self):
        """Start the socket.io server in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = Thread(target=self._run_app, daemon=True)
        self._thread.start()
        print(f"⏳ Starting Socket.IO on http://{self.host}:{self.port}...")

    def broadcast_data(self, data):
        """Emit a ``status`` event to all connected clients."""
        # ``start_background_task`` is thread-safe and will schedule the
        # coroutine on the event loop used by the server.
        self.sio.start_background_task(self.sio.emit, "status", data)
