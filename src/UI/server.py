import threading
from flask import Flask, Response
from flask_sock import Sock

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Live Chess FEN</title>
  <style>
    body { font-family: Arial; text-align: center; margin-top: 40px; }
    #fen { font-size: 18px; margin-top: 20px; word-break: break-all; }
  </style>
</head>
<body>
  <h1>Current Position</h1>
  <div id="fen">Waiting...</div>

  <script>
    const ws = new WebSocket(`ws://${location.host}/ws`);

    ws.onmessage = e => {
      document.getElementById("fen").innerText = e.data;
    };

    ws.onclose = () => {
      document.getElementById("fen").innerText = "Disconnected";
    };
  </script>
</body>
</html>
"""


class FenServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port

        self.app = Flask(__name__)
        self.sock = Sock(self.app)

        self.current_fen = "startpos"
        self.clients = set()
        self.lock = threading.Lock()

        self._setup_routes()

    # -------------------
    # Public API
    # -------------------
    def start(self):
        """Run Flask server in background thread."""
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def update_fen(self, fen: str):
        """Update FEN and push to all connected clients."""
        self.current_fen = fen
        self._broadcast(fen)

    # -------------------
    # Internal
    # -------------------
    def _setup_routes(self):
        @self.app.route("/")
        def index():
            return Response(HTML_PAGE, mimetype="text/html")

        @self.sock.route("/ws")
        def websocket(ws):
            with self.lock:
                self.clients.add(ws)

            # send current state immediately
            ws.send(self.current_fen)

            try:
                while True:
                    ws.receive()  # keep connection alive
            finally:
                with self.lock:
                    self.clients.discard(ws)

    def _broadcast(self, message: str):
        dead = []
        with self.lock:
            for ws in self.clients:
                try:
                    ws.send(message)
                except:
                    dead.append(ws)

            for ws in dead:
                self.clients.discard(ws)

    def _run(self):
        # Flask dev server is enough for LAN / Pi usage
        self.app.run(host=self.host, port=self.port, threaded=True)
