import threading
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Live Chess Board</title>
  <style>
    body { 
      font-family: Arial; 
      text-align: center; 
      margin-top: 40px;
      background: #312e2b;
      color: #fff;
    }
    
    #board {
      display: inline-grid;
      grid-template-columns: repeat(8, 60px);
      grid-template-rows: repeat(8, 60px);
      border: 3px solid #1a1715;
      box-shadow: 0 5px 15px rgba(0,0,0,0.5);
      margin: 20px auto;
    }
    
    .square {
      width: 60px;
      height: 60px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 45px;
      position: relative;
    }
    
    .light { background: #f0d9b5; }
    .dark { background: #b58863; }
    
    #fen { 
      font-size: 14px; 
      margin-top: 20px; 
      word-wrap: break-word;
      max-width: 600px;
      margin-left: auto;
      margin-right: auto;
      font-family: monospace;
      background: #1a1715;
      padding: 15px;
      border-radius: 5px;
    }
    
    h1 {
      color: #f0d9b5;
    }
    
    .status {
      margin: 10px;
      padding: 8px;
      border-radius: 4px;
      display: inline-block;
    }
    
    .connected { background: #5cb85c; }
    .disconnected { background: #d9534f; }
    .waiting { background: #f0ad4e; }
  </style>
</head>
<body>
  <h1>Live Chess Board</h1>
  <div id="status" class="status waiting">Connecting...</div>
  <div id="board"></div>
  <div id="fen">Waiting for data...</div>

  <script>
    const pieceSymbols = {
      'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
      'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
    };

    function createBoard() {
      const board = document.getElementById('board');
      board.innerHTML = '';
      
      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          const square = document.createElement('div');
          square.className = 'square ' + ((row + col) % 2 === 0 ? 'light' : 'dark');
          square.dataset.row = row;
          square.dataset.col = col;
          board.appendChild(square);
        }
      }
    }

    function updateBoard(fen) {
      const fenParts = fen.split(' ');
      const position = fenParts[0];
      const ranks = position.split('/');
      
      ranks.forEach((rank, rowIdx) => {
        let colIdx = 0;
        for (let char of rank) {
          if (char >= '1' && char <= '8') {
            colIdx += parseInt(char);
          } else {
            const square = document.querySelector(`[data-row="${rowIdx}"][data-col="${colIdx}"]`);
            if (square) {
              const piece = pieceSymbols[char] || '';
              square.textContent = piece;
              square.style.color = char === char.toUpperCase() ? '#fff' : '#000';
            }
            colIdx++;
          }
        }
      });
    }

    createBoard();

    const ws = new WebSocket(`ws://${location.host}/ws`);
    const statusEl = document.getElementById('status');

    ws.onopen = () => {
      statusEl.textContent = 'Connected';
      statusEl.className = 'status connected';
    };

    ws.onmessage = (event) => {
      const fen = event.data;
      document.getElementById('fen').innerText = fen;
      updateBoard(fen);
    };

    ws.onclose = () => {
      statusEl.textContent = 'Disconnected';
      statusEl.className = 'status disconnected';
      document.getElementById('fen').innerText = 'Connection closed';
    };

    ws.onerror = () => {
      statusEl.textContent = 'Connection Error';
      statusEl.className = 'status disconnected';
    };
  </script>
</body>
</html>
"""


class FenServer:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port

        self.app = FastAPI()
        self.clients = set()
        self.current_fen = "startpos"

        self._setup_routes()

        self.loop = None

    # -------------------------
    # Public API
    # -------------------------
    def start(self):
        """Start server in background thread."""
        thread = threading.Thread(target=self._run_server, daemon=True)
        thread.start()

    def update_fen(self, fen: str):
        """Update FEN and push to all clients."""
        self.current_fen = fen
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(fen),
                self.loop
            )

    # -------------------------
    # Internal
    # -------------------------
    def _setup_routes(self):
        @self.app.get("/")
        async def index():
            return HTMLResponse(HTML_PAGE)

        @self.app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self.clients.add(ws)

            # send current state immediately
            await ws.send_text(self.current_fen)

            try:
                while True:
                    await ws.receive_text()  # keep alive
            except WebSocketDisconnect:
                self.clients.remove(ws)

    async def _broadcast(self, message: str):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(message)
            except:
                dead.append(ws)

        for ws in dead:
            self.clients.discard(ws)

    def _run_server(self):
        config = uvicorn.Config(self.app, host=self.host,
                                port=self.port, log_level="info")
        server = uvicorn.Server(config)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.loop.run_until_complete(server.serve())
