import threading
from flask import Flask, Response
from flask_sock import Sock

# .light {
#         /*background: #f0d9b5;*/
#         background: #bda37f;
#     }
#     .dark {
#         /*background: #b58863;*/
#         background: #7a583e;
#     }
#               square.style.color = char === char.toUpperCase() ? '#dbc9b4' : '#4e4743';


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
    
    .square::before {
      content: attr(data-piece);
      position: absolute;
      -webkit-text-stroke: 2px #fff;
      text-stroke: 2px #fff;
      z-index: 0;
    }
    
    .piece {
      position: relative;
      z-index: 1;
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
      'K': '♚', 'Q': '♛', 'R': '♜', 'B': '♝', 'N': '♞', 'P': '♟',
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
      
      // Clear all squares first
      document.querySelectorAll('.square').forEach(square => {
        square.innerHTML = '';
      });
      
      ranks.forEach((rank, rowIdx) => {
        let colIdx = 0;
        for (let char of rank) {
          if (char >= '1' && char <= '8') {
            colIdx += parseInt(char);
          } else {
            const square = document.querySelector(`[data-row="${rowIdx}"][data-col="${colIdx}"]`);
            if (square) {
              const piece = pieceSymbols[char] || '';
              square.innerHTML = `<span class="piece" style="color: ${char === char.toUpperCase() ? '#2c2c2c' : '#000'}; text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;">${piece}</span>`;
            }
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

        self.app = Flask(__name__)
        self.sock = Sock(self.app)

        self.current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
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
