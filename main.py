from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from paho.mqtt import client as mqtt_client
import json
import asyncio
import logging
import os
import time
import csv
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# --- Configuration Loading ---
with open('config.json', 'r') as f:
    config = json.load(f)

MQTT_BROKER = config['mqtt_broker_ip']
MQTT_PORT = config['mqtt_broker_port']
WEB_APP_PORT = config['web_app_port']
TOURNAMENT_NAME = config.get('tournament_name', 'tournament_1')

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("logs/controller.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("CONTROLLER")

# --- FastAPI App Setup ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# --- Global State for Controller ---
connected_dashboard_clients = {}
node_states = defaultdict(lambda: {"state": "unknown", "last_seen": 0, "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"})
fastapi_loop = None

# Track file modification times for change detection
file_mtimes = {}

# Heartbeat monitoring
HEARTBEAT_TIMEOUT = 90  # seconds - if no update for 90s, mark as disconnected

# --- MQTT Client for Controller ---
mqtt_client_instance = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id="controller")

def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        logger.info(f"Controller connected to MQTT Broker: {MQTT_BROKER}")
        result1 = client.subscribe("chess/+/fen")
        result2 = client.subscribe("chess/+/status")
        logger.info(f"Subscribed to topics: chess/+/fen, chess/+/status")
    else:
        logger.error(f"Controller failed to connect to MQTT Broker, return code {rc}")

async def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload = json.loads(msg.payload.decode())
        logger.info(f"MQTT message received on topic: {topic}")
        
        pi_id = topic.split('/')[1]

        if topic.endswith("/fen"):
            fen = payload.get("fen")
            game_id = payload.get("game_id", "default_game")
            timestamp = payload.get("timestamp", int(time.time()))

            node_states[pi_id]["fen"] = fen
            node_states[pi_id]["last_seen"] = int(time.time())
            
            # Save to tournament folder: tournament_name/pi_id/fen_log.csv
            pi_folder = os.path.join(TOURNAMENT_NAME, pi_id)
            os.makedirs(pi_folder, exist_ok=True)
            csv_path = os.path.join(pi_folder, "fen_log.csv")
            
            file_exists = os.path.isfile(csv_path)
            
            try:
                with open(csv_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["timestamp", "fen"])
                    writer.writerow([timestamp, fen])
                logger.info(f"Saved FEN for {pi_id} to {csv_path}")
            except Exception as e:
                logger.error(f"Error saving FEN: {e}", exc_info=True)

            # Broadcast FEN update to all connected clients
            await broadcast_to_clients({
                "type": "FEN_UPDATE", 
                "pi_id": pi_id, 
                "fen": fen, 
                "timestamp": timestamp, 
                "game_id": game_id
            })
                
        elif topic.endswith("/status"):
            state = payload.get("state")
            last_seen = int(time.time())
            
            # Update node state
            node_states[pi_id].update({"state": state, "last_seen": last_seen})
            
            # Log special states
            if state == "DISCONNECTED":
                logger.warning(f"Pi {pi_id} disconnected (LWT triggered)")
            elif state == "OFFLINE":
                logger.info(f"Pi {pi_id} went offline gracefully")
            
            # Broadcast status update
            await broadcast_to_clients({
                "type": "STATUS_UPDATE", 
                "pi_id": pi_id, 
                "state": state, 
                "last_seen": last_seen
            })
        
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON received on {topic}: {msg.payload}")
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}", exc_info=True)

async def broadcast_to_clients(message):
    """Broadcast message to all connected WebSocket clients"""
    disconnected_clients = []
    for client_id, client_ws in list(connected_dashboard_clients.items()):
        try:
            if client_ws.client_state.name == "CONNECTED":
                await client_ws.send_json(message)
            else:
                disconnected_clients.append(client_id)
        except Exception as e:
            logger.warning(f"Failed to send to client {client_id}: {e}")
            disconnected_clients.append(client_id)
            
    for client_id in disconnected_clients:
        connected_dashboard_clients.pop(client_id, None)

def on_message_threaded_wrapper(client, userdata, msg):
    """Wrapper to schedule async on_message from MQTT thread to FastAPI event loop"""
    if fastapi_loop:
        future = asyncio.run_coroutine_threadsafe(on_message(client, userdata, msg), fastapi_loop)
        try:
            future.result(timeout=5.0)
        except Exception as e:
            logger.error(f"Error waiting for on_message coroutine: {e}")
    else:
        logger.error("FastAPI event loop not available to schedule MQTT message.")

# --- Heartbeat Monitor Thread ---
def heartbeat_monitor():
    """Monitor Pi nodes for timeouts and mark as disconnected"""
    logger.info("Heartbeat monitor started")
    
    while True:
        try:
            current_time = int(time.time())
            
            for pi_id, state_info in list(node_states.items()):
                last_seen = state_info.get("last_seen", 0)
                current_state = state_info.get("state", "unknown")
                time_since_last_seen = current_time - last_seen if last_seen > 0 else 999999
                
                # Check if node has timed out (no heartbeat for 90+ seconds)
                if time_since_last_seen > HEARTBEAT_TIMEOUT:
                    if current_state not in ["DISCONNECTED", "OFFLINE", "unknown"]:
                        logger.warning(f"Pi {pi_id} heartbeat timeout ({time_since_last_seen}s) - marking as DISCONNECTED")
                        node_states[pi_id]["state"] = "DISCONNECTED"
                        
                        # Notify clients
                        if fastapi_loop:
                            asyncio.run_coroutine_threadsafe(
                                broadcast_to_clients({
                                    "type": "STATUS_UPDATE",
                                    "pi_id": pi_id,
                                    "state": "DISCONNECTED",
                                    "last_seen": last_seen
                                }),
                                fastapi_loop
                            )
                
                # If was DISCONNECTED but now has recent heartbeat, restore to IDLE
                elif time_since_last_seen < 30:  # Heartbeat within last 30 seconds
                    if current_state == "DISCONNECTED":
                        logger.info(f"Pi {pi_id} reconnected - restoring to STREAMING")
                        node_states[pi_id]["state"] = "STREAMING"
                        
                        # Notify clients
                        if fastapi_loop:
                            asyncio.run_coroutine_threadsafe(
                                broadcast_to_clients({
                                    "type": "STATUS_UPDATE",
                                    "pi_id": pi_id,
                                    "state": "STREAMING",
                                    "last_seen": last_seen
                                }),
                                fastapi_loop
                            )
            
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}", exc_info=True)
            time.sleep(10)


# --- File Watcher Thread ---
def tournament_file_watcher():
    """Watch tournament CSV files for changes and notify clients"""
    logger.info("Tournament file watcher started")
    
    while True:
        try:
            # Scan all tournament folders
            for tournament_dir in Path(".").glob("tournament*"):
                if tournament_dir.is_dir():
                    # Check each Pi's CSV file
                    for pi_dir in tournament_dir.iterdir():
                        if pi_dir.is_dir():
                            pi_id = pi_dir.name  # Add this line
                            csv_path = pi_dir / "fen_log.csv"
                            if csv_path.exists():
                                mtime = csv_path.stat().st_mtime
                                file_key = str(csv_path)
                                
                                # Check if file was modified
                                if file_key not in file_mtimes or file_mtimes[file_key] < mtime:
                                    file_mtimes[file_key] = mtime
                                    logger.info(f"File changed: {csv_path}")
                                    
                                    # Read the latest FEN from the file
                                    try:
                                        with open(csv_path, "r") as f:
                                            lines = f.readlines()
                                            if len(lines) > 1:  # Skip header
                                                last_line = lines[-1].strip()
                                                if last_line:
                                                    timestamp, fen = last_line.split(',', 1)
                                                    
                                                    # Notify clients with actual FEN data
                                                    if fastapi_loop:
                                                        asyncio.run_coroutine_threadsafe(
                                                            broadcast_to_clients({
                                                                "type": "FEN_UPDATE",
                                                                "pi_id": pi_id,
                                                                "fen": fen,
                                                                "timestamp": int(timestamp)
                                                            }),
                                                            fastapi_loop
                                                        )
                                    except Exception as e:
                                        logger.error(f"Error reading latest FEN from {csv_path}: {e}")
            
            time.sleep(1)  # Check every second
            
        except Exception as e:
            logger.error(f"Error in file watcher: {e}", exc_info=True)
            time.sleep(1)


# --- MQTT Loop in background ---
@app.on_event("startup")
async def startup_event():
    global fastapi_loop
    fastapi_loop = asyncio.get_running_loop()
    mqtt_client_instance.on_connect = on_connect
    mqtt_client_instance.on_message = on_message_threaded_wrapper
    mqtt_client_instance.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    mqtt_client_instance.loop_start()
    
    # Start file watcher thread
    watcher_thread = threading.Thread(target=tournament_file_watcher, daemon=True)
    watcher_thread.start()
    
    # Start heartbeat monitor thread
    heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    heartbeat_thread.start()
    
    logger.info("Background threads started")

@app.on_event("shutdown")
async def shutdown_event():
    mqtt_client_instance.loop_stop()
    mqtt_client_instance.disconnect()

# --- API Endpoint for Commands ---
@app.post("/api/command/{pi_id}")
async def send_pi_command(pi_id: str, command: dict):
    action = command.get("action")
    game_id = command.get("game_id")
    
    fen_csv_to_stream = "/home/justagoat/Desktop/networking_thesis/chess-node/games/game1/game.csv"
    # This path is hardcoded for now; in a real system, it might be dynamic.
    
    if action:
        payload = {"action": action}
        if action == "start_recording":
            payload["game_id"] = game_id
            payload["fen_csv_path"] = fen_csv_to_stream
        mqtt_client_instance.publish(f"chess/cmd/{pi_id}", json.dumps(payload), qos=1)
        logger.info(f"Published command '{action}' for '{pi_id}'")
        return {"status": "Command sent", "pi_id": pi_id, "action": action}
    else:
        raise HTTPException(status_code=400, detail="'action' field is required in command.")

# --- HTTP Root Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- API Endpoint to get tournaments ---
@app.get("/api/tournaments")
async def get_tournaments():
    """Return list of all tournaments and their FEN data"""
    tournaments_dict = {}
    tournament_list = []
    
    # Scan for tournament folders
    for tournament_dir in Path(".").glob("tournament*"):
        if tournament_dir.is_dir():
            tournament_name = tournament_dir.name
            tournament_list.append(tournament_name)
            tournaments_dict[tournament_name] = {}
            
            # Load FENs for each Pi
            for pi_dir in tournament_dir.iterdir():
                if pi_dir.is_dir():
                    pi_id = pi_dir.name
                    csv_path = pi_dir / "fen_log.csv"
                    
                    fens = []
                    if csv_path.exists():
                        try:
                            with open(csv_path, "r") as f:
                                reader = csv.DictReader(f)
                                for row in reader:
                                    if row.get("fen"):
                                        fens.append(row["fen"])
                        except Exception as e:
                            logger.error(f"Error reading {csv_path}: {e}")
                    
                    tournaments_dict[tournament_name][pi_id] = fens
    
    return {
        "tournament_list": sorted(tournament_list),
        "tournaments": tournaments_dict
    }

# --- API Endpoint to get Pi status ---
@app.get("/api/pi-status")
async def get_pi_status():
    """Return current status of all Pi nodes"""
    return {
        "nodes": {
            pid: {
                "state": status.get("state", "unknown"), 
                "last_seen": status.get("last_seen", 0)
            } 
            for pid, status in node_states.items()
        }
    }

# --- WebSocket for Dashboard UI Updates ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(id(websocket))
    connected_dashboard_clients[client_id] = websocket
    logger.info(f"Dashboard client {client_id} connected via WebSocket.")

    try:
        # Send initial state to new client
        initial_state = {
            "type": "INITIAL_STATE",
            "nodes": {pid: status for pid, status in node_states.items()}
        }
        await websocket.send_json(initial_state)
        
        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except Exception:
                break
                
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {e}", exc_info=True)
    finally:
        connected_dashboard_clients.pop(client_id, None)
        logger.info(f"Dashboard client {client_id} disconnected.")