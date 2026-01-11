import paho.mqtt.client as mqtt_client
import time
import json
import threading
import logging
import os
import csv
from datetime import datetime

# --- Configuration Loading ---
with open('config.json', 'r') as f:
    config = json.load(f)

DEVICE_ID = config['device_id']
MQTT_BROKER = config['mqtt_broker_ip']
MQTT_PORT = config['mqtt_broker_port']

# --- Global State for Pi Node ---
node_state = "IDLE"
current_game_id = None
current_fen_csv_path = None
fen_data_file = None
fen_data_reader = None
last_file_size = 0
published_fens = set()
streaming_lock = threading.Lock()
published_fens_file = "data/published_fens.txt"

# Message queue for when network is down
message_queue = []
queue_lock = threading.Lock()
mqtt_connected = False

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"logs/{DEVICE_ID}.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(DEVICE_ID)

# --- MQTT Client ---
mqtt_client_instance = None

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc, properties):
    global mqtt_connected
    if rc == 0:
        logger.info(f"Connected to MQTT Broker: {MQTT_BROKER}")
        mqtt_connected = True
        client.subscribe(f"chess/cmd/{DEVICE_ID}")
        client.subscribe(f"chess/cmd/all")
        
        # Set Last Will Testament - will be sent if connection drops unexpectedly
        client.will_set(
            f"chess/{DEVICE_ID}/status", 
            json.dumps({"state": "DISCONNECTED", "reason": "LWT", "timestamp": int(time.time())}), 
            qos=1, 
            retain=False
        )
        
        publish_status("IDLE")
        
        # Flush queued messages
        flush_message_queue()
    else:
        logger.error(f"Failed to connect, return code {rc}")
        mqtt_connected = False

def on_disconnect(client, userdata, rc, properties=None):
    global mqtt_connected
    mqtt_connected = False
    if rc != 0:
        logger.warning(f"Unexpected disconnect from MQTT broker (rc={rc}). Will attempt reconnection...")
    else:
        logger.info("Disconnected from MQTT broker")

def publish_status(state, **kwargs):
    status_payload = {"state": state, "timestamp": int(time.time()), **kwargs}
    message = json.dumps(status_payload)
    
    if mqtt_connected:
        try:
            result = mqtt_client_instance.publish(f"chess/{DEVICE_ID}/status", message, qos=1)
            if result.rc == 0:
                logger.debug(f"Published status: {state}")
            else:
                logger.warning(f"Failed to publish status, queuing message")
                queue_message(f"chess/{DEVICE_ID}/status", message, qos=1)
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
            queue_message(f"chess/{DEVICE_ID}/status", message, qos=1)
    else:
        logger.warning(f"MQTT not connected, queuing status message")
        queue_message(f"chess/{DEVICE_ID}/status", message, qos=1)

def publish_fen(fen, game_id):
    fen_payload = {"timestamp": int(time.time()), "fen": fen, "game_id": game_id}
    message = json.dumps(fen_payload)
    
    if mqtt_connected:
        try:
            result = mqtt_client_instance.publish(f"chess/{DEVICE_ID}/fen", message, qos=0)
            if result.rc == 0:
                logger.info(f"Published FEN: {fen}")
                # Save to persistent storage
                save_published_fen(fen)
            else:
                logger.warning(f"Failed to publish FEN, queuing message")
                queue_message(f"chess/{DEVICE_ID}/fen", message, qos=0)
        except Exception as e:
            logger.error(f"Error publishing FEN: {e}")
            queue_message(f"chess/{DEVICE_ID}/fen", message, qos=0)
    else:
        logger.warning(f"MQTT not connected, queuing FEN message")
        queue_message(f"chess/{DEVICE_ID}/fen", message, qos=0)

def save_published_fen(fen):
    """Save published FEN to persistent storage"""
    try:
        os.makedirs("data", exist_ok=True)
        with open(published_fens_file, "a") as f:
            f.write(f"{fen}\n")
    except Exception as e:
        logger.error(f"Error saving published FEN to file: {e}")

def queue_message(topic, message, qos=0):
    """Queue message when MQTT is disconnected"""
    with queue_lock:
        message_queue.append({"topic": topic, "message": message, "qos": qos, "timestamp": time.time()})
        logger.info(f"Queued message (queue size: {len(message_queue)})")

def flush_message_queue():
    """Send all queued messages when connection is restored"""
    global message_queue
    with queue_lock:
        if len(message_queue) > 0:
            logger.info(f"Flushing {len(message_queue)} queued messages...")
            for msg in message_queue:
                try:
                    mqtt_client_instance.publish(msg["topic"], msg["message"], qos=msg["qos"])
                    logger.debug(f"Flushed message to {msg['topic']}")
                except Exception as e:
                    logger.error(f"Error flushing message: {e}")
            message_queue = []
            logger.info("Message queue flushed")

def load_published_fens():
    """Load previously published FENs from persistent storage"""
    if os.path.exists(published_fens_file):
        try:
            with open(published_fens_file, "r") as f:
                fens = set(line.strip() for line in f if line.strip())
            logger.info(f"Loaded {len(fens)} previously published FENs from storage")
            return fens
        except Exception as e:
            logger.error(f"Error loading published FENs: {e}")
            return set()
    return set()

def on_message(client, userdata, msg):
    global node_state, current_game_id, current_fen_csv_path, fen_data_file, fen_data_reader, last_file_size, published_fens
    topic = msg.topic
    try:
        payload = json.loads(msg.payload.decode())
        logger.info(f"Received MQTT command on {topic}: {payload}")

        action = payload.get("action")

        if action == "create_game":
            game_number = payload.get("game_number")
            if game_number is not None:
                folder_name = f"game{game_number}"
                folder_path = os.path.join("games", folder_name)
                os.makedirs(folder_path, exist_ok=True)
                csv_path = os.path.join(folder_path, "game.csv")
                # Create empty CSV with header
                with open(csv_path, "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["timestamp", "fen"])
                logger.info(f"Created folder and CSV for {folder_name} at {csv_path}")
            else:
                logger.error("create_game command missing game_number")

        elif action == "start_recording":
            requested_game_id = payload.get("game_id", f"game_{DEVICE_ID}_{int(time.time())}")
            fen_csv_path_from_cmd = payload.get("fen_csv_path")

            if not fen_csv_path_from_cmd or not os.path.exists(fen_csv_path_from_cmd):
                logger.error(f"FEN CSV not found or path not provided in command: {fen_csv_path_from_cmd}")
                publish_status("IDLE", error="CSV_NOT_FOUND")
                return
            
            with streaming_lock:
                if node_state == "IDLE":
                    current_game_id = requested_game_id
                    current_fen_csv_path = fen_csv_path_from_cmd
                    
                    # Load previously published FENs from persistent storage
                    published_fens.clear()
                    published_fens.update(load_published_fens())

                    try:
                        # Check file exists and is readable
                        if not os.path.exists(current_fen_csv_path):
                            raise FileNotFoundError(f"File not found: {current_fen_csv_path}")
                        
                        # Count unpublished FENs
                        unpublished_count = 0
                        with open(current_fen_csv_path, "r", newline='') as f:
                            reader = csv.reader(f)
                            next(reader, None)  # Skip header
                            
                            for row in reader:
                                if len(row) >= 2:
                                    fen = row[1]
                                    if fen not in published_fens:
                                        unpublished_count += 1
                        
                        logger.info(f"Found {unpublished_count} unpublished FENs in {current_fen_csv_path}")
                        
                        # Get initial file size
                        last_file_size = os.path.getsize(current_fen_csv_path)
                        
                        node_state = "STREAMING"
                        publish_status("STREAMING", game_id=current_game_id)
                        logger.info(f"Started streaming from {current_fen_csv_path} for game: {current_game_id}")
                    except Exception as e:
                        logger.error(f"Error setting up streaming for {current_fen_csv_path}: {e}")
                        node_state = "IDLE"
                        current_fen_csv_path = None
                        current_game_id = None
                        publish_status("IDLE", reason=f"FILE_ERROR:{e}")
                else:
                    logger.warning(f"Received start_recording command but node is not IDLE (current state: {node_state})")

        elif action == "stop":
            with streaming_lock:
                if node_state == "STREAMING":
                    node_state = "IDLE"
                    publish_status("IDLE")
                    logger.info(f"Stopped streaming for game: {current_game_id}")
                    current_game_id = None
                    current_fen_csv_path = None
                    last_file_size = 0
                    # Don't clear published_fens here if you want persistence
                else:
                    logger.warning(f"Received stop command but node is not STREAMING (current state: {node_state})")

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON received on {topic}: {msg.payload}")
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}", exc_info=True)

# --- FEN Streaming Thread ---

def fen_streaming_thread():
    logger.info("FEN Streaming Thread started.")
    
    global last_file_size, published_fens, node_state, current_fen_csv_path, current_game_id
    
    last_processed_path = None  # Track which file we're currently processing
    
    while True:
        try:
            # Use lock to safely read state variables
            with streaming_lock:
                is_streaming = node_state == "STREAMING"
                csv_path = current_fen_csv_path
                game_id = current_game_id
            
            # Only stream if in STREAMING state and we have a valid path
            if is_streaming and csv_path and os.path.exists(csv_path):
                current_file_size = os.path.getsize(csv_path)
                
                # Check if this is a new file or first check for current file
                is_new_file = csv_path != last_processed_path
                file_changed = current_file_size != last_file_size
                
                if is_new_file or file_changed:
                    if is_new_file:
                        logger.info(f"New file detected - publishing all unpublished FENs from {csv_path}")
                        last_processed_path = csv_path
                        with streaming_lock:
                            last_file_size = 0  # Reset file size for new file
                    else:
                        logger.info(f"File size changed: {last_file_size} -> {current_file_size}")
                    
                    # Read the entire file to get all FENs
                    try:
                        with open(csv_path, "r", newline='') as f:
                            reader = csv.reader(f)
                            # header = next(reader, None)  # Skip header - REMOVE THIS LINE
                            
                            # Try to detect if first row is header
                            first_row = next(reader, None)
                            if first_row and first_row[0] == "timestamp":
                                # It's a header, skip it
                                logger.debug("CSV has header, skipping it")
                            else:
                                # No header, process this row as data
                                if first_row and len(first_row) >= 1:
                                    # Assume single column = FEN only, no timestamp
                                    fen = first_row[0] if len(first_row) == 1 else first_row[1]
                                    with streaming_lock:
                                        if fen not in published_fens and node_state == "STREAMING":
                                            published_fens.add(fen)
                                            publish_fen(fen, game_id)
                                            fens_published_this_round += 1
                            
                            fens_published_this_round = 0
                            for row in reader:
                                if len(row) >= 1:  # Changed from >= 2
                                    # Handle both formats: "timestamp,fen" or just "fen"
                                    fen = row[0] if len(row) == 1 else row[1]
                                    
                                    # Only publish if we haven't seen this FEN yet
                                    with streaming_lock:
                                        if fen not in published_fens and node_state == "STREAMING":
                                            published_fens.add(fen)
                                            publish_fen(fen, game_id)
                                            fens_published_this_round += 1                            
                    except Exception as e:
                        logger.error(f"Error reading CSV during streaming: {e}")
            else:
                # Not streaming, reset tracking
                if last_processed_path is not None:
                    logger.debug("Stopped streaming, resetting file tracking")
                    last_processed_path = None
                    with streaming_lock:
                        last_file_size = 0
            
            time.sleep(0.5)  # Check every 0.5 seconds for changes
            
        except Exception as e:
            logger.error(f"Error in FEN streaming thread: {e}", exc_info=True)
            time.sleep(1)

# --- Main Entry Point ---
if __name__ == "__main__":
    mqtt_client_instance = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, client_id=DEVICE_ID)
    mqtt_client_instance.on_connect = on_connect
    mqtt_client_instance.on_disconnect = on_disconnect
    mqtt_client_instance.on_message = on_message
    
    # Enable automatic reconnection
    mqtt_client_instance.reconnect_delay_set(min_delay=1, max_delay=120)
    
    try:
        mqtt_client_instance.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    except Exception as e:
        logger.error(f"Initial connection failed: {e}")
        mqtt_connected = False
    
    mqtt_client_instance.loop_start()

    # Start FEN streaming thread
    streaming_thread = threading.Thread(target=fen_streaming_thread, daemon=True)
    streaming_thread.start()

    logger.info(f"Pi node {DEVICE_ID} started. Waiting for commands...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Pi node...")
        # Send graceful shutdown status
        if mqtt_connected:
            publish_status("OFFLINE", reason="Shutdown")
            time.sleep(1)  # Give time for message to send
        mqtt_client_instance.loop_stop()
        mqtt_client_instance.disconnect()