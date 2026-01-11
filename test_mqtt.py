#!/usr/bin/env python3
"""
Simple MQTT test subscriber to verify messages are being published
Run this in a separate terminal to see if MQTT messages are flowing
"""

import paho.mqtt.client as mqtt_client
import json

MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883

def on_connect(client, userdata, flags, rc, properties):
    print(f"Connected to MQTT Broker with result code {rc}")
    client.subscribe("chess/#")  # Subscribe to ALL chess topics
    print("Subscribed to chess/#")

def on_message(client, userdata, msg):
    print(f"\n=== MQTT Message Received ===")
    print(f"Topic: {msg.topic}")
    print(f"QoS: {msg.qos}")
    print(f"Payload: {msg.payload.decode()}")
    try:
        payload_json = json.loads(msg.payload.decode())
        print(f"Parsed JSON: {json.dumps(payload_json, indent=2)}")
    except:
        pass
    print("=" * 30)

if __name__ == "__main__":
    client = mqtt_client.Client(
        mqtt_client.CallbackAPIVersion.VERSION2, 
        client_id="test_subscriber"
    )
    client.on_connect = on_connect
    client.on_message = on_message
    
    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    
    print("Starting MQTT loop... Press Ctrl+C to exit")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nDisconnecting...")
        client.disconnect()
