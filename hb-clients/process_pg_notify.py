import psycopg2
import select
import socket
import sys
import json
import requests
import threading
from datetime import datetime
import time
import queue
import base64

# Network configuration for connecting to the Node.js server.
# The script first attempts to connect on the local network. If that fails, it uses a fallback host.
LAN_SERVER_IP = "192.168.4.228"
FALLBACK_SERVER_HOST = "hb-server"
NODE_PORT = 3030

# When using the fallback host, we may need to specify the client's address manually.
# This is for cases where the client is on a different network (e.g., Tailscale)
# and its local IP is not reachable from the server. The server will use this address
# to communicate back to the client.
FALLBACK_CLIENT_ADDRESS_OVERRIDE = ""

# Friendly name for this client (sent on registration)
CLIENT_NAME = socket.gethostname()


def _select_node_host():
    for host in (LAN_SERVER_IP, FALLBACK_SERVER_HOST):
        try:
            socket.create_connection((host, NODE_PORT), timeout=1).close()
            print(f"[{datetime.now()}] Selected Node server host: {host}")
            return host
        except Exception:
            print(f"[{datetime.now()}] Could not connect to Node server host: {host}")
    print(f"[{datetime.now()}] Failed to connect to Node server at both preferred and fallback hosts.")
    sys.exit(1)


SELECTED_NODE_HOST = _select_node_host()
# We'll use the address override if we're on the fallback host and an override is set.
USE_OVERRIDE = (SELECTED_NODE_HOST == FALLBACK_SERVER_HOST and bool(FALLBACK_CLIENT_ADDRESS_OVERRIDE))
NODE_BASE_URL = f"http://{SELECTED_NODE_HOST}:{NODE_PORT}"


def register_client():
    """
    Inform the Node server of our presence: send CLIENT_NAME + our address.
    If on LAN: address = our local interface IP.
    If on fallback: address = FALLBACK_CLIENT_ADDRESS_OVERRIDE.
    """
    if USE_OVERRIDE:
        address = FALLBACK_CLIENT_ADDRESS_OVERRIDE
    else:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((SELECTED_NODE_HOST, NODE_PORT))
            address = s.getsockname()[0]
            s.close()
        except Exception:
            address = ""
    payload = {
        "friendlyName": CLIENT_NAME,
        "address": address
    }
    url = f"{NODE_BASE_URL}/register_client"
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            print(f"[{datetime.now()}] Registered client '{CLIENT_NAME}' with address '{address}'")
        else:
            print(f"[{datetime.now()}] Client registration failed {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[{datetime.now()}] Error registering client: {e}")


# Globals for debouncing `recent_stats`
latest_recent_stats_payload = None
recent_stats_debounce_lock = threading.Lock()
recent_stats_debounce_timer = None

# Debounce duration in seconds
DEBOUNCE_DURATION = 0.02

# Queue for inference outbox processing
inference_outbox_queue = queue.Queue()


def process_trial_outbox():
    conn = None
    BATCH_SIZE = 10
    try:
        conn = psycopg2.connect(dbname="base", user="postgres", password="postgres", host="localhost")
        conn.autocommit = True

        while True:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM outbox_trial ORDER BY base_trial_id ASC LIMIT %s;",
                    (BATCH_SIZE,)
                )
                rows = cur.fetchall()

            if not rows:
                break

            print(f"[{datetime.now()}] Processing batch of {len(rows)} trials from outbox_trial...")
            columns = [desc[0] for desc in cur.description]
            trials = []
            for row in rows:
                trial = {
                    col: (val.isoformat() if isinstance(val, datetime) else val)
                    for col, val in zip(columns, row)
                }
                if USE_OVERRIDE and "host" in trial:
                    trial["host"] = FALLBACK_CLIENT_ADDRESS_OVERRIDE
                trials.append(trial)

            node_url = f"{NODE_BASE_URL}/process_outbox_trial"
            response = requests.post(node_url, json={"rows": trials}, timeout=10)

            if response.status_code == 200:
                processed_rows = response.json().get("processedRows", [])
                if processed_rows:
                    print(f"[{datetime.now()}] Server processed {len(processed_rows)} trials.")
                    with conn.cursor() as cur:
                        trial_ids = [r["trial_id"] for r in processed_rows]
                        cur.execute(
                            "DELETE FROM outbox_trial WHERE trial_id = ANY(%s);",
                            (trial_ids,)
                        )
                        print(f"[{datetime.now()}] Deleted {cur.rowcount} rows from outbox_trial.")
                else:
                    print(f"[{datetime.now()}] No trial rows processed for this batch.")
            else:
                print(f"[{datetime.now()}] Node server error {response.status_code}: {response.text}")
                break

            if len(rows) < BATCH_SIZE:
                break

    except Exception as e:
        print(f"[{datetime.now()}] process_trial_outbox error: {e}")
    finally:
        if conn:
            conn.close()


def process_inference_outbox():
    conn = None
    BATCH_SIZE = 50
    total_processed = 0
    start_time = datetime.now()

    try:
        conn = psycopg2.connect(dbname="base", user="postgres", password="postgres", host="localhost")
        conn.autocommit = True

        while True:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM outbox_inference ORDER BY client_time ASC LIMIT %s;",
                    (BATCH_SIZE,)
                )
                rows = cur.fetchall()

            if not rows:
                print(f"[{datetime.now()}] No more rows to process in outbox_inference.")
                break

            print(f"[{datetime.now()}] Processing batch of {len(rows)} rows...")
            columns = [desc[0] for desc in cur.description]
            inferences = []
            for row in rows:
                inf = {}
                for col, val in zip(columns, row):
                    if isinstance(val, datetime):
                        inf[col] = val.isoformat()
                    elif isinstance(val, memoryview):
                        inf[col] = base64.b64encode(bytes(val)).decode('utf-8')
                    else:
                        inf[col] = val
                if USE_OVERRIDE and "host" in inf:
                    inf["host"] = FALLBACK_CLIENT_ADDRESS_OVERRIDE
                inferences.append(inf)

            node_url = f"{NODE_BASE_URL}/process_outbox_inference"
            print(f"[{datetime.now()}] Sending batch to Node server...")
            response = requests.post(node_url, json={"rows": inferences}, timeout=20)

            if response.status_code == 200:
                processed = response.json().get("processedRows", [])
                print(f"[{datetime.now()}] Server processed {len(processed)} rows successfully.")
                if processed:
                    ids = [r["infer_id"] for r in processed]
                    with conn.cursor() as cur:
                        cur.execute(
                            "DELETE FROM outbox_inference WHERE infer_id = ANY(%s);",
                            (ids,)
                        )
                        total_processed += len(processed)
                        print(f"[{datetime.now()}] Deleted {len(processed)} rows. Total: {total_processed}")
            else:
                print(f"[{datetime.now()}] Node server error {response.status_code}: {response.text}")
                break

            if len(rows) < BATCH_SIZE:
                break

        duration = (datetime.now() - start_time).total_seconds()
        print(f"[{datetime.now()}] Finished inference cycle: {total_processed} rows in {duration:.2f}s")

    except Exception as e:
        print(f"[{datetime.now()}] process_inference_outbox error: {e}")
    finally:
        if conn:
            conn.close()


def send_entire_status_to_node():
    conn = psycopg2.connect(dbname="base", user="postgres", password="postgres", host="localhost")
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM status WHERE status_type != 'system_script';")
            rows = cur.fetchall()
            if not rows:
                print("Status table empty, skipping...")
                return
            columns = [desc[0] for desc in cur.description]

        statuses = []
        for row in rows:
            s = {
                col: (val.isoformat() if isinstance(val, datetime) else val)
                for col, val in zip(columns, row)
            }
            if USE_OVERRIDE and "host" in s:
                s["host"] = FALLBACK_CLIENT_ADDRESS_OVERRIDE
            statuses.append(s)

        node_url = f"{NODE_BASE_URL}/upsert_status"
        response = requests.post(node_url, json={"rows": statuses}, timeout=5)
        if response.status_code != 200:
            print(f"Node server error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[{datetime.now()}] send_entire_status_to_node error: {e}")
    finally:
        conn.close()


def send_status_to_node(payload):
    if not payload:
        return
    try:
        status_data = json.loads(payload)
        if USE_OVERRIDE and isinstance(status_data, dict) and "host" in status_data:
            status_data["host"] = FALLBACK_CLIENT_ADDRESS_OVERRIDE

        node_url = f"{NODE_BASE_URL}/upsert_status"
        response = requests.post(node_url, json={"rows": [status_data]}, timeout=5)
        if response.status_code != 200:
            print(f"Node server error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[{datetime.now()}] send_status_to_node error: {e}")


def send_recent_stats_to_node():
    global latest_recent_stats_payload
    with recent_stats_debounce_lock:
        payload = latest_recent_stats_payload
        latest_recent_stats_payload = None

    if not payload:
        return
    try:
        recent_stats_data = json.loads(payload)
        rows = recent_stats_data if isinstance(recent_stats_data, list) else [recent_stats_data]
        for rec in rows:
            if USE_OVERRIDE and "host" in rec:
                rec["host"] = FALLBACK_CLIENT_ADDRESS_OVERRIDE

        node_url = f"{NODE_BASE_URL}/upsert_recent_stats"
        response = requests.post(node_url, json={"rows": rows}, timeout=5)
        if response.status_code != 200:
            print(f"Node server error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[{datetime.now()}] send_recent_stats_to_node error: {e}")


def update_recent_stats(conn, payload):
    global latest_recent_stats_payload, recent_stats_debounce_timer
    with recent_stats_debounce_lock:
        latest_recent_stats_payload = payload
    if recent_stats_debounce_timer:
        recent_stats_debounce_timer.cancel()
    recent_stats_debounce_timer = threading.Timer(DEBOUNCE_DURATION, send_recent_stats_to_node)
    recent_stats_debounce_timer.start()


def periodic_status_sync():
    while True:
        try:
            send_entire_status_to_node()
        except Exception as e:
            print(f"[{datetime.now()}] periodic_status_sync error: {e}")
        time.sleep(60)


def handle_image_notification(image_type: str):
    print(f"[{datetime.now()}] Processing '{image_type}' notification.")
    conn = None
    try:
        conn = psycopg2.connect(dbname="base", user="postgres", password="postgres", host="localhost")
        conn.autocommit = True

        with conn.cursor() as cur:
            cur.execute(
                "SELECT host, status_source, status_type, status_value "
                "FROM status WHERE status_type = %s LIMIT 1;",
                (image_type,)
            )
            row = cur.fetchone()

        if not row:
            print(f"No '{image_type}' entry found.")
            return

        columns = [desc[0] for desc in cur.description]
        image_data = {
            col: (val.isoformat() if isinstance(val, datetime) else val)
            for col, val in zip(columns, row)
        }
        if USE_OVERRIDE and "host" in image_data:
            image_data["host"] = FALLBACK_CLIENT_ADDRESS_OVERRIDE

        node_url = f"{NODE_BASE_URL}/upsert_status"
        response = requests.post(node_url, json={"rows": [image_data]}, timeout=10)
        if response.status_code != 200:
            print(f"Node server error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"[{datetime.now()}] handle_image_notification error: {e}")
    finally:
        if conn:
            conn.close()


def inference_outbox_worker():
    consecutive_errors = 0
    while True:
        try:
            inference_outbox_queue.get()
            drained = 0
            while not inference_outbox_queue.empty():
                inference_outbox_queue.get_nowait()
                drained += 1
            process_inference_outbox()
            consecutive_errors = 0
            if drained:
                continue
            time.sleep(0.1)
        except Exception as e:
            consecutive_errors += 1
            backoff = min(30, 2 ** consecutive_errors)
            print(f"[{datetime.now()}] inference_outbox_worker error: {e}, backing off {backoff}s")
            time.sleep(backoff)


def listen():
    conn = psycopg2.connect(dbname="base", user="postgres", password="postgres", host="localhost")
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    for ch in ("empty_outbox_trial", "empty_outbox_inference",
               "copy_status", "copy_status_oversized",
               "copy_recent_stats", "new_image"):
        cur.execute(f"LISTEN {ch};")
    print("Now listening for postgres notifications...")

    try:
        while True:
            if select.select([conn], [], []):
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    print(f"[{datetime.now()}] Notification: {notify.channel}, payload: {notify.payload}")

                    if notify.channel == "empty_outbox_trial":
                        threading.Thread(target=process_trial_outbox, daemon=True).start()
                    elif notify.channel == "empty_outbox_inference":
                        if inference_outbox_queue.qsize() < 100:
                            inference_outbox_queue.put(True)
                        else:
                            print("Inference queue full; skipping.")
                    elif notify.channel == "copy_status":
                        threading.Thread(target=send_status_to_node, args=(notify.payload,), daemon=True).start()
                    elif notify.channel == "new_image":
                        t = notify.payload
                        if t in ('photo_cartoon', 'screenshot'):
                            threading.Thread(target=handle_image_notification, args=(t,), daemon=True).start()
                    elif notify.channel == "copy_recent_stats":
                        threading.Thread(target=update_recent_stats, args=(conn, notify.payload), daemon=True).start()
                    # ignore copy_status_oversized
    except KeyboardInterrupt:
        print("\nTerminating listener.")
    finally:
        cur.close()
        conn.close()
        print("Connection closed.")


if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting process_pg_notify.py...")
    register_client()
    print(f"[{datetime.now()}] Initial inference outbox processing...")
    process_inference_outbox()
    print("Initial outbox processing complete.")
    threading.Thread(target=periodic_status_sync, daemon=True).start()
    threading.Thread(target=inference_outbox_worker, daemon=True).start()
    print(f"[{datetime.now()}] Starting notification listener...")
    listen()
