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

# Server configuration
PREFERRED_SERVER_IP = "192.168.4.228"
FALLBACK_SERVER_HOST = "hb-server"
NODE_PORT = 3030

def _select_node_host():
    for host in (PREFERRED_SERVER_IP, FALLBACK_SERVER_HOST):
        try:
            socket.create_connection((host, NODE_PORT), timeout=1).close()
            print(f"[{datetime.now()}] Selected Node server host: {host}")
            return host
        except Exception:
            print(f"[{datetime.now()}] Could not connect to Node server host: {host}")
    print(f"[{datetime.now()}] Failed to connect to Node server at both preferred and fallback hosts.")
    sys.exit(1)

SELECTED_NODE_HOST = _select_node_host()
NODE_BASE_URL = f"http://{SELECTED_NODE_HOST}:{NODE_PORT}"

# Globals for debouncing `recent_stats`
latest_recent_stats_payload = None
recent_stats_debounce_lock = threading.Lock()
recent_stats_debounce_timer = None

# Debounce duration in seconds
DEBOUNCE_DURATION = 0.02

# Queue for inference outbox processing
inference_outbox_queue = queue.Queue()

def process_trial_outbox():
    """
    Fetch rows from the outbox_trial table in batches, send them to the Node server,
    and delete rows upon successful processing. Continues fetching batches until the
    outbox is empty.
    """
    conn = None
    BATCH_SIZE = 10

    try:
        conn = psycopg2.connect(
            dbname="base",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        conn.autocommit = True

        while True:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM outbox_trial ORDER BY base_trial_id ASC LIMIT %s;", (BATCH_SIZE,))
                rows = cur.fetchall()

            if not rows:
                break

            print(f"[{datetime.now()}] Processing batch of {len(rows)} trials from outbox_trial...")

            columns = [desc[0] for desc in cur.description]
            trials = [
                {col: (value.isoformat() if isinstance(value, datetime) else value)
                 for col, value in zip(columns, row)}
                for row in rows
            ]

            node_url = f"{NODE_BASE_URL}/process_outbox_trial"
            response = requests.post(node_url, json={"rows": trials}, timeout=10)

            if response.status_code == 200:
                response_data = response.json()
                processed_rows = response_data.get("processedRows", [])

                if processed_rows:
                    print(f"[{datetime.now()}] Server processed {len(processed_rows)} trials.")
                    with conn.cursor() as cur:
                        trial_ids = [row["trial_id"] for row in processed_rows]
                        cur.execute(
                            "DELETE FROM outbox_trial WHERE trial_id = ANY(%s);",
                            (trial_ids,)
                        )
                        print(f"[{datetime.now()}] Deleted {cur.rowcount} rows from outbox_trial.")
                else:
                    print(f"[{datetime.now()}] Server reported no trial rows were processed for this batch.")
            else:
                print(f"[{datetime.now()}] Node server returned error code {response.status_code}: {response.text}")
                break

            if len(rows) < BATCH_SIZE:
                break

    except psycopg2.Error as e:
        print(f"[{datetime.now()}] Database error in process_trial_outbox: {e}")
    except requests.RequestException as e:
        print(f"[{datetime.now()}] Error sending data to Node server in process_trial_outbox: {e}")
    except Exception as e:
        print(f"[{datetime.now()}] An unexpected error occurred in process_trial_outbox: {e}")
    finally:
        if conn:
            conn.close()


def process_inference_outbox():
    """
    Fetch rows from the outbox_inference table in batches, send them to the Node server,
    and delete rows upon successful processing. Continues fetching batches if the
    previous batch was full.
    """
    conn = None
    BATCH_SIZE = 50
    total_processed = 0
    start_time = datetime.now()

    try:
        conn = psycopg2.connect(
            dbname="base",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        conn.autocommit = True

        while True:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM outbox_inference ORDER BY client_time ASC LIMIT %s;",
                    (BATCH_SIZE,)
                )
                rows = cur.fetchall()

            if not rows:
                print(f"[{datetime.now()}] No more rows to process in outbox_inference for this cycle.")
                break

            print(f"[{datetime.now()}] Processing batch of {len(rows)} rows...")
            
            columns = [desc[0] for desc in cur.description]
            inferences = []
            for row in rows:
                processed_row = {}
                for col, value in zip(columns, row):
                    if isinstance(value, datetime):
                        processed_row[col] = value.isoformat()
                    elif isinstance(value, memoryview):
                        processed_row[col] = base64.b64encode(bytes(value)).decode('utf-8')
                    else:
                        processed_row[col] = value
                inferences.append(processed_row)

            node_url = f"{NODE_BASE_URL}/process_outbox_inference"
            print(f"[{datetime.now()}] Sending batch to Node server...")
            response = requests.post(node_url, json={"rows": inferences}, timeout=20)

            if response.status_code == 200:
                response_data = response.json()
                processed_rows_from_server = response_data.get("processedRows", [])
                print(f"[{datetime.now()}] Server processed {len(processed_rows_from_server)} rows successfully")

                if processed_rows_from_server:
                    with conn.cursor() as cur:
                        infer_ids_to_delete = [row["infer_id"] for row in processed_rows_from_server]
                        if infer_ids_to_delete:
                            cur.execute(
                                "DELETE FROM outbox_inference WHERE infer_id = ANY(%s);",
                                (infer_ids_to_delete,)
                            )
                            total_processed += len(processed_rows_from_server)
                            print(f"[{datetime.now()}] Deleted {len(processed_rows_from_server)} rows. Total processed: {total_processed}")
            else:
                print(f"[{datetime.now()}] Node server returned error code {response.status_code}: {response.text} for a batch. Stopping for this cycle.")
                break

            if len(rows) < BATCH_SIZE:
                print(f"[{datetime.now()}] Processed partial batch, breaking loop.")
                break

        duration = (datetime.now() - start_time).total_seconds()
        print(f"[{datetime.now()}] Finished processing cycle. Processed {total_processed} rows in {duration:.2f} seconds")

    except psycopg2.Error as e:
        print(f"Database error in process_inference_outbox: {e}")
    except requests.RequestException as e:
        print(f"Error sending batch data to Node server in process_inference_outbox: {e}")
    except Exception as e:
        print(f"Unexpected error in process_inference_outbox: {e}")
    finally:
        if conn:
            conn.close()


def send_entire_status_to_node():
    """
    Periodically copies the entire `status` table to the server to ensure synchronization.
    """
    conn = psycopg2.connect(
        dbname="base",
        user="postgres",
        password="postgres",
        host="localhost"
    )
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM status WHERE status_type != 'system_script';")
            rows = cur.fetchall()

            if not rows:
                print("Table `status` is empty, skipping...")
                return

            columns = [desc[0] for desc in cur.description]

        statuses = [
            {col: (value.isoformat() if isinstance(value, datetime) else value)
             for col, value in zip(columns, row)}
            for row in rows
        ]

        node_url = f"{NODE_BASE_URL}/upsert_status"
        response = requests.post(node_url, json={"rows": statuses}, timeout=5)

        if response.status_code == 200:
            print("Status data successfully sent to Node server.")
        else:
            print(f"Node server returned error code {response.status_code}: {response.text}")

    except psycopg2.Error as e:
        print(f"Error querying the database: {e}")
    except requests.RequestException as e:
        print(f"Error sending data to Node server: {e}")
    finally:
        conn.close()


def send_status_to_node(payload):
    """
    Process the most recent payload and send it to the Node server for `status`.
    """
    if payload:
        try:
            status_data = json.loads(payload)

            if status_data:
                node_url = f"{NODE_BASE_URL}/upsert_status"
                response = requests.post(node_url, json={"rows": [status_data]}, timeout=5)

                if response.status_code == 200:
                    print("Status data successfully sent to Node server.")
                else:
                    print(f"Node server returned error code {response.status_code}: {response.text}")
            else:
                print("No status data received. Nothing to send.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON payload: {e}")
        except requests.RequestException as e:
            print(f"Error sending data to Node server: {e}")


def send_recent_stats_to_node():
    """
    Process the most recent payload and send it to the Node server for `recent_stats`.
    """
    global latest_recent_stats_payload
    with recent_stats_debounce_lock:
        payload = latest_recent_stats_payload
        latest_recent_stats_payload = None

    if payload:
        try:
            recent_stats_data = json.loads(payload)

            if recent_stats_data:
                node_url = f"{NODE_BASE_URL}/upsert_recent_stats"
                response = requests.post(node_url, json={"rows": recent_stats_data}, timeout=5)

                if response.status_code == 200:
                    print("Recent stats successfully sent to Node server.")
                else:
                    print(f"Node server returned error code {response.status_code}: {response.text}")
            else:
                print("No recent stats data received. Nothing to send.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON payload: {e}")
        except requests.RequestException as e:
            print(f"Error sending data to Node server: {e}")


def update_recent_stats(conn, payload):
    """
    Handles the `copy_recent_stats` notification. Uses debounce logic to send the most recent payload only.
    """
    print(f"[{datetime.now()}] Received payload for copy_recent_stats. Adding to debounce queue.")
    global latest_recent_stats_payload, recent_stats_debounce_timer
    with recent_stats_debounce_lock:
        latest_recent_stats_payload = payload

    if recent_stats_debounce_timer:
        recent_stats_debounce_timer.cancel()

    recent_stats_debounce_timer = threading.Timer(DEBOUNCE_DURATION, send_recent_stats_to_node)
    recent_stats_debounce_timer.start()


def periodic_status_sync():
    """
    Calls the `send_entire_status_to_node` function once per minute.
    """
    while True:
        try:
            send_entire_status_to_node()
        except Exception as e:
            print(f"Error in periodic status sync: {e}")
        time.sleep(60)


def handle_image_notification(image_type: str):
    """
    Fetches image-related status data from the 'status' table based on image_type
    and sends it to the Node.js server.
    """
    print(f"[{datetime.now()}] Processing '{image_type}' notification.")
    conn = None
    try:
        conn = psycopg2.connect(
            dbname="base",
            user="postgres",
            password="postgres",
            host="localhost"
        )
        conn.autocommit = True

        with conn.cursor() as cur:
            cur.execute(
                "SELECT host, status_source, status_type, status_value FROM status WHERE status_type = %s LIMIT 1;",
                (image_type,)
            )
            row = cur.fetchone()

            if not row:
                print(f"No '{image_type}' entry found in status table.")
                return

            columns = [desc[0] for desc in cur.description]
            image_data = {
                col: (value.isoformat() if isinstance(value, datetime) else value)
                for col, value in zip(columns, row)
            }

        node_url = f"{NODE_BASE_URL}/upsert_status"
        response = requests.post(node_url, json={"rows": [image_data]}, timeout=10)

        if response.status_code == 200:
            print(f"Successfully sent '{image_type}' data to Node server.")
        else:
            print(f"Node server returned error code {response.status_code} for '{image_type}': {response.text}")

    except psycopg2.Error as e:
        print(f"Database error processing '{image_type}': {e}")
    except requests.RequestException as e:
        print(f"Error sending '{image_type}' data to Node server: {e}")
    finally:
        if conn:
            conn.close()


def inference_outbox_worker():
    """
    Worker thread that processes items from the inference_outbox_queue.
    """
    consecutive_errors = 0
    while True:
        try:
            inference_outbox_queue.get()
            print(f"[{datetime.now()}] Inference worker woken up by queue.")

            drained = 0
            while not inference_outbox_queue.empty():
                try:
                    inference_outbox_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            if drained > 0:
                print(f"[{datetime.now()}] Drained {drained} additional notifications from queue.")

            print(f"[{datetime.now()}] Worker calling process_inference_outbox...")
            process_inference_outbox()
            consecutive_errors = 0

            if drained > 0:
                continue

            time.sleep(0.1)

        except psycopg2.Error as db_err:
            consecutive_errors += 1
            print(f"[{datetime.now()}] Database error in inference_outbox_worker: {db_err}")
            sleep_time = min(30, 2 ** consecutive_errors)
            print(f"[{datetime.now()}] Backing off for {sleep_time} seconds...")
            time.sleep(sleep_time)
        except requests.RequestException as req_err:
            consecutive_errors += 1
            print(f"[{datetime.now()}] Request error in inference_outbox_worker: {req_err}")
            sleep_time = min(30, 2 ** consecutive_errors)
            print(f"[{datetime.now()}] Backing off for {sleep_time} seconds...")
            time.sleep(sleep_time)
        except Exception as e:
            consecutive_errors += 1
            print(f"[{datetime.now()}] Unexpected error in inference_outbox_worker: {e}")
            sleep_time = min(30, 2 ** consecutive_errors)
            print(f"[{datetime.now()}] Backing off for {sleep_time} seconds...")
            time.sleep(sleep_time)


def listen():
    conn = psycopg2.connect(
        dbname="base",
        user="postgres",
        password="postgres",
        host="localhost"
    )
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    cur = conn.cursor()
    cur.execute("LISTEN empty_outbox_trial;")
    cur.execute("LISTEN empty_outbox_inference;")
    cur.execute("LISTEN copy_status;")
    cur.execute("LISTEN copy_status_oversized;")
    cur.execute("LISTEN copy_recent_stats;")
    cur.execute("LISTEN new_image;")

    print("Now listening for postgres notifications...")

    try:
        while True:
            if select.select([conn], [], []):
                conn.poll()
                while conn.notifies:
                    notify = conn.notifies.pop(0)
                    print(f"[{datetime.now()}] Received notification: {notify.channel}, payload: {notify.payload}")

                    if notify.channel == "empty_outbox_trial":
                        threading.Thread(target=process_trial_outbox, daemon=True).start()
                    elif notify.channel == "empty_outbox_inference":
                        if inference_outbox_queue.qsize() < 100:
                            inference_outbox_queue.put(True)
                        else:
                            print(f"[{datetime.now()}] Inference outbox queue full. Notification skipped.")
                    elif notify.channel == "copy_status":
                        threading.Thread(target=send_status_to_node, args=(notify.payload,), daemon=True).start()
                    elif notify.channel == "new_image":
                        image_type_from_payload = notify.payload
                        if image_type_from_payload in ('photo_cartoon', 'screenshot'):
                            threading.Thread(target=handle_image_notification, args=(image_type_from_payload,), daemon=True).start()
                        else:
                            print(f"[{datetime.now()}] Received unhandled image_type in new_image notification: {image_type_from_payload}")
                    elif notify.channel == "copy_recent_stats":
                        threading.Thread(target=update_recent_stats, args=(conn, notify.payload), daemon=True).start()
                    elif notify.channel == "copy_status_oversized":
                        print('We got a bigggg message from table: status. ignoring it...')
    except KeyboardInterrupt:
        print("\nTerminating listener.")
    finally:
        cur.close()
        conn.close()
        print("Connection closed.")


if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting process_pg_notify.py...")

    print(f"[{datetime.now()}] Processing any existing rows in outbox_inference...")
    process_inference_outbox()
    print(f"[{datetime.now()}] Initial outbox processing complete.")

    threading.Thread(target=periodic_status_sync, daemon=True).start()
    threading.Thread(target=inference_outbox_worker, daemon=True).start()

    print(f"[{datetime.now()}] Starting notification listener...")
    listen()
