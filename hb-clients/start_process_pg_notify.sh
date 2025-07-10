#!/bin/bash

if ! tmux has-session -t process_pg_notify 2>/dev/null; then
  # Start an interactive bash that runs your script, then drops to a shell
  tmux new-session -d -s process_pg_notify \
    "bash -lc 'python /usr/local/bin/process_pg_notify.py; exec bash'"
fi
