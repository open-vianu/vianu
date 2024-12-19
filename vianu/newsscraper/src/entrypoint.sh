#!/usr/bin/env bash

set -e

echo "Initializing database..."
/app/venv/bin/python db_setup.py

# echo "Starting cron daemon as root..."
# service cron reload
# service cron restart

echo "Switching back to seluser..."
/app/venv/bin/python main.py

