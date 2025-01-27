#!/usr/bin/env bash

set -e

echo "Initializing database..."
/app/venv/bin/python db_setup.py

echo "Switching back to seluser..."
/app/venv/bin/python main.py

