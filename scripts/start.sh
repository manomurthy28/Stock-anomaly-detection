#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
cd scripts
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
