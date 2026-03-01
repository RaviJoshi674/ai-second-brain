#!/bin/bash

# Configuration
export ENDEE_URL="http://127.0.0.1:8080"
export PORT_BACKEND=8000
export PORT_FRONTEND=8501

echo "🧠 AI Second Brain - Starting up..."

# Check if Endee is running
if ! curl -s $ENDEE_URL/api/v1/health > /dev/null; then
    echo "❌ Error: Endee doesn't seem to be running on $ENDEE_URL."
    echo "Please start Endee first:"
    echo "  cd ../ && ./run.sh"
    exit 1
fi

echo "✅ Endee is running!"

# Setup virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies (this might take a minute)..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Define cleanup procedure
cleanup() {
    echo "🛑 Shutting down AI Second Brain..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Trap termination signals to kill background processes
trap cleanup SIGINT SIGTERM

echo "🚀 Starting FastAPI backend on port $PORT_BACKEND..."
uvicorn app.api:app --host 0.0.0.0 --port $PORT_BACKEND --reload &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 3

echo "🎨 Starting Streamlit frontend on port $PORT_FRONTEND..."
streamlit run ui/app.py --server.port $PORT_FRONTEND --server.headless true --browser.gatherUsageStats false &
FRONTEND_PID=$!

echo ""
echo "========================================================"
echo "🌟 AI Second Brain is running!"
echo "📡 Backend API: http://127.0.0.1:$PORT_BACKEND"
echo "🖥️  Frontend UI: http://127.0.0.1:$PORT_FRONTEND"
echo "Press Ctrl+C to stop both servers."
echo "========================================================"
echo ""

# Wait for background processes
wait
