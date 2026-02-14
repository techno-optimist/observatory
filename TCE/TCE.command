#!/bin/bash
#
# TCE - Table of Cognitive Elements
# One-click launcher for macOS
#
# Double-click this file to start both backend and frontend
#

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                          ║${NC}"
echo -e "${CYAN}║      ${GREEN}⚛️  Table of Cognitive Elements v3.0  ⚛️${CYAN}            ║${NC}"
echo -e "${CYAN}║                                                          ║${NC}"
echo -e "${CYAN}║          Cognitive Element Engineering Platform          ║${NC}"
echo -e "${CYAN}║                                                          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check for required tools
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed.${NC}"
        echo -e "Please install $1 first: $2"
        return 1
    fi
    return 0
}

echo -e "${YELLOW}Checking dependencies...${NC}"

# Check Python
if ! check_dependency "python3" "brew install python3"; then
    read -p "Press Enter to exit..."
    exit 1
fi

# Check Node
if ! check_dependency "node" "brew install node"; then
    read -p "Press Enter to exit..."
    exit 1
fi

# Check npm
if ! check_dependency "npm" "brew install node"; then
    read -p "Press Enter to exit..."
    exit 1
fi

echo -e "${GREEN}✓ All dependencies found${NC}"
echo ""

# Install backend dependencies if needed
echo -e "${YELLOW}Setting up backend...${NC}"
cd "$DIR/backend"

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q fastapi uvicorn pydantic sqlalchemy aiosqlite websockets 2>/dev/null

echo -e "${GREEN}✓ Backend ready${NC}"

# Install frontend dependencies if needed
echo -e "${YELLOW}Setting up frontend...${NC}"
cd "$DIR/frontend"

# Check if node_modules needs refresh (esbuild version mismatch fix)
if [ -d "node_modules" ]; then
    # Test if vite can start - if not, we need to reinstall
    if ! npm run build --dry-run &>/dev/null; then
        echo -e "${YELLOW}Detected dependency issues, reinstalling...${NC}"
        rm -rf node_modules package-lock.json 2>/dev/null
    fi
fi

if [ ! -d "node_modules" ]; then
    echo "Installing npm packages (this may take a minute)..."
    npm install 2>/dev/null
fi

echo -e "${GREEN}✓ Frontend ready${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down TCE...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Goodbye!${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${CYAN}Starting backend server...${NC}"
cd "$DIR/backend"
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend
echo -e "${CYAN}Starting frontend dev server...${NC}"
cd "$DIR/frontend"
npm run dev -- --host &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    TCE is running!                       ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}║   Frontend:  ${CYAN}http://localhost:5173${GREEN}                      ║${NC}"
echo -e "${GREEN}║   Backend:   ${CYAN}http://localhost:8001${GREEN}                      ║${NC}"
echo -e "${GREEN}║   API Docs:  ${CYAN}http://localhost:8001/docs${GREEN}                 ║${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}║   Press ${YELLOW}Ctrl+C${GREEN} to stop all servers                      ║${NC}"
echo -e "${GREEN}║                                                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Open browser
sleep 1
open "http://localhost:5173" 2>/dev/null || xdg-open "http://localhost:5173" 2>/dev/null

# Keep script running
wait
