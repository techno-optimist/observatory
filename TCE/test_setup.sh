#!/bin/bash
#
# TCE Setup Tester
# Run this to verify everything is set up correctly
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "TCE Setup Test"
echo "=============="
echo ""

# Check backend
echo "1. Checking backend..."
cd backend
if [ -d "venv" ]; then
    echo "   ✓ Python venv exists"
    source venv/bin/activate

    # Test imports
    python -c "from app.services.element_service import ElementService; print(f'   ✓ ElementService loaded: {len(ElementService.get_all())} elements')" 2>&1
    python -c "from app.services.compound_service import CompoundService; print(f'   ✓ CompoundService loaded: {len(CompoundService.get_presets())} presets')" 2>&1
else
    echo "   ✗ Python venv missing - run: python3 -m venv venv && source venv/bin/activate && pip install fastapi uvicorn pydantic"
fi

# Check frontend
echo ""
echo "2. Checking frontend..."
cd "$DIR/frontend"
if [ -d "node_modules" ]; then
    echo "   ✓ node_modules exists"
else
    echo "   ✗ node_modules missing - run: npm install"
fi

# Check if ports are in use
echo ""
echo "3. Checking ports..."
if lsof -i :8000 >/dev/null 2>&1; then
    echo "   ✓ Port 8000 is in use (backend running)"
else
    echo "   ✗ Port 8000 is free (backend not running)"
fi

if lsof -i :5173 >/dev/null 2>&1; then
    echo "   ✓ Port 5173 is in use (frontend running)"
else
    echo "   ✗ Port 5173 is free (frontend not running)"
fi

echo ""
echo "To start TCE:"
echo "  Option 1: Double-click TCE.command"
echo "  Option 2: In two terminals:"
echo "    Terminal 1: cd backend && source venv/bin/activate && python -m uvicorn app.main:app --port 8000"
echo "    Terminal 2: cd frontend && npm run dev"
echo ""
