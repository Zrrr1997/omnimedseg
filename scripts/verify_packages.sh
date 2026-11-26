#!/bin/bash
# Setup environment for benchmark tools
# Usage: source setup_path.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"

# Add local tool paths
export PATH=$PATH:$LIB_DIR/unrar/
export PATH=$PATH:$LIB_DIR/
export PATH=$PATH:$LIB_DIR/plastimatch/build/
export PATH=$PATH:$LIB_DIR/ITK/build/bin/

echo "Environment set up for OmniMedSeg tools"
echo ""
echo "Available tools:"
echo "  plastimatch: $(which plastimatch 2>/dev/null || echo 'not found')"
echo "  7z:         $(which 7z 2>/dev/null || echo 'not found')"
echo "  7zz:        $(which 7zz 2>/dev/null || echo 'not found')"
echo "  unrar:      $(which unrar 2>/dev/null || echo 'not found')"
echo "  isic:       $(which isic 2>/dev/null || echo 'not found')"
echo ""
echo "Python dependencies:"
echo "   If not already installed, run:"
echo "   pip install -r requirements.txt"
echo ""
echo "If system tools are not found, run: ./scripts/install_tools.sh"
