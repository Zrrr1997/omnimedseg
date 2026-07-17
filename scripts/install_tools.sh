#!/bin/bash
# Fast installer for benchmark tools
# Uses apt for packages when available, falls back to building from source

set -e

echo "🔧 Benchmark Tools Fast Installer"
echo "================================"
echo ""

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
CORES=$(nproc)
MAKE_JOBS=$((CORES * 2))
echo "🚀 Using $MAKE_JOBS parallel jobs for compilation"

mkdir -p lib
cd lib

# ----------------------------------------------------------------------------
# Helpers - Force a fresh check
# ----------------------------------------------------------------------------
has_sudo() { command -v sudo &> /dev/null && sudo -n true 2>/dev/null; }
is_debian_based() { [ -f /etc/os-release ] && grep -qiE "debian|ubuntu" /etc/os-release; }

# Reliable command check - clears hash table first
command_exists() {
    hash -r 2>/dev/null || true
    command -v "$1" &> /dev/null
}

# ----------------------------------------------------------------------------
# 1. 7zip
# ----------------------------------------------------------------------------
hash -r 2>/dev/null || true
if ! command_exists 7z && ! command_exists 7zz; then
    echo "📥 Downloading 7zip..."
    LATEST_7Z=$(wget -qO- https://www.7-zip.org/download.html | grep -oP '7z\d{4}-linux-x64\.tar\.xz' | head -1)
    if [ -z "$LATEST_7Z" ]; then
        LATEST_7Z="7z2409-linux-x64.tar.xz"
    fi
    wget -q --show-progress "https://www.7-zip.org/a/$LATEST_7Z"
    tar -xf "$LATEST_7Z"
    chmod +x 7zz 7zzs
    rm "$LATEST_7Z"
    echo "✅ 7zip installed"
else
    echo "⏭️  7zip already installed (skipping)"
fi

# ----------------------------------------------------------------------------
# 2. unrar - ALWAYS INSTALL (force reinstall to be safe)
# ----------------------------------------------------------------------------
echo "📦 Installing unrar..."

# Remove any stale cache
hash -r 2>/dev/null || true

# Try apt if available
UNRAR_INSTALLED=false
if is_debian_based && has_sudo; then
    echo "   Installing via apt..."
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y --reinstall unrar
    hash -r 2>/dev/null || true
    if command_exists unrar; then
        echo "✅ unrar installed via apt"
        UNRAR_INSTALLED=true
    else
        echo "⚠️  apt install failed"
    fi
fi

# If apt failed, build from source
if [ "$UNRAR_INSTALLED" = false ]; then
    echo "   Building from source..."
    rm -rf unrar unrarsrc-7.1.1.tar.gz
    wget -q --show-progress https://www.rarlab.com/rar/unrarsrc-7.1.1.tar.gz
    tar -xf unrarsrc-7.1.1.tar.gz
    cd unrar
    make -j$MAKE_JOBS
    cd ..
    rm -rf unrar.bin
    cp unrar/unrar unrar.bin
    chmod +x unrar.bin
    mv unrar.bin unrar
    rm -rf unrar unrarsrc-7.1.1.tar.gz
    echo "✅ unrar installed in lib/"
fi

# ----------------------------------------------------------------------------
# 3. plastimatch - ALWAYS INSTALL (force reinstall to be safe)
# ----------------------------------------------------------------------------
echo "📦 Installing plastimatch..."

hash -r 2>/dev/null || true

PLASTIMATCH_INSTALLED=false
if is_debian_based && has_sudo; then
    echo "   Installing via apt..."
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y --reinstall plastimatch
    hash -r 2>/dev/null || true
    if command_exists plastimatch; then
        echo "✅ plastimatch installed via apt"
        PLASTIMATCH_INSTALLED=true
    else
        echo "⚠️  apt install failed"
    fi
fi

# If apt failed, build from source
if [ "$PLASTIMATCH_INSTALLED" = false ]; then
    echo "   Building from source..."
    
    # Check if DCMTK is available
    DCMTK_DIR=""
    if command_exists dcmdump && [ -d "/usr/include/dcmtk" ]; then
        echo "✅ Found system DCMTK"
        DCMTK_DIR="/usr/lib/x86_64-linux-gnu/cmake/dcmtk"
    elif [ -d "dcmtk/build" ]; then
        echo "✅ Using local DCMTK build"
        DCMTK_DIR="$(pwd)/dcmtk/build"
    else
        echo "📥 Building DCMTK (minimal)..."
        git clone --depth 1 https://github.com/DCMTK/dcmtk.git
        cd dcmtk
        mkdir -p build && cd build
        cmake -DCMAKE_INSTALL_PREFIX=../.. \
              -DBUILD_SHARED_LIBS=ON \
              -DDCMTK_WITH_DOXYGEN=OFF \
              -DDCMTK_WITH_OPENSSL=OFF \
              -DDCMTK_WITH_PNG=OFF \
              -DDCMTK_WITH_TIFF=OFF \
              -DDCMTK_WITH_XML=OFF \
              -DCMAKE_BUILD_TYPE=Release \
              ..
        make -j$MAKE_JOBS
        cd ../..
        DCMTK_DIR="$(pwd)/dcmtk/build"
    fi
    
    # Check if ITK is available
    ITK_DIR=""
    if [ -d "/usr/lib/x86_64-linux-gnu/cmake/ITK-5.3" ]; then
        echo "✅ Found system ITK"
        ITK_DIR="/usr/lib/x86_64-linux-gnu/cmake/ITK-5.3"
    elif [ -d "ITK/build" ]; then
        echo "✅ Using local ITK build"
        ITK_DIR="$(pwd)/ITK/build"
    else
        echo "📥 Building ITK 4.13.3..."
        git clone --branch v4.13.3 --depth 1 https://github.com/InsightSoftwareConsortium/ITK.git
        cd ITK
        mkdir -p build && cd build
        cmake -DCMAKE_INSTALL_PREFIX=../.. \
            -DBUILD_EXAMPLES=OFF \
            -DBUILD_TESTING=OFF \
            -DModule_ITKReview=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_STANDARD=14 \
            -DCMAKE_CXX_STANDARD_REQUIRED=ON \
            -DCMAKE_CXX_EXTENSIONS=OFF \
            -DITK_USE_SYSTEM_HDF5=OFF \
            -DITK_USE_SYSTEM_ZLIB=OFF \
            ..
        make -j$(nproc)
        cd ../..
        ITK_DIR="$(pwd)/ITK/build"
        echo "✅ ITK 4.13.3 installed"
    fi
    
    # Build plastimatch
    if [ ! -d "plastimatch" ]; then
        echo "📥 Cloning plastimatch..."
        git clone --depth 1 https://gitlab.com/plastimatch/plastimatch.git
    fi
    
    cd plastimatch
    
    # Disable SuperBuild
    echo "🔧 Disabling SuperBuild..."
    sed -i 's/INCLUDE_SUPERBUILD ON/INCLUDE_SUPERBUILD OFF/g' CMakeLists.txt 2>/dev/null || true
    sed -i 's/INCLUDE_SUPERBUILD 1/INCLUDE_SUPERBUILD OFF/g' CMakeLists.txt 2>/dev/null || true
    
    if [ -d "SuperBuild" ]; then
        mv SuperBuild SuperBuild_Disabled
    fi
    
    mkdir -p SuperBuild
    echo "# Disabled SuperBuild" > SuperBuild/CMakeLists.txt
    echo "message(STATUS \"SuperBuild disabled\")" >> SuperBuild/CMakeLists.txt
    
    if [ -d "build" ]; then
        rm -rf build
    fi
    
    mkdir -p build && cd build
    
    echo "🔧 Configuring plastimatch..."
    cmake -DCMAKE_INSTALL_PREFIX=../.. \
        -DITK_DIR="$ITK_DIR" \
        -DDCMTK_DIR="$DCMTK_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPLASTIMATCH_BUILD_CLI=ON \
        -DPLASTIMATCH_BUILD_DOCS=OFF \
        -DPLASTIMATCH_BUILD_TESTS=OFF \
        ..

    echo "🔨 Building plastimatch..."
    make -j$(nproc)
    cd ../..
    echo "✅ plastimatch built"
fi



# ----------------------------------------------------------------------------
# 6. Create setup_path.sh
# ----------------------------------------------------------------------------
cd ..

cat > setup_path.sh << 'EOF'
#!/bin/bash
# Setup script for benchmark tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$SCRIPT_DIR/lib:$PATH"

# Clear command hash table
hash -r 2>/dev/null || true

# Set unrar library path
export UNRAR_LIB_PATH="/usr/lib/x86_64-linux-gnu/libunrar.so"
export UNRAR_LIB_PATH="/usr/lib/libunrar.so"

echo "🔧 PATH updated: $SCRIPT_DIR/lib"
echo ""
echo "🔍 Available tools:"
echo "  7zip:              $(which 7z 2>/dev/null || which 7zz 2>/dev/null || echo 'not found')"
echo "  unrar:             $(which unrar 2>/dev/null || echo 'not found')"
echo "  plastimatch:       $(which plastimatch 2>/dev/null || echo 'not found')"
echo "  isic:              $(which isic 2>/dev/null || echo 'not found')"
echo "  segimage2itkimage: $(which segimage2itkimage 2>/dev/null || echo 'not found')"
EOF

chmod +x setup_path.sh

# ----------------------------------------------------------------------------
# 7. Summary
# ----------------------------------------------------------------------------
echo ""
echo "========================================="
echo "✅ Installation complete!"
echo "========================================="
echo ""
echo "📁 Tools installed in: $(pwd)/lib/"
echo ""
echo "📍 Run: source setup_path.sh"
echo ""
echo "🔍 Installed tools:"
echo "  7zip:              $(which 7z 2>/dev/null || which 7zz 2>/dev/null || echo 'not found')"
echo "  unrar:             $(which unrar 2>/dev/null || echo 'not found')"
echo "  plastimatch:       $(which plastimatch 2>/dev/null || echo 'not found')"
echo "  isic:              $(which isic 2>/dev/null || echo 'not found')"
echo "  segimage2itkimage: $(which segimage2itkimage 2>/dev/null || echo 'not found')"
echo ""