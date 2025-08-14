#!/bin/bash

# Build Python bindings for hiholo
# chmod +x build.sh && ./build.sh [clean]

set -e

echo "Building hiholo Python bindings..."

if [ "$1" = "clean" ]; then
    echo "Cleaning build files..."
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info/
    rm -rf hiholo*.so
    echo "Clean complete"
    exit 0
fi

# Check dependencies
echo "Checking dependencies..."

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found. Please ensure CUDA is properly installed."
    exit 1
fi

# Check Python development headers
if ! python3-config --includes &> /dev/null; then
    echo "Error: Python development headers not found. Please install python3-dev or python3-devel package."
    exit 1
fi

# Check pybind11
if ! python3 -c "import pybind11" &> /dev/null; then
    echo "Error: pybind11 module not found. Please install: pip3 install pybind11"
    exit 1
fi

# Get pybind11 path
PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)
echo "Found pybind11 CMake directory: $PYBIND11_CMAKE_DIR"

echo "Dependency check complete"

# Create build directory
mkdir -p build
cd build

echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$PYBIND11_CMAKE_DIR"

echo "Compiling..."
make -j$(nproc)

cd ..

echo "Installing Python module..."
# Copy compiled module to current directory
if [ -f build/hiholo*.so ]; then
    cp build/hiholo*.so ./
    echo "Module copied to current directory"
else
    echo "Error: Compiled module file not found"
    exit 1
fi

echo "Build complete!"
echo ""
echo "To test installation in Python:"
echo "import hiholo"
echo "print(dir(hiholo))"