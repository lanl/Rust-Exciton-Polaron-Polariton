#!/bin/bash
# Example script for running MPI-parallelized band structure calculations

# Configuration
N_RANKS=4
EXECUTABLE="./target/release/light-matter"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MPI Band Structure Calculation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if MPI is available
if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Please install OpenMPI or MPICH."
    echo ""
    echo "Installation:"
    echo "  Ubuntu/Debian: sudo apt-get install openmpi-bin libopenmpi-dev"
    echo "  CentOS/RHEL:   sudo yum install openmpi openmpi-devel"
    echo "  macOS:         brew install open-mpi"
    exit 1
fi

echo -e "${GREEN}✓${NC} MPI found: $(which mpirun)"
echo ""

# Build the project
echo -e "${BLUE}Building project in release mode...${NC}"
cargo build --release

if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo -e "${GREEN}✓${NC} Build successful"
echo ""

# Run with MPI
echo -e "${BLUE}Running with ${N_RANKS} MPI ranks...${NC}"
echo ""

# Set environment to show MPI info
export OMPI_MCA_mpi_show_mca_params=1

# Run the calculation
mpirun -np ${N_RANKS} ${EXECUTABLE}

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓${NC} Calculation completed successfully"
    echo ""
    echo "Results saved to: pp_data/"
else
    echo ""
    echo "Error: Calculation failed"
    exit 1
fi

# Clean up temporary MPI files
if [ -d "pp_data/mpi_temp" ]; then
    echo "Cleaning up temporary MPI files..."
    rm -rf pp_data/mpi_temp
    echo -e "${GREEN}✓${NC} Cleanup complete"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Done!${NC}"
echo -e "${BLUE}========================================${NC}"
