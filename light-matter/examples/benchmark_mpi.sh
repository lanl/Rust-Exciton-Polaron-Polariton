#!/bin/bash
# Benchmark script to compare sequential vs MPI performance

# Configuration
EXECUTABLE="./target/release/light-matter"
MAX_RANKS=16
RESULTS_FILE="benchmark_results.txt"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MPI Performance Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Build
echo -e "${BLUE}Building project...${NC}"
cargo build --release
echo ""

# Clear previous results
> ${RESULTS_FILE}
echo "MPI Performance Benchmark Results" >> ${RESULTS_FILE}
echo "Date: $(date)" >> ${RESULTS_FILE}
echo "========================================" >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

# Function to run and time a calculation
run_benchmark() {
    local n_ranks=$1
    local label=$2
    
    echo -e "${YELLOW}Testing: ${label}${NC}"
    
    # Clear cache to ensure fair comparison
    rm -f pp_data/*.bin
    
    # Run and capture time
    local start_time=$(date +%s.%N)
    
    if [ ${n_ranks} -eq 1 ]; then
        # Sequential run (no MPI)
        ${EXECUTABLE} > /dev/null 2>&1
    else
        # MPI run
        mpirun -np ${n_ranks} ${EXECUTABLE} > /dev/null 2>&1
    fi
    
    local end_time=$(date +%s.%N)
    local elapsed=$(echo "$end_time - $start_time" | bc)
    
    echo "  Time: ${elapsed} seconds"
    echo "${label}: ${elapsed} seconds" >> ${RESULTS_FILE}
    
    # Return elapsed time for speedup calculation
    echo ${elapsed}
}

# Run sequential baseline
echo -e "${BLUE}Running sequential baseline...${NC}"
sequential_time=$(run_benchmark 1 "Sequential (1 rank)")
echo ""

# Run MPI with different numbers of ranks
for n_ranks in 2 4 8 16; do
    if [ ${n_ranks} -le $(nproc) ] && [ ${n_ranks} -le ${MAX_RANKS} ]; then
        echo -e "${BLUE}Running with ${n_ranks} MPI ranks...${NC}"
        mpi_time=$(run_benchmark ${n_ranks} "MPI (${n_ranks} ranks)")
        
        # Calculate speedup
        speedup=$(echo "scale=2; ${sequential_time} / ${mpi_time}" | bc)
        efficiency=$(echo "scale=2; ${speedup} / ${n_ranks} * 100" | bc)
        
        echo "  Speedup: ${speedup}x"
        echo "  Efficiency: ${efficiency}%"
        echo "  Speedup: ${speedup}x" >> ${RESULTS_FILE}
        echo "  Efficiency: ${efficiency}%" >> ${RESULTS_FILE}
        echo ""
    fi
done

# Summary
echo "" >> ${RESULTS_FILE}
echo "========================================" >> ${RESULTS_FILE}
echo "Summary" >> ${RESULTS_FILE}
echo "========================================" >> ${RESULTS_FILE}
echo "Sequential time: ${sequential_time} seconds" >> ${RESULTS_FILE}
echo "System cores: $(nproc)" >> ${RESULTS_FILE}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo ""
echo "Summary:"
cat ${RESULTS_FILE}

# Create a simple plot if gnuplot is available
if command -v gnuplot &> /dev/null; then
    echo ""
    echo "Generating speedup plot..."
    
    # Extract data for plotting
    grep "MPI" ${RESULTS_FILE} | grep -oP '\d+ ranks.*\d+\.\d+ seconds' | \
        awk '{print $1, $NF}' > speedup_data.txt
    
    # Create gnuplot script
    cat > plot_speedup.gp << 'EOF'
set terminal png size 800,600
set output 'speedup_plot.png'
set title 'MPI Speedup vs Number of Ranks'
set xlabel 'Number of MPI Ranks'
set ylabel 'Speedup'
set grid
set key left top
plot 'speedup_data.txt' using 1:2 with linespoints title 'Actual Speedup', \
     x with lines title 'Ideal Speedup (linear)' dashtype 2
EOF
    
    gnuplot plot_speedup.gp
    echo -e "${GREEN}✓${NC} Speedup plot saved to: speedup_plot.png"
    
    # Cleanup
    rm speedup_data.txt plot_speedup.gp
fi

# Cleanup
rm -rf pp_data/mpi_temp

echo ""
echo -e "${BLUE}Done!${NC}"
