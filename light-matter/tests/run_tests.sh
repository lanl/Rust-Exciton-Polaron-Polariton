#!/bin/bash
# Test runner script for the light-matter crate
# This script provides convenient commands for running different test suites

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

print_header() {
    echo ""
    print_msg "═══════════════════════════════════════════════════" "$BLUE"
    print_msg "$1" "$BLUE"
    print_msg "═══════════════════════════════════════════════════" "$BLUE"
    echo ""
}

# Function to run tests and capture result
run_test_suite() {
    local test_name=$1
    local description=$2
    
    print_header "$description"
    
    if [ -z "$test_name" ]; then
        # Run all tests
        cargo test
    else
        # Run specific test file
        cargo test --test "$test_name"
    fi
    
    local result=$?
    
    if [ $result -eq 0 ]; then
        print_msg "✓ Tests passed!" "$GREEN"
    else
        print_msg "✗ Tests failed!" "$RED"
        exit $result
    fi
    
    return $result
}

# Show usage
show_usage() {
    print_header "Light-Matter Test Suite Runner"
    
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  all               Run all tests (default)"
    echo "  matrix            Run MatrixOps tests"
    echo "  hamiltonians      Run Hamiltonian tests"
    echo "  operators         Run ComposedOperator tests"
    echo "  properties        Run property-based tests"
    echo "  quick             Run quick smoke tests"
    echo "  verbose           Run all tests with output"
    echo "  ignored           Run ignored tests (SLEPc)"
    echo "  coverage          Generate test coverage report"
    echo "  clean             Clean test artifacts"
    echo "  help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 matrix         # Run only MatrixOps tests"
    echo "  $0 verbose        # Run all tests with detailed output"
    echo "  $0 coverage       # Generate coverage report"
}

# Main script
case "${1:-all}" in
    all)
        print_header "Running Full Test Suite"
        cargo test
        ;;
        
    matrix)
        run_test_suite "test_matrix_ops" "Matrix Operations Tests"
        ;;
        
    hamiltonians)
        run_test_suite "test_hamiltonians" "Hamiltonian Tests"
        ;;
        
    operators)
        run_test_suite "test_composed_operators" "Composed Operator Tests"
        ;;
        
    properties)
        run_test_suite "test_properties" "Property-Based Tests"
        ;;
        
    quick)
        print_header "Quick Smoke Tests"
        print_msg "Running subset of fast tests..." "$YELLOW"
        cargo test test_dense_eye test_sparse_eye test_composed_identity --test test_matrix_ops
        cargo test test_boson_mode_creation --test test_hamiltonians
        print_msg "✓ Smoke tests passed!" "$GREEN"
        ;;
        
    verbose)
        print_header "Running Tests with Verbose Output"
        cargo test -- --nocapture --test-threads=1
        ;;
        
    ignored)
        print_header "Running Ignored Tests (SLEPc Required)"
        print_msg "Note: These tests require PETSc with complex support" "$YELLOW"
        cargo test -- --ignored
        ;;
        
    coverage)
        print_header "Generating Test Coverage Report"
        
        # Check if tarpaulin is installed
        if ! command -v cargo-tarpaulin &> /dev/null; then
            print_msg "cargo-tarpaulin not found. Installing..." "$YELLOW"
            cargo install cargo-tarpaulin
        fi
        
        print_msg "Generating coverage report..." "$YELLOW"
        cargo tarpaulin --out Html --output-dir ./coverage
        
        print_msg "✓ Coverage report generated in ./coverage/index.html" "$GREEN"
        ;;
        
    clean)
        print_header "Cleaning Test Artifacts"
        cargo clean
        rm -rf coverage/
        rm -f test_output.log
        print_msg "✓ Cleaned!" "$GREEN"
        ;;
        
    help)
        show_usage
        ;;
        
    *)
        print_msg "Unknown command: $1" "$RED"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
