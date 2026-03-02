#!/bin/bash
# Quick test script to verify all fixes

echo "🔧 Testing Light-Matter Test Suite"
echo "=================================="
echo ""

echo "📝 Running test_matrix_ops..."
cargo test --test test_matrix_ops 2>&1 | head -20

echo ""
echo "📝 Running test_composed_operators..."
cargo test --test test_composed_operators 2>&1 | head -20

echo ""
echo "📝 Running test_properties..."
cargo test --test test_properties 2>&1 | head -20

echo ""
echo "📝 Running test_hamiltonians_minimal..."
cargo test --test test_hamiltonians_minimal 2>&1 | head -20

echo ""
echo "=================================="
echo "✅ Test compilation check complete!"
