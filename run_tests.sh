#!/bin/bash
# Script to run all tests

echo "================================"
echo "Running Test Suite"
echo "================================"
echo ""

echo "1. Testing ISIC DataModule..."
pytest tests/test_isic_datamodule.py -v

echo ""
echo "2. Testing Generic DataModules..."
pytest tests/test_datamodules.py -v

echo ""
echo "3. Testing Configurations..."
pytest tests/test_configs.py -v -k "not slow"

echo ""
echo "4. Testing Spatial Encoding..."
pytest tests/test_spatial_encoding.py -v

echo ""
echo "5. Testing Model..."
pytest tests/test_model.py -v

echo ""
echo "================================"
echo "Test Summary"
echo "================================"
pytest tests/ --co -q
