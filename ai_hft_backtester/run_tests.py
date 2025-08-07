#!/usr/bin/env python
"""Run all tests for AI-HFT Backtester"""

import subprocess
import sys
import os

def run_tests():
    """Run all test suites"""
    print("=" * 60)
    print("Running AI-HFT Backtester Tests")
    print("=" * 60)
    
    # Change to tests directory
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Run pytest with coverage
    cmd = [
        sys.executable, '-m', 'pytest',
        test_dir,
        '-v',                          # Verbose output
        '--cov=ai_hft_backtester',    # Coverage for our package
        '--cov-report=html',          # HTML coverage report
        '--cov-report=term-missing',  # Terminal report with missing lines
        '--durations=10',             # Show 10 slowest tests
    ]
    
    # Add markers if specified
    if len(sys.argv) > 1:
        if sys.argv[1] == 'unit':
            cmd.extend(['-m', 'unit'])
        elif sys.argv[1] == 'integration':
            cmd.extend(['-m', 'integration'])
        elif sys.argv[1] == 'fast':
            cmd.extend(['-m', 'not slow'])
    
    # Run tests
    result = subprocess.run(cmd)
    
    # Print coverage report location
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("📊 Coverage report generated at: htmlcov/index.html")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        print("=" * 60)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())