"""
Run all tests.

Executes all test scripts in sequence and reports results.
"""

import subprocess
import sys
from pathlib import Path


def run_test(script_name, config_name="local_dev"):
    """Run a test script and return success status."""
    print("\n" + "=" * 80)
    print(f"Running: {script_name}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name, f"--config-name={config_name}"],
            capture_output=False,
            check=True,
            timeout=120  # 2 minutes timeout
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with return code {e.returncode}")
        return False
    except subprocess.TimeoutExpired:
        print(f"\n✗ {script_name} timed out (>120s)")
        return False
    except Exception as e:
        print(f"\n✗ {script_name} failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("Running All Tests")
    print("=" * 80)
    
    # List of tests to run
    tests = [
        ("test_config.py", "Configuration system"),
        ("test_model.py", "Model creation and forward pass"),
        ("test_losses.py", "Loss functions"),
        ("test_data.py", "Data loading pipeline"),
    ]
    
    results = []
    
    for script, description in tests:
        script_path = Path(script)
        
        if not script_path.exists():
            print(f"\n✗ {script} not found, skipping")
            results.append((script, description, False))
            continue
        
        success = run_test(script)
        results.append((script, description, success))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, _, success in results if success)
    total = len(results)
    
    for script, description, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {description:40s} ({script})")
    
    print("\n" + "-" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
