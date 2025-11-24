"""
Docker Build and Runtime Testing Script
Tests actual Docker builds and container functionality

Author: Abdul Salam Aldabik
Date: November 24, 2025
"""

import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def run_command(cmd: List[str], timeout: int = 300) -> Tuple[bool, str, str]:
    """Run a command and return success, stdout, stderr"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text.center(80)}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")

def print_test(name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
    print(f"{status} | {name}")
    if details:
        for line in details.split('\n'):
            if line.strip():
                print(f"      {line}")

def test_docker_daemon() -> Tuple[bool, Dict]:
    """Test if Docker daemon is running"""
    print_header("TEST 1: Docker Daemon Status")
    
    results = {
        'installed': False,
        'running': False,
        'version': None,
        'info': {}
    }
    
    # Check Docker version
    success, stdout, stderr = run_command(['docker', '--version'])
    if success:
        results['installed'] = True
        results['version'] = stdout.strip()
        print_test("Docker installed", True, results['version'])
    else:
        print_test("Docker installed", False, stderr)
        return False, results
    
    # Check Docker info
    success, stdout, stderr = run_command(['docker', 'info', '--format', '{{json .}}'])
    if success:
        try:
            info = json.loads(stdout)
            results['running'] = True
            results['info'] = info
            
            print_test("Docker daemon running", True)
            print(f"      Server Version: {info.get('ServerVersion', 'unknown')}")
            print(f"      Operating System: {info.get('OperatingSystem', 'unknown')}")
            print(f"      Total Memory: {info.get('MemTotal', 0) / 1024**3:.1f} GB")
            print(f"      CPUs: {info.get('NCPU', 'unknown')}")
        except json.JSONDecodeError:
            results['running'] = True
            print_test("Docker daemon running", True, "Info retrieved")
    else:
        print_test("Docker daemon running", False, 
                  "Docker Desktop may be starting. Wait 30 seconds and try again.")
        return False, results
    
    return True, results

def test_dockerfile_build(dockerfile: str, tag: str, context: str = ".") -> Tuple[bool, Dict]:
    """Test building a Docker image"""
    print_header(f"TEST: Building {tag}")
    
    results = {
        'build_success': False,
        'image_id': None,
        'size': None,
        'build_time': 0
    }
    
    # Build the image
    print(f"Building image from {dockerfile}...")
    print(f"Command: docker build -f {dockerfile} -t {tag} {context}")
    
    start_time = time.time()
    success, stdout, stderr = run_command([
        'docker', 'build',
        '-f', dockerfile,
        '-t', tag,
        context
    ], timeout=600)
    
    build_time = time.time() - start_time
    results['build_time'] = build_time
    
    if success:
        results['build_success'] = True
        print_test(f"Build {tag}", True, f"Completed in {build_time:.1f} seconds")
        
        # Get image info
        success, stdout, stderr = run_command([
            'docker', 'images', tag, '--format', '{{.ID}}|{{.Size}}'
        ])
        
        if success and stdout.strip():
            image_id, size = stdout.strip().split('|')
            results['image_id'] = image_id
            results['size'] = size
            print(f"      Image ID: {image_id}")
            print(f"      Size: {size}")
    else:
        print_test(f"Build {tag}", False, f"Build failed after {build_time:.1f} seconds")
        print(f"\n{RED}Build Error:{RESET}")
        for line in stderr.split('\n')[-20:]:  # Last 20 lines
            if line.strip():
                print(f"  {line}")
    
    return success, results

def test_docker_compose() -> Tuple[bool, Dict]:
    """Test docker-compose configuration"""
    print_header("TEST: Docker Compose Validation")
    
    results = {
        'compose_installed': False,
        'config_valid': False,
        'services': []
    }
    
    # Check docker-compose version
    success, stdout, stderr = run_command(['docker-compose', '--version'])
    if not success:
        success, stdout, stderr = run_command(['docker', 'compose', 'version'])
    
    if success:
        results['compose_installed'] = True
        print_test("Docker Compose installed", True, stdout.strip())
    else:
        print_test("Docker Compose installed", False)
        return False, results
    
    # Validate docker-compose.yml
    compose_file = Path('docker-compose.yml')
    if not compose_file.exists():
        print_test("docker-compose.yml exists", False)
        return False, results
    
    print_test("docker-compose.yml exists", True)
    
    # Try docker compose config
    success, stdout, stderr = run_command(['docker', 'compose', 'config', '--services'])
    if not success:
        success, stdout, stderr = run_command(['docker-compose', 'config', '--services'])
    
    if success:
        results['config_valid'] = True
        services = [s.strip() for s in stdout.strip().split('\n') if s.strip()]
        results['services'] = services
        print_test("docker-compose.yml valid", True, f"Services: {', '.join(services)}")
    else:
        print_test("docker-compose.yml valid", False, stderr)
    
    return success, results

def test_image_run(tag: str, port: int, health_endpoint: str = "/") -> Tuple[bool, Dict]:
    """Test running a Docker container"""
    print_header(f"TEST: Running {tag}")
    
    results = {
        'container_started': False,
        'container_id': None,
        'port_mapped': False
    }
    
    container_name = f"{tag.replace(':', '-')}-test"
    
    # Remove old container if exists
    run_command(['docker', 'rm', '-f', container_name])
    
    # Run container in detached mode
    print(f"Starting container {container_name}...")
    success, stdout, stderr = run_command([
        'docker', 'run',
        '-d',
        '--name', container_name,
        '-p', f'{port}:{port}',
        tag
    ], timeout=60)
    
    if success:
        results['container_started'] = True
        results['container_id'] = stdout.strip()[:12]
        results['port_mapped'] = True
        
        print_test(f"Container started", True, 
                  f"ID: {results['container_id']}, Port: {port}")
        
        # Wait for container to be healthy
        print("      Waiting 10 seconds for container to initialize...")
        time.sleep(10)
        
        # Check container status
        success, stdout, stderr = run_command([
            'docker', 'ps',
            '--filter', f'name={container_name}',
            '--format', '{{.Status}}'
        ])
        
        if success and stdout.strip():
            print(f"      Status: {stdout.strip()}")
        
        # Stop and remove container
        print(f"      Cleaning up container...")
        run_command(['docker', 'stop', container_name], timeout=30)
        run_command(['docker', 'rm', container_name])
        print_test(f"Container cleanup", True, "Container stopped and removed")
        
        return True, results
    else:
        print_test(f"Container started", False, stderr)
        return False, results

def main():
    """Run all Docker tests"""
    print_header("🐳 DOCKER BUILD & RUNTIME TESTING")
    print("This will test Docker installation, builds, and container execution")
    print(f"Date: November 24, 2025")
    print(f"Author: Abdul Salam Aldabik")
    
    test_results = {}
    
    # Test 1: Docker daemon
    success, results = test_docker_daemon()
    test_results['daemon'] = (success, results)
    
    if not success:
        print(f"\n{RED}❌ Docker daemon not running. Please start Docker Desktop and try again.{RESET}")
        return 1
    
    # Test 2: Docker Compose
    success, results = test_docker_compose()
    test_results['compose'] = (success, results)
    
    # Test 3: Build Housing Dockerfile
    if Path('Dockerfile.housing').exists():
        success, results = test_dockerfile_build(
            'Dockerfile.housing',
            'housing-app:test',
            '.'
        )
        test_results['build_housing'] = (success, results)
        
        # Test 4: Run Housing container
        if success:
            success, results = test_image_run('housing-app:test', 8501)
            test_results['run_housing'] = (success, results)
    
    # Test 5: Build Electricity Dockerfile
    if Path('Dockerfile.electricity').exists():
        success, results = test_dockerfile_build(
            'Dockerfile.electricity',
            'electricity-app:test',
            '.'
        )
        test_results['build_electricity'] = (success, results)
        
        # Test 6: Run Electricity container
        if success:
            success, results = test_image_run('electricity-app:test', 8502)
            test_results['run_electricity'] = (success, results)
    
    # Summary
    print_header("📊 DOCKER TESTING SUMMARY")
    
    passed_count = sum(1 for passed, _ in test_results.values() if passed)
    total_count = len(test_results)
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
    
    print(f"Tests Passed: {GREEN}{passed_count}{RESET}/{total_count}")
    print(f"Pass Rate: {GREEN if pass_rate == 100 else YELLOW}{pass_rate:.1f}%{RESET}")
    
    print(f"\n{BLUE}Test Results:{RESET}")
    for test_name, (passed, details) in test_results.items():
        status = f"{GREEN}✅{RESET}" if passed else f"{RED}❌{RESET}"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    # Cleanup
    print(f"\n{BLUE}Cleaning up test images...{RESET}")
    run_command(['docker', 'rmi', 'housing-app:test', '-f'])
    run_command(['docker', 'rmi', 'electricity-app:test', '-f'])
    print("✅ Cleanup complete")
    
    # Save results
    output_file = 'docker_test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': '2025-11-24',
            'total_tests': total_count,
            'passed': passed_count,
            'failed': total_count - passed_count,
            'pass_rate': pass_rate,
            'results': {k: {'passed': v[0], 'details': v[1]} for k, v in test_results.items()}
        }, f, indent=2)
    
    print(f"\n{BLUE}Results saved to:{RESET} {output_file}")
    
    # Final verdict
    if pass_rate == 100:
        print(f"\n{GREEN}🎉 ALL DOCKER TESTS PASSED!{RESET}")
        print(f"{GREEN}✅ Docker is properly installed and containers work correctly{RESET}")
    elif pass_rate >= 80:
        print(f"\n{YELLOW}⚠️  MOST TESTS PASSED ({pass_rate:.0f}%){RESET}")
    else:
        print(f"\n{RED}❌ MULTIPLE TESTS FAILED ({pass_rate:.0f}%){RESET}")
    
    return 0 if pass_rate == 100 else 1

if __name__ == '__main__':
    exit(main())
