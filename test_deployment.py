"""
Comprehensive Deployment Testing Script
Tests GitHub workflows, Docker configurations, and deployment readiness

Author: Abdul Salam Aldabik
Date: November 24, 2025
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

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
        print(f"      {details}")

def test_github_workflow() -> Tuple[bool, Dict]:
    """Test GitHub Actions workflow configuration"""
    print_header("TEST 1: GitHub Actions Workflow")
    
    workflow_path = Path('.github/workflows/ml_pipeline.yml')
    results = {
        'exists': False,
        'valid_yaml': False,
        'has_triggers': False,
        'has_jobs': False,
        'job_count': 0,
        'jobs': []
    }
    
    if not workflow_path.exists():
        print_test("Workflow file exists", False, f"File not found: {workflow_path}")
        return False, results
    
    results['exists'] = True
    print_test("Workflow file exists", True, str(workflow_path))
    
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
        
        results['valid_yaml'] = True
        print_test("Valid YAML syntax", True)
        
        # Check triggers
        if 'on' in workflow:
            results['has_triggers'] = True
            triggers = workflow['on']
            if isinstance(triggers, dict):
                trigger_list = list(triggers.keys())
            else:
                trigger_list = [triggers]
            print_test("Workflow triggers configured", True, f"Triggers: {', '.join(trigger_list)}")
        
        # Check jobs
        if 'jobs' in workflow:
            results['has_jobs'] = True
            results['job_count'] = len(workflow['jobs'])
            results['jobs'] = list(workflow['jobs'].keys())
            print_test("Jobs defined", True, f"Found {results['job_count']} jobs: {', '.join(results['jobs'])}")
            
            # Check each job
            for job_name, job_config in workflow['jobs'].items():
                has_steps = 'steps' in job_config
                step_count = len(job_config.get('steps', []))
                print_test(f"Job '{job_name}' configured", has_steps, 
                          f"{step_count} steps defined")
        
        return True, results
        
    except yaml.YAMLError as e:
        print_test("Valid YAML syntax", False, f"YAML parse error: {e}")
        return False, results
    except Exception as e:
        print_test("Workflow parsing", False, f"Error: {e}")
        return False, results

def test_docker_files() -> Tuple[bool, Dict]:
    """Test Docker configuration files"""
    print_header("TEST 2: Docker Configuration Files")
    
    docker_files = {
        'Dockerfile.housing': Path('Dockerfile.housing'),
        'Dockerfile.electricity': Path('Dockerfile.electricity'),
        'docker-compose.yml': Path('docker-compose.yml')
    }
    
    results = {
        'files_exist': {},
        'valid_syntax': {},
        'has_required_commands': {}
    }
    
    all_passed = True
    
    for name, path in docker_files.items():
        # Check existence
        exists = path.exists()
        results['files_exist'][name] = exists
        print_test(f"{name} exists", exists, str(path))
        
        if not exists:
            all_passed = False
            continue
        
        # Read and validate
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if name.startswith('Dockerfile'):
                # Check Dockerfile syntax
                required_commands = ['FROM', 'COPY', 'RUN', 'CMD']
                found_commands = [cmd for cmd in required_commands if cmd in content]
                
                valid = len(found_commands) == len(required_commands)
                results['valid_syntax'][name] = valid
                results['has_required_commands'][name] = found_commands
                
                print_test(f"{name} has required commands", valid, 
                          f"Found: {', '.join(found_commands)}")
                
                # Check Python version
                if 'python:3.10' in content.lower() or 'python:3.11' in content.lower():
                    print_test(f"{name} uses appropriate Python", True, "Python 3.10/3.11")
                
                # Check port exposure
                if 'EXPOSE' in content:
                    port = '8501' if 'housing' in name else '8502'
                    has_port = port in content
                    print_test(f"{name} exposes correct port", has_port, f"Port {port}")
                
            elif name == 'docker-compose.yml':
                # Validate docker-compose YAML
                try:
                    compose_config = yaml.safe_load(content)
                    results['valid_syntax'][name] = True
                    print_test(f"{name} valid YAML", True)
                    
                    # Check services
                    if 'services' in compose_config:
                        services = list(compose_config['services'].keys())
                        print_test(f"{name} defines services", True, 
                                  f"Services: {', '.join(services)}")
                        
                        # Check each service
                        for service in services:
                            service_config = compose_config['services'][service]
                            has_build = 'build' in service_config or 'image' in service_config
                            has_ports = 'ports' in service_config
                            
                            print_test(f"Service '{service}' configured", 
                                      has_build and has_ports,
                                      f"Build: {has_build}, Ports: {has_ports}")
                    
                except yaml.YAMLError as e:
                    results['valid_syntax'][name] = False
                    print_test(f"{name} valid YAML", False, f"Parse error: {e}")
                    all_passed = False
                    
        except Exception as e:
            print_test(f"{name} readable", False, f"Error: {e}")
            all_passed = False
    
    return all_passed, results

def test_streamlit_apps() -> Tuple[bool, Dict]:
    """Test Streamlit application files"""
    print_header("TEST 3: Streamlit Applications")
    
    apps = {
        'Housing Price Prediction': Path('Dataset_1_UK_Housing/Code/streamlit_app.py'),
        'Electricity Demand Forecasting': Path('Dataset_2_UK_Historic_Electricity_Demand_Data/Code/streamlit_app.py')
    }
    
    results = {
        'exists': {},
        'has_streamlit_import': {},
        'has_main_function': {},
        'model_files_referenced': {}
    }
    
    all_passed = True
    
    for name, path in apps.items():
        exists = path.exists()
        results['exists'][name] = exists
        print_test(f"{name} app exists", exists, str(path))
        
        if not exists:
            all_passed = False
            continue
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check imports
            has_streamlit = 'import streamlit' in content
            results['has_streamlit_import'][name] = has_streamlit
            print_test(f"{name} imports streamlit", has_streamlit)
            
            # Check for main/run logic
            has_main = 'if __name__' in content or 'st.title' in content
            results['has_main_function'][name] = has_main
            print_test(f"{name} has main logic", has_main)
            
            # Check for model file references
            model_extensions = ['.pkl', '.joblib', '.h5', '.json']
            model_refs = any(ext in content for ext in model_extensions)
            results['model_files_referenced'][name] = model_refs
            print_test(f"{name} references model files", model_refs)
            
            if not (has_streamlit and has_main):
                all_passed = False
                
        except Exception as e:
            print_test(f"{name} readable", False, f"Error: {e}")
            all_passed = False
    
    return all_passed, results

def test_deployment_docs() -> Tuple[bool, Dict]:
    """Test deployment documentation"""
    print_header("TEST 4: Deployment Documentation")
    
    docs = {
        'DEPLOYMENT.md': Path('DEPLOYMENT.md'),
        'CONTRIBUTING.md': Path('CONTRIBUTING.md'),
        '.dockerignore': Path('.dockerignore'),
        'requirements.txt': Path('requirements.txt')
    }
    
    results = {
        'exists': {},
        'size_bytes': {},
        'has_content': {}
    }
    
    all_passed = True
    
    for name, path in docs.items():
        exists = path.exists()
        results['exists'][name] = exists
        
        if exists:
            size = path.stat().st_size
            results['size_bytes'][name] = size
            results['has_content'][name] = size > 100  # At least 100 bytes
            
            print_test(f"{name} exists", True, f"{size} bytes")
            
            if size < 100:
                print_test(f"{name} has content", False, "File too small")
                all_passed = False
        else:
            print_test(f"{name} exists", False)
            all_passed = False
    
    return all_passed, results

def test_github_templates() -> Tuple[bool, Dict]:
    """Test GitHub issue and PR templates"""
    print_header("TEST 5: GitHub Templates")
    
    templates = {
        'PR Template': Path('.github/pull_request_template.md'),
        'Bug Report': Path('.github/ISSUE_TEMPLATE/bug_report.md'),
        'Feature Request': Path('.github/ISSUE_TEMPLATE/feature_request.md')
    }
    
    results = {
        'exists': {},
        'has_checklist': {}
    }
    
    all_passed = True
    
    for name, path in templates.items():
        exists = path.exists()
        results['exists'][name] = exists
        print_test(f"{name} exists", exists, str(path))
        
        if not exists:
            all_passed = False
            continue
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for checklist items or structured format
            has_structure = '- [ ]' in content or '##' in content
            results['has_checklist'][name] = has_structure
            print_test(f"{name} is structured", has_structure)
            
            if not has_structure:
                all_passed = False
                
        except Exception as e:
            print_test(f"{name} readable", False, f"Error: {e}")
            all_passed = False
    
    return all_passed, results

def test_model_files() -> Tuple[bool, Dict]:
    """Test that model files exist for deployment"""
    print_header("TEST 6: Model Files for Deployment")
    
    model_dirs = {
        'Dataset 1 Models': Path('Dataset_1_UK_Housing/Code'),
        'Dataset 2 Models': Path('Dataset_2_UK_Historic_Electricity_Demand_Data/Code')
    }
    
    results = {
        'dirs_exist': {},
        'model_files': {},
        'count': {}
    }
    
    all_passed = True
    
    for name, path in model_dirs.items():
        exists = path.exists()
        results['dirs_exist'][name] = exists
        
        if not exists:
            print_test(f"{name} directory exists", False)
            all_passed = False
            continue
        
        # Find model files
        model_extensions = ['.pkl', '.joblib', '.h5', '.json']
        model_files = []
        
        for ext in model_extensions:
            model_files.extend(list(path.glob(f'*{ext}')))
        
        results['model_files'][name] = [f.name for f in model_files]
        results['count'][name] = len(model_files)
        
        has_models = len(model_files) > 0
        print_test(f"{name} has model files", has_models, 
                  f"Found {len(model_files)} model files")
        
        if has_models:
            for model_file in model_files[:5]:  # Show first 5
                print(f"      - {model_file.name}")
            if len(model_files) > 5:
                print(f"      ... and {len(model_files) - 5} more")
        else:
            all_passed = False
    
    return all_passed, results

def test_git_configuration() -> Tuple[bool, Dict]:
    """Test Git repository configuration"""
    print_header("TEST 7: Git Repository Configuration")
    
    results = {
        'is_git_repo': False,
        'has_gitignore': False,
        'remote_configured': False,
        'branch': None
    }
    
    # Check if .git exists
    git_dir = Path('.git')
    results['is_git_repo'] = git_dir.exists()
    print_test("Git repository initialized", results['is_git_repo'])
    
    # Check .gitignore
    gitignore = Path('.gitignore')
    results['has_gitignore'] = gitignore.exists()
    print_test(".gitignore exists", results['has_gitignore'])
    
    if results['has_gitignore']:
        try:
            with open(gitignore, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for important patterns
            important_patterns = ['*.pkl', '__pycache__', '.venv', '.ipynb_checkpoints']
            found_patterns = [p for p in important_patterns if p in content]
            
            print_test(".gitignore has ML patterns", len(found_patterns) > 0,
                      f"Found: {', '.join(found_patterns)}")
        except:
            pass
    
    # Check git config (if git is available)
    try:
        import subprocess
        result = subprocess.run(['git', 'remote', '-v'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            results['remote_configured'] = True
            print_test("Git remote configured", True)
            lines = result.stdout.strip().split('\n')[:2]
            for line in lines:
                print(f"      {line}")
        
        # Get current branch
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            results['branch'] = result.stdout.strip()
            print_test("Current branch", True, results['branch'])
    except:
        print_test("Git commands", False, "Git not available in PATH")
    
    all_passed = results['is_git_repo'] and results['has_gitignore']
    return all_passed, results

def main():
    """Run all deployment tests"""
    print_header("🧪 DEPLOYMENT TESTING SUITE")
    print(f"Testing GitHub workflows, Docker configs, and deployment readiness")
    print(f"Date: November 24, 2025")
    print(f"Author: Abdul Salam Aldabik")
    
    # Change to project root
    os.chdir(Path(__file__).parent)
    
    # Run tests
    test_results = {}
    
    test_results['github_workflow'] = test_github_workflow()
    test_results['docker_files'] = test_docker_files()
    test_results['streamlit_apps'] = test_streamlit_apps()
    test_results['deployment_docs'] = test_deployment_docs()
    test_results['github_templates'] = test_github_templates()
    test_results['model_files'] = test_model_files()
    test_results['git_config'] = test_git_configuration()
    
    # Summary
    print_header("📊 TEST SUMMARY")
    
    passed_count = sum(1 for passed, _ in test_results.values() if passed)
    total_count = len(test_results)
    pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
    
    print(f"Tests Passed: {GREEN}{passed_count}{RESET}/{total_count}")
    print(f"Pass Rate: {GREEN if pass_rate == 100 else YELLOW}{pass_rate:.1f}%{RESET}")
    
    # Individual results
    print(f"\n{BLUE}Individual Test Results:{RESET}")
    for test_name, (passed, details) in test_results.items():
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"  {status} | {test_name.replace('_', ' ').title()}")
    
    # Save results
    output_file = 'deployment_test_results.json'
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
        print(f"\n{GREEN}🎉 ALL DEPLOYMENT TESTS PASSED!{RESET}")
        print(f"{GREEN}✅ GitHub workflows, Docker configs, and deployment files are ready{RESET}")
    elif pass_rate >= 80:
        print(f"\n{YELLOW}⚠️  MOST TESTS PASSED ({pass_rate:.0f}%){RESET}")
        print(f"{YELLOW}Review failed tests and fix issues before deployment{RESET}")
    else:
        print(f"\n{RED}❌ MULTIPLE TESTS FAILED ({pass_rate:.0f}%){RESET}")
        print(f"{RED}Significant issues found - deployment not recommended{RESET}")
    
    return 0 if pass_rate == 100 else 1

if __name__ == '__main__':
    exit(main())
