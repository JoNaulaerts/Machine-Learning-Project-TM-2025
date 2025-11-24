"""
Docker Syntax Validation Script
Validates Dockerfile and docker-compose.yml syntax without requiring Docker installation

Author: Abdul Salam Aldabik
Date: November 24, 2025
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def validate_dockerfile(filepath: Path) -> Tuple[bool, List[str]]:
    """Validate Dockerfile syntax and best practices"""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception as e:
        return False, [f"Cannot read file: {e}"]
    
    # Check required instructions
    required = ['FROM', 'WORKDIR', 'COPY', 'RUN', 'CMD']
    for instruction in required:
        if not re.search(rf'^{instruction}\s+', content, re.MULTILINE | re.IGNORECASE):
            issues.append(f"Missing required instruction: {instruction}")
    
    # Check FROM is first non-comment instruction
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            if not line.upper().startswith('FROM'):
                issues.append("FROM instruction must be the first instruction")
            break
    
    # Check for Python base image
    if 'FROM python' not in content and 'FROM python' not in content.lower():
        issues.append("Should use Python base image")
    
    # Check for requirements installation
    if 'pip install' not in content:
        issues.append("Missing pip install command")
    
    # Check for EXPOSE instruction (for web apps)
    if 'EXPOSE' not in content:
        issues.append("Warning: No EXPOSE instruction (may be intentional)")
    
    # Check for proper CMD format
    cmd_match = re.search(r'^CMD\s+(.+)$', content, re.MULTILINE | re.IGNORECASE)
    if cmd_match:
        cmd = cmd_match.group(1)
        # Prefer exec form
        if not cmd.startswith('['):
            issues.append("CMD should use exec form (JSON array) for better signal handling")
    
    # Check for HEALTHCHECK (best practice)
    if 'HEALTHCHECK' in content:
        print(f"      {GREEN}✓{RESET} Includes HEALTHCHECK (best practice)")
    
    # Check for multi-stage build optimization
    from_count = len(re.findall(r'^FROM\s+', content, re.MULTILINE | re.IGNORECASE))
    if from_count > 1:
        print(f"      {GREEN}✓{RESET} Uses multi-stage build")
    
    # Check for .dockerignore reference
    dockerignore = filepath.parent / '.dockerignore'
    if dockerignore.exists():
        print(f"      {GREEN}✓{RESET} .dockerignore exists")
    else:
        issues.append("Warning: No .dockerignore file found")
    
    return len(issues) == 0, issues

def validate_compose_file(filepath: Path) -> Tuple[bool, List[str]]:
    """Validate docker-compose.yml structure"""
    issues = []
    
    try:
        import yaml
        with open(filepath, 'r', encoding='utf-8') as f:
            compose = yaml.safe_load(f)
    except ImportError:
        return False, ["PyYAML not installed - cannot validate YAML syntax"]
    except yaml.YAMLError as e:
        return False, [f"Invalid YAML syntax: {e}"]
    except Exception as e:
        return False, [f"Cannot read file: {e}"]
    
    # Check version
    if 'version' in compose:
        version = str(compose['version'])
        print(f"      {BLUE}ℹ{RESET} Compose version: {version}")
    
    # Check services
    if 'services' not in compose:
        issues.append("Missing 'services' section")
        return False, issues
    
    services = compose['services']
    print(f"      {BLUE}ℹ{RESET} Found {len(services)} service(s): {', '.join(services.keys())}")
    
    # Validate each service
    for service_name, service_config in services.items():
        service_issues = []
        
        # Check build or image
        if 'build' not in service_config and 'image' not in service_config:
            service_issues.append(f"Service '{service_name}': Missing 'build' or 'image'")
        
        # Check ports for web services
        if 'ports' in service_config:
            ports = service_config['ports']
            print(f"      {GREEN}✓{RESET} Service '{service_name}': Exposes ports {ports}")
        
        # Check volumes
        if 'volumes' in service_config:
            print(f"      {GREEN}✓{RESET} Service '{service_name}': Has volume mounts")
        
        # Check environment
        if 'environment' in service_config:
            print(f"      {GREEN}✓{RESET} Service '{service_name}': Has environment variables")
        
        # Check depends_on for ordering
        if 'depends_on' in service_config:
            print(f"      {GREEN}✓{RESET} Service '{service_name}': Has dependency ordering")
        
        # Check restart policy
        if 'restart' in service_config:
            print(f"      {GREEN}✓{RESET} Service '{service_name}': Has restart policy ({service_config['restart']})")
        else:
            service_issues.append(f"Service '{service_name}': No restart policy (consider 'restart: unless-stopped')")
        
        issues.extend(service_issues)
    
    # Check networks
    if 'networks' in compose:
        networks = compose['networks']
        print(f"      {GREEN}✓{RESET} Defines {len(networks)} network(s): {', '.join(networks.keys())}")
    
    # Check volumes
    if 'volumes' in compose:
        volumes = compose['volumes']
        print(f"      {GREEN}✓{RESET} Defines {len(volumes)} named volume(s)")
    
    return len(issues) == 0, issues

def main():
    """Run Docker syntax validation"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{'🐳 DOCKER SYNTAX VALIDATION'.center(80)}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    # Validate Dockerfiles
    dockerfiles = [
        ('Dockerfile.housing', Path('Dockerfile.housing')),
        ('Dockerfile.electricity', Path('Dockerfile.electricity'))
    ]
    
    all_valid = True
    
    for name, filepath in dockerfiles:
        print(f"\n{BLUE}Validating: {name}{RESET}")
        print(f"Path: {filepath}")
        
        if not filepath.exists():
            print(f"{RED}❌ File not found{RESET}")
            all_valid = False
            continue
        
        valid, issues = validate_dockerfile(filepath)
        
        if valid:
            print(f"{GREEN}✅ Valid Dockerfile{RESET}")
        else:
            print(f"{RED}❌ Issues found:{RESET}")
            for issue in issues:
                print(f"  {YELLOW}⚠{RESET} {issue}")
            all_valid = False
    
    # Validate docker-compose.yml
    print(f"\n{BLUE}Validating: docker-compose.yml{RESET}")
    compose_path = Path('docker-compose.yml')
    
    if not compose_path.exists():
        print(f"{RED}❌ File not found{RESET}")
        all_valid = False
    else:
        valid, issues = validate_compose_file(compose_path)
        
        if valid:
            print(f"{GREEN}✅ Valid docker-compose.yml{RESET}")
        else:
            print(f"{RED}❌ Issues found:{RESET}")
            for issue in issues:
                print(f"  {YELLOW}⚠{RESET} {issue}")
            all_valid = False
    
    # Summary
    print(f"\n{BLUE}{'='*80}{RESET}")
    if all_valid:
        print(f"{GREEN}✅ ALL DOCKER CONFIGURATIONS VALID{RESET}")
        print(f"{GREEN}Docker files are syntactically correct and follow best practices{RESET}")
    else:
        print(f"{YELLOW}⚠️  ISSUES FOUND{RESET}")
        print(f"{YELLOW}Review and fix issues above before building Docker images{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    return 0 if all_valid else 1

if __name__ == '__main__':
    exit(main())
