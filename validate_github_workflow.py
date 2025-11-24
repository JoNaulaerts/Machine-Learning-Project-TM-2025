"""
GitHub Actions Workflow Validator
Validates GitHub Actions workflow YAML syntax and structure

Author: Abdul Salam Aldabik
Date: November 24, 2025
"""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def validate_workflow(filepath: Path) -> Tuple[bool, List[str]]:
    """Validate GitHub Actions workflow"""
    issues = []
    warnings = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"Invalid YAML: {e}"]
    except Exception as e:
        return False, [f"Cannot read file: {e}"]
    
    # Check required fields
    if 'name' in workflow:
        print(f"  {GREEN}✓{RESET} Workflow name: {workflow['name']}")
    else:
        warnings.append("No workflow name specified")
    
    # Check triggers (handle both 'on' and True as YAML key)
    triggers_key = 'on' if 'on' in workflow else (True if True in workflow else None)
    
    if triggers_key is None:
        issues.append("Missing 'on' (workflow triggers)")
    else:
        triggers = workflow[triggers_key]
        if isinstance(triggers, dict):
            trigger_types = list(triggers.keys())
        elif isinstance(triggers, list):
            trigger_types = triggers
        else:
            trigger_types = [triggers]
        
        print(f"  {GREEN}✓{RESET} Triggers: {', '.join(str(t) for t in trigger_types)}")
        
        # Check push/pull_request paths
        for trigger in ['push', 'pull_request']:
            if trigger in triggers and isinstance(triggers[trigger], dict):
                if 'paths' in triggers[trigger]:
                    paths = triggers[trigger]['paths']
                    print(f"    {BLUE}ℹ{RESET} {trigger} monitors: {', '.join(paths[:3])}")
                if 'branches' in triggers[trigger]:
                    branches = triggers[trigger]['branches']
                    print(f"    {BLUE}ℹ{RESET} {trigger} on branches: {', '.join(branches)}")
    
    # Check jobs
    if 'jobs' not in workflow:
        issues.append("Missing 'jobs' section")
        return False, issues
    
    jobs = workflow['jobs']
    print(f"  {GREEN}✓{RESET} Jobs defined: {len(jobs)}")
    
    # Validate each job
    for job_name, job_config in jobs.items():
        print(f"\n  {BLUE}Job: {job_name}{RESET}")
        
        # Check runs-on
        if 'runs-on' not in job_config:
            issues.append(f"Job '{job_name}': Missing 'runs-on'")
        else:
            print(f"    {GREEN}✓{RESET} Runs on: {job_config['runs-on']}")
        
        # Check steps
        if 'steps' not in job_config:
            issues.append(f"Job '{job_name}': Missing 'steps'")
        else:
            steps = job_config['steps']
            print(f"    {GREEN}✓{RESET} Steps: {len(steps)}")
            
            # Validate steps
            for i, step in enumerate(steps, 1):
                step_name = step.get('name', f'Step {i}')
                
                # Check if step has action or run
                if 'uses' not in step and 'run' not in step:
                    issues.append(f"Job '{job_name}', Step {i}: Must have 'uses' or 'run'")
                
                # Check popular actions
                if 'uses' in step:
                    action = step['uses']
                    if 'actions/checkout' in action:
                        print(f"      {GREEN}✓{RESET} {step_name}: Checks out code")
                    elif 'actions/setup-python' in action:
                        version = step.get('with', {}).get('python-version', 'unknown')
                        print(f"      {GREEN}✓{RESET} {step_name}: Sets up Python {version}")
                    elif 'actions/cache' in action:
                        print(f"      {GREEN}✓{RESET} {step_name}: Caches dependencies")
                    else:
                        print(f"      {BLUE}ℹ{RESET} {step_name}: Uses {action}")
                
                # Check run commands
                if 'run' in step:
                    run_cmd = step['run']
                    if isinstance(run_cmd, str):
                        lines = run_cmd.split('\n')
                        cmd_preview = lines[0][:50] + '...' if len(lines[0]) > 50 else lines[0]
                        print(f"      {GREEN}✓{RESET} {step_name}: {cmd_preview}")
        
        # Check needs (job dependencies)
        if 'needs' in job_config:
            needs = job_config['needs']
            if isinstance(needs, list):
                print(f"    {BLUE}ℹ{RESET} Depends on: {', '.join(needs)}")
            else:
                print(f"    {BLUE}ℹ{RESET} Depends on: {needs}")
        
        # Check if
        if 'if' in job_config:
            print(f"    {BLUE}ℹ{RESET} Conditional: {job_config['if'][:50]}...")
    
    # Check for common best practices
    print(f"\n  {BLUE}Best Practices Check:{RESET}")
    
    # Caching
    has_cache = any(
        'actions/cache' in str(step.get('uses', ''))
        for job in jobs.values()
        for step in job.get('steps', [])
    )
    if has_cache:
        print(f"    {GREEN}✓{RESET} Uses caching for dependencies")
    else:
        warnings.append("Consider adding dependency caching for faster builds")
    
    # Checkout action
    has_checkout = any(
        'actions/checkout' in str(step.get('uses', ''))
        for job in jobs.values()
        for step in job.get('steps', [])
    )
    if has_checkout:
        print(f"    {GREEN}✓{RESET} Checks out repository code")
    else:
        issues.append("Missing actions/checkout - code won't be available")
    
    # Environment variables
    has_env = any(
        'env' in job or any('env' in step for step in job.get('steps', []))
        for job in jobs.values()
    )
    if has_env:
        print(f"    {GREEN}✓{RESET} Uses environment variables")
    
    return len(issues) == 0, issues + [f"Warning: {w}" for w in warnings]

def main():
    """Run GitHub Actions workflow validation"""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{'⚙️  GITHUB ACTIONS WORKFLOW VALIDATION'.center(80)}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    workflow_path = Path('.github/workflows/ml_pipeline.yml')
    
    if not workflow_path.exists():
        print(f"{RED}❌ Workflow file not found: {workflow_path}{RESET}")
        return 1
    
    print(f"{BLUE}Validating: {workflow_path}{RESET}\n")
    
    valid, issues = validate_workflow(workflow_path)
    
    if issues:
        print(f"\n{YELLOW}Issues/Warnings:{RESET}")
        for issue in issues:
            if issue.startswith('Warning:'):
                print(f"  {YELLOW}⚠{RESET} {issue}")
            else:
                print(f"  {RED}✗{RESET} {issue}")
    
    print(f"\n{BLUE}{'='*80}{RESET}")
    if valid:
        print(f"{GREEN}✅ WORKFLOW VALID{RESET}")
        print(f"{GREEN}GitHub Actions workflow is correctly configured{RESET}")
    else:
        print(f"{RED}❌ WORKFLOW HAS ERRORS{RESET}")
        print(f"{RED}Fix errors before pushing to GitHub{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")
    
    return 0 if valid else 1

if __name__ == '__main__':
    exit(main())
