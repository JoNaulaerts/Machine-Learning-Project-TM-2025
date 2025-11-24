# Contributing to CloudAI Analytics ML Project

## Team Members
- **Jo Naulaerts** - Dataset 1 Models (Ridge, PyCaret)
- **Abdul Salam Aldabik** - Dataset 2 Models, AWS, Deployment

## Development Workflow

### 1. Branch Strategy
```bash
# Always pull latest before starting work
git pull origin main

# Create feature branch for new work
git checkout -b feature/your-feature-name

# Example branches:
# - feature/housing-eda
# - feature/electricity-lstm
# - feature/streamlit-deployment
# - fix/data-cleaning-bug
```

### 2. Commit Guidelines

**Good Commit Messages:**
```bash
git commit -m "Add XGBoost model for electricity forecasting"
git commit -m "Fix: Correct NA handling in housing data cleaning"
git commit -m "Update: Improve LSTM model architecture (7% → 6% MAPE)"
git commit -m "Docs: Add deployment guide for Oracle Cloud"
```

**Bad Commit Messages:**
```bash
git commit -m "update"
git commit -m "fix"
git commit -m "changes"
```

**Format:**
- Start with verb: Add, Fix, Update, Remove, Refactor, Docs
- Be specific about what changed
- Include metrics if relevant (e.g., "Improve accuracy from X% to Y%")

### 3. Notebook Guidelines

**Every notebook MUST have:**
```markdown
# [Title] - [Purpose]
**Author:** [Your Name]

[Brief description of what this notebook does]
```

**Before committing notebooks:**
```bash
# Clear outputs if file size > 10MB
# Kernel → Restart & Clear Output

# Add author information if missing
# Add markdown explanations before code blocks
```

### 4. Pull Request Process

**Creating PR:**
```bash
# Push your branch
git push origin feature/your-feature-name

# On GitHub: Create Pull Request
# - Title: Clear description of changes
# - Description: What, why, testing done
# - Assign reviewer: Team member for code review
```

**PR Template:**
```markdown
## What Changed
- Added LSTM model for electricity forecasting
- Achieved 7% MAPE on test set

## Why
- Assignment requires multiple models per dataset
- LSTM can capture long-term patterns in time series

## Testing Done
- Trained on 2009-2020 data
- Validated on 2021-2022 data
- Tested on 2023-2024 data
- All cells run without errors

## Files Changed
- Dataset_2_UK_Historic_Electricity_Demand_Data/Code/05_lstm_model.ipynb
- Dataset_2_UK_Historic_Electricity_Demand_Data/Code/06_model_comparison.ipynb
```

### 5. Code Review Checklist

**Reviewer checks:**
- [ ] Author attribution present
- [ ] Markdown explanations clear
- [ ] Code runs without errors
- [ ] No large files (> 100MB) committed
- [ ] .gitignore excludes large .pkl files
- [ ] Commit messages are descriptive

### 6. Merge Strategy
```bash
# Only merge to main after:
# 1. Code review completed
# 2. All tests pass
# 3. No merge conflicts
# 4. CI/CD pipeline passes (if triggered)

# Squash commits for clean history
git merge --squash feature/your-feature-name
git commit -m "Add LSTM model for electricity forecasting (7% MAPE)"
```

## Automated Pipeline

### GitHub Actions Triggers
Our ML pipeline automatically runs when:
- Code pushed to `main` branch
- Changes in `Dataset_*/Code/**` or `Dataset_*/Data/**`
- Manual trigger via Actions tab

**What it does:**
1. Retrains Housing Ridge model
2. Retrains Electricity XGBoost model
3. Saves updated models
4. Auto-commits with `[skip ci]` tag

**Monitoring:**
```bash
# Check pipeline status
# GitHub → Actions → ML Model Training Pipeline
# Green ✅ = success, Red ❌ = failed

# View logs for debugging
# Click on workflow run → Click on job → View logs
```

## File Size Guidelines

### GitHub Limits
- Individual files: < 100MB
- Repository: < 1GB recommended

### Large Files Handling
```bash
# Check file sizes before commit
git ls-files --others --ignored --exclude-standard | while read f; do
  size=$(wc -c < "$f")
  if [ $size -gt 10485760 ]; then
    echo "Large file: $f ($(($size / 1048576)) MB)"
  fi
done

# Add large files to .gitignore
echo "*.pkl" >> .gitignore
echo "*.h5" >> .gitignore
echo "!*_pipeline.pkl" >> .gitignore  # Exception for small models
```

### Model Storage Strategy
- **Small models (< 10MB):** Commit to git (e.g., `*_pipeline.pkl`)
- **Large models (> 10MB):** Store elsewhere, document in notebooks
- **Training outputs:** Keep in notebooks (File → Save with outputs)
- **Screenshots:** Include for long-running models

## Testing Before Commit

### Pre-commit Checklist
```bash
# 1. Verify notebooks run
# Open in VS Code → Run All Cells
# Check for errors

# 2. Test Streamlit apps
cd Dataset_1_UK_Housing/Code
streamlit run streamlit_app.py
# Test functionality in browser

# 3. Verify Docker builds
docker-compose build
# Should complete without errors

# 4. Check git status
git status
# Review all changed files

# 5. Verify no large files
git diff --cached --name-only | while read f; do
  size=$(wc -c < "$f" 2>/dev/null || echo 0)
  if [ $size -gt 104857600 ]; then
    echo "ERROR: File too large: $f"
    exit 1
  fi
done
```

## Common Issues & Solutions

### Issue: "File too large" error
**Solution:**
```bash
# Remove large file from staging
git reset HEAD path/to/large/file.pkl

# Add to .gitignore
echo "path/to/large/file.pkl" >> .gitignore

# If already committed:
git rm --cached path/to/large/file.pkl
git commit -m "Remove large model file (>100MB)"
```

### Issue: Merge conflicts
**Solution:**
```bash
# Pull latest main
git checkout main
git pull origin main

# Rebase your branch
git checkout your-branch
git rebase main

# Resolve conflicts in VS Code
# Edit files → Stage → Continue rebase
git add resolved-file.ipynb
git rebase --continue
```

### Issue: Notebook merge conflicts
**Solution:**
```bash
# Accept theirs (main branch) or yours
git checkout --theirs path/to/notebook.ipynb  # Use main version
git checkout --ours path/to/notebook.ipynb    # Use your version

# Or use nbdiff for smart merging
pip install nbdime
nbdime config-git --enable
git mergetool path/to/notebook.ipynb
```

### Issue: Pipeline fails
**Solution:**
```bash
# Check GitHub Actions logs
# GitHub → Actions → Failed workflow → View logs

# Common causes:
# 1. Missing dependencies → Update .github/workflows/ml_pipeline.yml
# 2. Data file not found → Check file paths
# 3. Import errors → Verify library versions

# Test locally before push:
cd Dataset_1_UK_Housing/Code
python -c "from sklearn.linear_model import Ridge; print('✅ OK')"
```

## Repository Hygiene

### Regular Maintenance
```bash
# Remove old branches
git branch -d feature/completed-feature
git push origin --delete feature/completed-feature

# Clean up local repo
git gc
git prune

# Update .gitignore
# Review and add patterns for generated files
```

### Documentation Updates
```bash
# Keep README current
# Update after major changes:
# - New models added
# - Performance improvements
# - Deployment changes

# Update DEPLOYMENT.md
# Document new hosting options or procedures

# Update PROJECT_REQUIREMENTS_CHECKLIST.md
# Mark completed tasks
```

## Emergency Procedures

### Accidentally Committed Large File
```bash
# Remove from history (USE WITH CAUTION)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/large/file.pkl' \
  --prune-empty --tag-name-filter cat -- --all

# Force push (coordinate with team!)
git push origin main --force
```

### Lost Work / Need to Recover
```bash
# Find lost commits
git reflog

# Restore from reflog
git checkout <commit-hash>
git checkout -b recovery-branch

# Recover deleted files
git checkout <commit-hash> -- path/to/deleted/file.ipynb
```

### Broken Main Branch
```bash
# Revert to last known good commit
git revert <bad-commit-hash>
git push origin main

# Or reset (coordinate with team!)
git reset --hard <good-commit-hash>
git push origin main --force
```

## Best Practices

### Daily Workflow
1. **Morning:** Pull latest, check for updates
2. **During work:** Commit frequently with good messages
3. **Before leaving:** Push branch, create PR if ready
4. **Evening:** Review team PRs, provide feedback

### Communication
- **Slack/Teams:** Notify before large changes
- **PR Comments:** Explain non-obvious code
- **Commit Messages:** Document decisions
- **README:** Keep team informed of progress

### Quality Standards
- **Code:** PEP 8 style, clear variable names
- **Notebooks:** Markdown explanations, author tags
- **Commits:** Atomic, descriptive, tested
- **PRs:** Small, focused, well-documented

## Resources

- **GitHub Docs:** https://docs.github.com/
- **Git Tutorial:** https://git-scm.com/book/en/v2
- **Notebook Diff:** https://nbdime.readthedocs.io/
- **PEP 8 Style:** https://pep8.org/

---

**Last Updated:** November 24, 2025  
**Team:** CloudAI Analytics Team  
**Repository:** https://github.com/JoNaulaerts/Machine-Learning-Project-TM-2025
