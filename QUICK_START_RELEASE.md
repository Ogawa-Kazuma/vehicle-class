# Quick Start: GitHub Release

This is a quick guide to publish your project to GitHub. Follow these steps in order.

## ⚡ Quick Steps

### 1. Clean Repository (CRITICAL!)

Remove large files from git tracking:

```bash
# Run the cleanup script
./cleanup_repo.sh

# OR manually:
git rm --cached *.mp4 *.pt *.zip
git rm -r --cached venv/ myenv/ video/ vehicles_van_package/ dataset/
git add .gitignore
git commit -m "Remove large files from tracking"
```

### 2. Verify .gitignore

Check that `.gitignore` exists and includes all large files:

```bash
cat .gitignore | grep -E "(\.mp4|\.pt|venv|video)"
```

### 3. Stage New Files

```bash
git add LICENSE README.md CHANGELOG.md CONTRIBUTING.md
git add setup.sh setup.bat requirements.txt
git add examples/ docs/ archive/
git add PREPARATION_SUMMARY.md RELEASE_CHECKLIST.md
git status  # Review what will be committed
```

### 4. Commit Changes

```bash
git commit -m "Prepare project for public release

- Add MIT License
- Comprehensive README with installation guide
- Setup scripts for Linux/Windows
- Documentation and examples
- Archive old script versions
- Add .gitignore to exclude large files"
```

### 5. Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click "New repository"
3. Repository name: `traffic-video-analyzer` (or your choice)
4. Description: "AI-powered traffic video analyzer with YOLOv8"
5. **Keep it PUBLIC** (or private if you prefer)
6. **DO NOT** initialize with README (we already have one)
7. Click "Create repository"

### 6. Push to GitHub

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/traffic-video-analyzer.git

# Rename branch to main (if needed)
git branch -M main

# Push everything
git push -u origin main
```

### 7. Update README URL

After creating the repository, update the URL in README.md:

```bash
# Replace placeholder URLs with your actual repository URL
sed -i 's/yourusername/YOUR_USERNAME/g' README.md
```

Or edit manually in README.md:
- Replace `https://github.com/yourusername/traffic-video-analyzer.git` with your URL

### 8. Create First Release

```bash
# Create and push version tag
git tag -a v1.0.0 -m "Initial public release"
git push origin v1.0.0
```

Then on GitHub:
1. Go to your repository
2. Click "Releases" → "Create a new release"
3. Choose tag: `v1.0.0`
4. Title: `v1.0.0 - Initial Release`
5. Description: Copy content from CHANGELOG.md
6. Click "Publish release"

## ✅ Verification Checklist

Before pushing, verify:

- [ ] Repository size < 100MB (check with `du -sh .git`)
- [ ] No large files in `git ls-files` (check with `git ls-files | grep -E '\.(mp4|pt|zip)$'`)
- [ ] `.gitignore` is committed
- [ ] LICENSE file present
- [ ] README.md looks good
- [ ] All sensitive data removed
- [ ] URLs in README updated (or placeholders noted)

## 🐛 Troubleshooting

### Problem: "remote repository is empty"

**Solution:** Make sure you didn't initialize the GitHub repo with a README. If you did:
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Problem: "file is too large"

**Solution:** The file is already in git history. You need to remove it:
```bash
git rm --cached filename
git commit -m "Remove large file"
git push
```

### Problem: Push fails due to size

**Solution:** Large files might be in git history. Use git filter-branch or BFG Repo Cleaner:
```bash
# Install BFG Repo Cleaner (recommended)
# Then:
bfg --strip-blobs-bigger-than 50M
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

## 📊 Expected Repository Size

After cleanup, your repository should be:
- Source code: ~5-10 MB
- Documentation: ~1-2 MB  
- **Total: <50 MB** (ideally)

## 🎉 Done!

Your project is now on GitHub! Share the link and start getting contributors.

## Next Steps

- Add topics/tags on GitHub (traffic-analysis, yolo, opencv, computer-vision)
- Add a project description
- Consider adding screenshots/demo GIFs
- Share on social media/communities
- Respond to issues and contributions

---

For detailed information, see:
- `PREPARATION_SUMMARY.md` - What was prepared
- `RELEASE_CHECKLIST.md` - Complete checklist
- `RELEASE_SUMMARY.md` - Priority summary
