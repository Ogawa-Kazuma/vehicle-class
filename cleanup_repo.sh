#!/bin/bash
# Repository Cleanup Script
# Removes large files from git tracking (but keeps them locally)

echo "Repository Cleanup Script"
echo "========================"
echo ""
echo "This script will remove large files from git tracking."
echo "Files will remain on your local disk but won't be tracked by git."
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: Not a git repository. Run this from the project root."
    exit 1
fi

# Ensure .gitignore exists
if [ ! -f ".gitignore" ]; then
    echo "Error: .gitignore not found. Creating from template..."
    if [ -f ".gitignore.template" ]; then
        cp .gitignore.template .gitignore
        echo "✓ Created .gitignore from template"
    else
        echo "Error: .gitignore.template not found. Please create .gitignore first."
        exit 1
    fi
fi

echo "Removing large files from git tracking..."
echo ""

# Remove video files
echo "Removing video files..."
git rm --cached *.mp4 2>/dev/null && echo "  ✓ Removed .mp4 files" || echo "  ℹ No .mp4 files tracked"
git rm --cached *.avi 2>/dev/null && echo "  ✓ Removed .avi files" || echo "  ℹ No .avi files tracked"
git rm --cached *.mov 2>/dev/null && echo "  ✓ Removed .mov files" || echo "  ℹ No .mov files tracked"

# Remove model files
echo "Removing model files..."
git rm --cached *.pt 2>/dev/null && echo "  ✓ Removed .pt files" || echo "  ℹ No .pt files tracked"
git rm --cached *.pth 2>/dev/null && echo "  ✓ Removed .pth files" || echo "  ℹ No .pth files tracked"

# Remove zip files
echo "Removing zip files..."
git rm --cached *.zip 2>/dev/null && echo "  ✓ Removed .zip files" || echo "  ℹ No .zip files tracked"

# Remove directories
echo "Removing large directories..."
git rm -r --cached venv/ 2>/dev/null && echo "  ✓ Removed venv/" || echo "  ℹ venv/ not tracked"
git rm -r --cached myenv/ 2>/dev/null && echo "  ✓ Removed myenv/" || echo "  ℹ myenv/ not tracked"
git rm -r --cached vehicles_van_package/ 2>/dev/null && echo "  ✓ Removed vehicles_van_package/" || echo "  ℹ vehicles_van_package/ not tracked"
git rm -r --cached video/ 2>/dev/null && echo "  ✓ Removed video/" || echo "  ℹ video/ not tracked"
git rm -r --cached Video_Baru/ 2>/dev/null && echo "  ✓ Removed Video_Baru/" || echo "  ℹ Video_Baru/ not tracked"
git rm -r --cached video_penuh/ 2>/dev/null && echo "  ✓ Removed video_penuh/" || echo "  ℹ video_penuh/ not tracked"
git rm -r --cached dataset/ 2>/dev/null && echo "  ✓ Removed dataset/" || echo "  ℹ dataset/ not tracked"
git rm -r --cached runs/ 2>/dev/null && echo "  ✓ Removed runs/" || echo "  ℹ runs/ not tracked"
git rm -r --cached vehicle_captures/ 2>/dev/null && echo "  ✓ Removed vehicle_captures/" || echo "  ℹ vehicle_captures/ not tracked"
git rm -r --cached frames/ 2>/dev/null && echo "  ✓ Removed frames/" || echo "  ℹ frames/ not tracked"
git rm -r --cached frames-*/ 2>/dev/null && echo "  ✓ Removed frames-*/" || echo "  ℹ frames-*/ not tracked"
git rm -r --cached captured_objects/ 2>/dev/null && echo "  ✓ Removed captured_objects/" || echo "  ℹ captured_objects/ not tracked"
git rm -r --cached Result/ 2>/dev/null && echo "  ✓ Removed Result/" || echo "  ℹ Result/ not tracked"
git rm -r --cached output/ 2>/dev/null && echo "  ✓ Removed output/" || echo "  ℹ output/ not tracked"
git rm -r --cached veh_cls/ 2>/dev/null && echo "  ✓ Removed veh_cls/" || echo "  ℹ veh_cls/ not tracked"

# Remove log files
echo "Removing log files..."
git rm --cached *.csv 2>/dev/null && echo "  ✓ Removed .csv files" || echo "  ℹ No .csv files tracked"
git rm --cached *.log 2>/dev/null && echo "  ✓ Removed .log files" || echo "  ℹ No .log files tracked"

echo ""
echo "========================"
echo "Cleanup complete!"
echo ""
echo "Next steps:"
echo "1. Review the changes: git status"
echo "2. Commit the cleanup: git commit -m 'Remove large files from tracking'"
echo "3. Push to remote (when ready): git push"
echo ""
echo "Note: Files are still on your disk but will no longer be tracked by git."
echo "The .gitignore file will prevent them from being added again."
