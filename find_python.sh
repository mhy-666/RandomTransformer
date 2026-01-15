cat > check_torch.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=check_torch
#SBATCH --account=h200ea
#SBATCH --output=check_torch_%j.out
#SBATCH --time=00:05:00
#SBATCH --partition=h200ea

echo "Checking different Python versions for torch..."
echo ""

# 检查系统Python
echo "1. System python3 (/usr/bin/python3):"
/usr/bin/python3 --version
/usr/bin/python3 -c "import torch; print('torch found:', torch.__version__)" 2>&1 || echo "No torch"
echo ""

# 检查可能的conda/venv Python
echo "2. Searching for Python with torch installed:"
for pybin in \
    /hpc/home/hm235/.local/bin/python* \
    /hpc/home/hm235/miniconda3/bin/python* \
    /hpc/home/hm235/anaconda3/bin/python* \
    /hpc/home/hm235/venv/bin/python* \
    /opt/conda/bin/python* \
    /usr/local/bin/python*
do
    if [ -f "$pybin" ]; then
        echo "  Testing: $pybin"
        $pybin --version 2>&1
        $pybin -c "import torch; print('  ✓ torch:', torch.__version__)" 2>&1 || echo "  ✗ No torch"
        echo ""
    fi
done

echo "3. Checking if conda/mamba exists:"
which conda 2>&1 || echo "conda not found"
which mamba 2>&1 || echo "mamba not found"
which micromamba 2>&1 || echo "micromamba not found"
echo ""

echo "4. Checking module system for Python:"
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module avail python 2>&1 | head -20
fi
EOF

sbatch check_torch.sbatch
