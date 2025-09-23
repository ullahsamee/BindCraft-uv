#!/bin/bash
###################################################################
### Install BindCraft with Python3.12, CUDA 13.0 and jax 0.7.2 ####
###################################################################

install_dir=$(pwd)

cuda='13.0'

OPTIONS=c:
LONGOPTIONS=cuda:

PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

while true; do
  case "$1" in
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

echo -e "CUDA version: $cuda"
echo -e "Package manager: UV (with source install and patches)"

SECONDS=0

echo -e "Installing/updating UV package manager\n"
if ! command -v uv &> /dev/null; then
    echo -e "Installing UV...\n"
    curl -LsSf https://astral.sh/uv/install.sh | sh || { echo -e "Error: Failed to install UV"; exit 1; }
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo -e "UV already installed. Continuing...\n"
fi
uv --version || { echo -e "Error: UV installation failed"; exit 1; }
echo -e "UV is ready\n"

echo -e "Creating BindCraft environment with Python 3.12\n"
if [ ! -d "BindCraft/bin" ]; then
    echo -e "BindCraft environment not found, creating...\n"
    uv venv BindCraft --python 3.12 || { echo -e "Error: Failed to create UV environment"; exit 1; }
else
    echo -e "BindCraft environment already exists. Activating...\n"
fi

source BindCraft/bin/activate || { echo -e "Error: Failed to activate UV environment"; exit 1; }
echo -e "UV environment activated: $(which python)\n"

echo -e "Installing packages from PyPI...\n"
uv pip install \
    "pandas>=2.0.0" \
    "matplotlib>=3.7.0" \
    "numpy>=1.24.0" \
    "scipy>=1.10.0" \
    "seaborn>=0.12.0" \
    "tqdm>=4.65.0" \
    "jupyter>=1.0.0" \
    "ipykernel>=6.25.0" \
    "biopython>=1.81" \
    "chex>=0.1.7" \
    "dm-haiku>=0.0.9" \
    "flax>=0.7.0" \
    "dm-tree>=0.1.8" \
    "joblib>=1.3.0" \
    "ml-collections>=0.1.1" \
    "immutabledict>=3.0.0" \
    "optax>=0.1.7" \
    "py3dmol>=2.0.0" \
    "fsspec>=2023.6.0" \
    "wheel>=0.41.0" \
    "setuptools>=68.0.0" \
    "MDAnalysis>=2.6.0" \
    "openmm>=8.0.0" \
    || { echo -e "Error: Failed to install base packages with UV"; exit 1; }

#Install JAX with CUDA support
echo -e "Verifying/installing JAX with CUDA ${cuda} support\n"
JAX_CHECK=$(python -c "import jax.numpy as jnp; print('cuda' in jnp.ones(1).device().platform)" 2>/dev/null)
if [ "$JAX_CHECK" != "True" ]; then
    echo "CUDA-enabled JAX not found, installing..."
    if [[ "$cuda" == "13"* ]]; then
        uv pip install "jax[cuda13]" || { echo -e "Error: Failed to install JAX with CUDA 13"; exit 1; }
    elif [[ "$cuda" == "12"* ]]; then
        uv pip install "jax[cuda12]" || { echo -e "Error: Failed to install JAX with CUDA 12"; exit 1; }
    else
        uv pip install jax || { echo -e "Error: Failed to install JAX"; exit 1; }
    fi
else
    echo "✅ CUDA-enabled JAX already installed."
fi

# Install PyRosetta
echo -e "Handling PyRosetta installation\n"
pyrosetta_wheel="pyrosetta-2024.39+release.59628fb-cp312-cp312-linux_x86_64.whl"
pyrosetta_url="https://graylab.jhu.edu/download/PyRosetta4/archive/release/PyRosetta4.Release.python312.linux.wheel/${pyrosetta_wheel}"
if ! python -c "import pyrosetta" &> /dev/null; then
    if [ ! -f "${pyrosetta_wheel}" ]; then
        wget --progress=bar:force:noscroll -O "${pyrosetta_wheel}" "${pyrosetta_url}" || { echo -e "Error: Failed to download PyRosetta"; exit 1; }
    fi
    uv pip install "${pyrosetta_wheel}" || { echo -e "Error: Failed to install PyRosetta"; exit 1; }
else
    echo "✅ PyRosetta already installed."
fi

#Install ColabDesign
echo -e "Installing and patching ColabDesign for JAX compatibility\n"
uv pip install --upgrade --force-reinstall --no-cache-dir "git+https://github.com/sokrypton/ColabDesign.git" || { echo "Error: Failed to install ColabDesign"; exit 1; }

MAPPING_FILE="BindCraft/lib/python3.12/site-packages/colabdesign/af/alphafold/model/mapping.py"
if [ -f "$MAPPING_FILE" ]; then
    echo "Patching ColabDesign file: $MAPPING_FILE"
    # Check if patch is needed before applying
    if grep -q "@jax.util.wraps" "$MAPPING_FILE"; then
        sed -i "1i import functools" "$MAPPING_FILE"
        sed -i "s/@jax.util.wraps(fun, docstr=docstr)/@functools.wraps(fun)/g" "$MAPPING_FILE"
        echo "✅ ColabDesign successfully patched."
    else
        echo "✅ ColabDesign already patched or does not require patching."
    fi
else
    echo "⚠️ Warning: ColabDesign mapping file not found. Could not apply patch."
fi

#Install PDBFixer from source
echo -e "Handling PDBFixer installation from source\n"
if ! python -c "import pdbfixer" &> /dev/null; then
    echo "PDBFixer not found, installing from source..."
    PDBFIXER_URL="https://github.com/openmm/pdbfixer/archive/refs/tags/v1.11.tar.gz"
    PDBFIXER_ARCHIVE="pdbfixer-1.11.tar.gz"
    PDBFIXER_DIR="pdbfixer-1.11"
    wget --progress=bar:force:noscroll -O "${PDBFIXER_ARCHIVE}" "${PDBFIXER_URL}" || { echo -e "Error: Failed to download PDBFixer"; exit 1; }
    tar -xzvf "${PDBFIXER_ARCHIVE}" || { echo -e "Error: Failed to extract PDBFixer"; exit 1; }
    ( cd "${PDBFIXER_DIR}" && python setup.py install ) || { echo -e "Error: Failed to install PDBFixer"; exit 1; }
    rm -f "${PDBFIXER_ARCHIVE}" && rm -rf "${PDBFIXER_DIR}"
    echo "✅ PDBFixer installed successfully from source."
else
    echo "✅ PDBFixer already installed."
fi

echo -e "\nPatching settings files to use local paths...\n"
SETTINGS_FILE="./settings_target/PDL1.json"
if [ -f "$SETTINGS_FILE" ]; then
    sed -i 's|/content/drive/My Drive/BindCraft/PDL1/|./designs/PDL1/|g' "$SETTINGS_FILE"
    sed -i 's|/content/bindcraft/example/PDL1.pdb|./example/PDL1.pdb|g' "$SETTINGS_FILE"
    echo "✅ $SETTINGS_FILE successfully patched."
else
    echo "⚠️ Warning: Settings file $SETTINGS_FILE not found. Could not apply path corrections."
fi

#Verify packages installations
echo -e "\nVerifying all critical package installations\n"
python -c "import sys; print(f'Python version: {sys.version}')"
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'JAX devices: {jax.devices()}')" || { echo "Error: JAX verification failed"; exit 1; }
python -c "import pyrosetta; print('✅ PyRosetta imported successfully')" || { echo "Error: PyRosetta verification failed"; exit 1; }
python -c "import colabdesign; print('✅ ColabDesign imported successfully')" || { echo "Error: ColabDesign verification failed"; exit 1; }
python -c "import pdbfixer; print('✅ PDBFixer imported successfully')" || { echo "Error: PDBFixer verification failed"; exit 1; }
echo -e "\nAll package verifications passed ✅\n"

#AlphaFold2 weights
echo -e "Handling AlphaFold2 model weights...\n"
params_dir="${install_dir}/params"
if [ ! -f "${params_dir}/params_model_5_ptm.npz" ]; then
    echo -e "AlphaFold2 weights not found. Downloading (5.2 GB)...\n"
    mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
    params_archive="${params_dir}/alphafold_params_2022-12-06.tar"
    wget --progress=bar:force:noscroll -O "${params_archive}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
    tar -xf "${params_archive}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2 weights"; exit 1; }
    rm "${params_archive}"
else
    echo -e "✅ AlphaFold2 weights already exist."
fi

echo -e "\nCreating activation script...\n"
cat > activate_bindcraft.sh << 'EOF'
#!/bin/bash
# BindCraft Environment Activation Script
echo "Activating BindCraft environment..."
source "$(dirname "$0")/BindCraft/bin/activate"
echo "✅ BindCraft environment activated!"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
EOF
chmod +x activate_bindcraft.sh

deactivate 2>/dev/null || true
t=$SECONDS
echo -e " Successfully finished BindCraft installation!\n"
echo -e "  Installation completed in $(($t / 3600))h $((($t / 60) % 60))m $(($t % 60))s\n"

echo -e "--- NEXT STEPS ---\n"
echo -e "1. Activate the new environment with this command:\n"
printf "   \033[1;32msource ./activate_bindcraft.sh\033[0m\n\n"

echo -e "2. After activation, run an example design by copying and pasting this command:\n"
printf "   \033[1;32mpython -u ./bindcraft.py --settings './settings_target/PDL1.json' --filters './settings_filters/default_filters.json' --advanced './settings_advanced/default_4stage_multimer.json'\033[0m\n\n"
