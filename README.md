# AQ_Net2 Project: PM2.5 Prediction with ClimaX

This project provides a professional, well-structured pipeline for fine-tuning the ClimaX model for PM2.5 forecasting. It leverages meteorological and air quality data, incorporating a flexible, lead-time-aware prediction mechanism inspired by the original ClimaX paper.

## Key Features

- **Professional Structure**: The codebase is organized into modular components (`src`, `configs`, `scripts`) for clarity and maintainability.
- **Centralized Configuration**: All parameters, from hyperparameters to data paths, are managed in a single `config.yaml` file.
- **ClimaX-Inspired Lead Time Forecasting**: The model uses a lead time embedding, allowing it to make predictions for variable future horizons (e.g., 24h, 48h, 72h) without changing its architecture.
- **Automated Temporal Conversion**: The configuration manager automatically converts human-readable settings (e.g., `forecast_days: 3`) into the required hourly format for the model (`time_future: 72`).
- **Cleaned Dependencies**: Only the essential components of the ClimaX library are included, minimizing clutter and potential conflicts.
- **Reproducibility**: SLURM submission scripts and detailed configuration ensure that experiments are easy to reproduce.

---

## Project Structure

The project is organized as follows:

```
/scratch/project_462000640/ammar/aq_net2/
├── configs/
│   └── config.yaml          # Central configuration for all parameters
├── src/
│   ├── __init__.py
│   ├── climax_core/         # Minimal, essential ClimaX source code
│   │   ├── __init__.py
│   │   ├── arch.py            # Core ClimaX architecture
│   │   ├── parallelpatchembed.py
│   │   └── utils/           # Positional embedding utilities
│   ├── config_manager.py    # Manages loading and processing of config.yaml
│   ├── dataloader.py        # CAQRADataset for loading NetCDF files
│   ├── model.py             # PM25Model definition (ClimaX + regression head)
│   └── train.py             # Training, validation, and evaluation loops
├── scripts/
│   ├── run.sh               # Sets up environment and runs the main script
│   └── submit.sh            # SLURM script for submitting jobs on HPC
├── outputs/
│   └── checkpoints/         # Directory for saving the best model weights
├── logs/                    # Directory for SLURM output and error logs
├── archive/
│   └── ...                  # Old scripts and unused files are archived here
├── main.py                  # Main entry point to start training and evaluation
└── README.md                # This documentation file
```

---

## Configuration

All project parameters are managed in **`configs/config.yaml`**. This is the **single source of truth** for any experiment.

### Temporal Configuration

This is the most important section for controlling the prediction horizon. All settings are automatically converted from days to hours by the `ConfigManager`.

```yaml
data:
  time_step: 1          # Time resolution of the data in hours.
  history_days: 7       # Number of past days to use as input.
  forecast_days: 3      # Number of future days to predict.
```

- With `forecast_days: 3`, the model will be trained to predict the weather **at** a lead time of 72 hours.
- To predict 5 days ahead, simply change `forecast_days: 5`.

### Other Key Configurations

- **`train`**: `epochs`, `batch_size`, `learning_rate`, `patience` for early stopping.
- **`data`**: `data_path`, `checkpoint_path`, `train/val/test_years`, `variables`.
- **`model`**: ClimaX architecture details (`embed_dim`, `depth`, `num_heads`, etc.).
- **`system`**: Environment settings like `miopen_cache_dir` and `num_workers`.

---

## Architecture: Lead Time Aware Prediction

The model follows the ClimaX architecture for horizon-agnostic forecasting.

1.  **Lead Time Embedding**: Instead of predicting a fixed future step, the model takes a `lead_times` argument (in hours). This value is fed into an embedding layer (`self.climax.lead_time_embed`), making the model aware of the desired temporal distance for the prediction.

2.  **Flexible Forward Pass**: The model's `forward` method was updated:

    *   **Old (fixed prediction)**: `model.forward(x)`
    *   **New (flexible prediction)**: `model.forward(x, lead_times)`

    This allows the same trained model to be used for different forecast horizons during inference.

3.  **Dataloader**: The `CAQRADataset` was updated to calculate the `lead_time` in hours between the last input timestamp and the target timestamp for each sample.

---

## Usage

### 1. Configure Your Experiment

Modify **`configs/config.yaml`** to set your desired parameters (e.g., `forecast_days`, `learning_rate`).

### 2. Run the Project

There are three ways to run the project:

**A. Direct Execution (for testing/debugging):**

Make sure your virtual environment is activated.

```bash
python main.py --config configs/config.yaml
```

**B. Using the Run Script:**

This script handles setting up all environment variables before launching `main.py`.

```bash
# Make sure the script is executable
chmod +x scripts/run.sh

# Run it
./scripts/run.sh
```

**C. Submitting to a SLURM Cluster (for production training):**

The `submit.sh` script contains all the necessary `#SBATCH` directives for the LUMI supercomputer. It calls `run.sh` to execute the training.

```bash
sbatch scripts/submit.sh
```

The logs will be saved to the `logs/` directory.

---

## Project Details

- **Input Variables**: `u`, `v`, `temp`, `rh`, `psfc`, `pm10`, `so2`, `no2`, `co`, `o3`
- **Target Variable**: `pm25`
- **Evaluation Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).
- **Core Dependencies**: PyTorch, xarray, NetCDF4, scikit-learn, PyYAML.

This project is now structured for clarity, reproducibility, and ease of future development.

---

## Environment Setup

### For LUMI Supercomputer

1. Run the setup script:
```bash
bash setup_lumi.sh
```

2. For subsequent sessions:
```bash
module load LUMI/24.03 partition/G pytorch/2.4.0
source venv_lumi/bin/activate
```

### For Other Systems

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Requirements Files

- **`requirements.txt`**: General requirements for any system
- **`requirements_lumi.txt`**: Specific requirements for LUMI (AMD GPU optimized)
- **`setup_lumi.sh`**: Automated setup script for LUMI

### Note for LUMI Users

On LUMI, PyTorch should be installed via system modules (optimized for AMD GPU) rather than pip:
```bash
module load LUMI/24.03 partition/G pytorch/2.4.0
```

Then install additional packages from `requirements_lumi.txt`.

---

## Git Usage

The project includes a `.gitignore` file that excludes:
- Virtual environments (`venv/`, `local_packages/`)
- Model checkpoints and data files
- Temporary and cache files
- IDE-specific files

Make sure to push to a clean repository without these unnecessary files.
