# Snake human vs AI

Barebones setup for running the game.

## Requirements

For running the game (without training), the top-level Python packages are:

- `pygame` (project currently uses `pygame-ce` in `requirements.txt`)
- `tensorflow`
- `numpy`

Note: `requirements.txt` was generated on Linux and includes many dev/GPU/Linux-specific packages that may not install on Windows.

## Install

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install --upgrade pip
pip install pygame tensorflow numpy
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Run

From the project root:

```bash
python -m src.main
```

## Optional (Linux/full environment)

If you want the full Linux environment used in development:

```bash
pip install -r requirements.txt
```
