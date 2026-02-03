# Predictive Maintenance Backend

This is a backend system for failure detection using NASA's CMAPSS dataset.

## Setup Instructions

1. **Clone the repository** (if not already done)
2. **Create a virtual environment**:
   ```powershell
   python -m venv .venv
   ```
3. **Activate the virtual environment**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
4. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
5. **Run the API**:
   ```powershell
   uvicorn app.main:app --reload
   ```

## Project Structure
- `app/`: FastAPI application and business logic.
- `ml/`: Machine Learning model training and inference.
- `data/`: Dataset storage (ignored by git).
- `scripts/`: Utility scripts.
