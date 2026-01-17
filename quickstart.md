## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Set Up Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Kaggle API
This project downloads data from Kaggle. You'll need API credentials:
1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Click "Create New Token" to download `kaggle.json`
3. Place the file in `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)

### 4. Download and Organize Data
```bash
python ImportData.py
```
This will download the dataset and organize it into the required folder structure.

### 5. Train the Model
```bash
python segmentation.py
```

> **Note:** Training requires a CUDA-compatible GPU. The default configuration uses batch size 16 with 32 initial filters on SegResNet. Adjust these parameters in `segmentation.py` if you encounter memory issues.
