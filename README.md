# GelSlim 4.0 Shear Field Package

This package provides optical flow-based approximations of shear fields using the RGB vision-based tactile sensor **GelSlim 4.0**.

👉 Original Repository: https://github.com/MMintLab/gelslim_shear

---

## 🔧 Setup

### Option 1: Simple Setup (with `venv`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Using [uv](https://github.com/astral-sh/uv)

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync dependencies (reads `pyproject.toml` and creates a virtual environment):

```bash
uv sync
```

---

## ▶️ Running Optical Flow Scripts

Both scripts will open a pop-up window via `cv2.imshow`, visualizing the computed optical flow.

### Sparse Optical Flow

```bash
python3 single_vector_shear.py
```

Or using `uv`:

```bash
uv run python3 single_vector_shear.py
```

### Dense Optical Flow

```bash
python3 dense_optical_flow.py
```

Or using `uv`:

```bash
uv run python3 dense_optical_flow.py
```