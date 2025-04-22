# GelSlim 4.0 Shear Field Package

This package provides optical flow-based approximations of shear fields using the RGB vision-based tactile sensor **GelSlim 4.0**.

👉 Original Repository: [MMintLab/gelslim_shear](https://github.com/MMintLab/gelslim_shear)

---

## 🔧 Setup

### Option 1: Simple Setup (with `venv`)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Using [`uv`](https://github.com/astral-sh/uv)

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

Both scripts will open a pop-up window using `cv2.imshow` to visualize the computed optical flow.

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

---

## 📈 Running Shear Estimation Script

### Pre-Setup: Handle `opencv-python` and `matplotlib` Conflict

Due to a known conflict between OpenCV and `matplotlib` using Qt (see [this StackOverflow post](https://stackoverflow.com/questions/63424083/archlinux-python-qt-cant-use-opencv-and-matplotlib-at-the-same-time)), we need to replace `opencv-python` with the headless version:

Using pip:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

Or using `uv`:

```bash
uv remove opencv-python
uv add opencv-python-headless
```

### External Dependency: Natural Helmholtz-Hodge Decomposition

This project uses the [NaturalHHD](https://github.com/bhatiaharsh/naturalHHD) package. Clone it under the `external/` folder, then navigate to:

```
external/naturalHHD/pynhhd-v1.1
```

And install the module:

Using pip:

```bash
pip install -e .
```

Or using `uv`:

```bash
uv pip install -e .
```

Verify installation:

```bash
pip list | grep pynhhd
```

---

### ▶️ Running the Shear Estimation Scripts

Two scripts are provided:

- `single_vector_shear.py`: compares only the first and second frames.
- `multiple_vector_shear.py`: continuously updates shear estimation throughout the video.

Run with:

```bash
uv run python3 multiple_vector_shear.py
```

This will launch a matplotlib window showing:

- Shear field
- Solenoidal component
- Irrotational component
- Time derivative
- Shear change
- Divergence
- Curl

📌 **Note**: Since real GelSlim video data is not available (and not provided by the original repo), a random video is used for demonstration.