# Video Enhancement Backend

This backend provides a unified API for various video enhancement models, including low-light enhancement, glare reduction, dehazing, and tilt correction.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Server:**
    ```bash
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
    ```

## API

The primary endpoint for video processing is:

- `POST /api/process_video/`

This endpoint accepts a video file and a `method` string to specify the desired enhancement.

### Available Methods

- `clahe`: Contrast Limited Adaptive Histogram Equalization
- `unet`: U-Net based low-light enhancement
- `unet_selective`: U-Net based selective low-light enhancement
- `flare-reduction`: Flare reduction
- `glare-dim`: Glare dimming
- `combined`: Combined flare reduction and glare dimming
- `dehaze`: Dehazing
- `tilt`: Tilt correction
