# FloodWatch Drone — Streamlit Community Cloud

Cloud-safe version of FloodWatch Drone.

## Files

- `app.py`
- `requirements.txt`
- `.streamlit/secrets.toml.example`

## Deploy

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Deploy the repository on Streamlit Community Cloud.
4. Open **App settings / Secrets**.
5. Add:

```toml
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"
```

6. Save and reboot the app if necessary.

## Important security note

Do not commit your real Roboflow API key to GitHub.

If an API key was previously committed or shared publicly, rotate/revoke it in
Roboflow and use the new key in Streamlit Secrets.

## Why this version has no OpenCV

The first workshop/demo version only needs image upload and browser camera
snapshot. Pillow handles image processing, while inference is performed through
Roboflow's serverless API. This avoids OpenCV binary/dependency issues in cloud
deployment.

Video support can be added later as a separate version using
`opencv-python-headless`.
