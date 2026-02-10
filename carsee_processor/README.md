# CarSee Processor (Cloud API)

Runs CarSee ONNX damage/reference models in the cloud so phones can offload heavy inference to a server (e.g. Render.com).

## Deploy on Render.com

### 1. One-time setup

1. Go to [Render](https://render.com) and sign in (or create an account).
2. Click **New +** → **Web Service**.
3. Connect your repository (GitHub/GitLab) that contains this `carsee_processor` folder, or use the steps below to deploy without linking a repo.

### 2. Deploy from repo (recommended)

- **Repository**: Select the repo that contains `Carsee_V0.99` (or the repo where `carsee_processor` lives).
- **Root Directory**: Set to `carsee_processor` (so Render uses this folder as the project root).
- **Environment**: **Docker**.
- **Name**: e.g. `carsee-processor`.
- **Region**: Choose one close to your users.
- **Plan**: Free or paid. Free tier may spin down after inactivity; first request after idle can take 30–60 seconds while models load.

Click **Create Web Service**. Render will build the Docker image and start the service. The first deploy can take a few minutes because the app downloads the ONNX models from GitHub at startup.

### 3. Get your service URL

After the service is live, Render shows a URL like:

`https://carsee-processor-xxxx.onrender.com`

Use this as the **processing API URL** in the Flutter app (see below).

### 4. Configure the Flutter app

In your Flutter project, set the cloud processing URL so the app uses the server instead of on-device models:

**File:** `car_handle_detector/lib/config/supabase_config.dart`

Set:

```dart
static const String processingApiUrl = 'https://YOUR-SERVICE-NAME.onrender.com';
```

Replace with your actual Render URL (no trailing slash). Leave empty (`''`) to process photos on the device.

Rebuild and install the app. New reports will then be processed in the cloud when `processingApiUrl` is set.

## API

- **GET /health** – Returns `{ "status": "ok", "models_loaded": 7 }` when models are ready.
- **POST /process** – Multipart form:
  - `images`: one or more image files (e.g. JPEG).
  - `tire_diameter` (optional, default 57.47)
  - `handle_width` (optional, default 20.6)
  - `license_plate_width` (optional, default 32.0)

Response: `{ "photos": [ { "detectionsByModel": { "car_component_detector.onnx": [...], ... }, "referenceResult": { "type", "scale", "realSizeCm" } }, ... ] }`

## Models

Models are downloaded from GitHub at startup:

`https://github.com/Icestreamm/carsee/releases/download/v1.0.0/<filename>`

Ensure the release exists and contains the ONNX files used by the Flutter app (e.g. `car_component_detector.onnx`, `damage_detector_capstone.onnx`, etc.).

## Run locally

```bash
cd carsee_processor
pip install -r requirements.txt
python main.py
```

Then open `http://localhost:8000/docs` for Swagger UI. Use `http://localhost:8000` as `processingApiUrl` when testing the app against local processing.
