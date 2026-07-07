# AI Attendance System Using Face

A Flask-based student registration and attendance system that uses OpenCV face detection and recognition to mark live attendance. The app stores student details in CSV files, saves training images locally, performs basic liveness checks, sends optional email notifications, and can export attendance reports as PDF.

## Features

- Student registration with name, phone, email, password, camera capture, and image upload
- Face-based attendance using saved training images
- Liveness checks such as blink, mouth movement, motion, texture, and glare checks
- One attendance entry per student per day
- CSV storage for registrations, attendance, and email logs
- PDF attendance report downloads for all, weekly, or monthly attendance
- Mobile-friendly web UI with PWA manifest
- Optional HTTPS mode for camera access on mobile devices

## Project Structure

```text
.
|-- app.py                         # Main Flask application
|-- email_config.py                # Optional SMTP email settings
|-- generate_mobile_cert.py        # Creates a local HTTPS certificate
|-- run_mobile_https.bat           # Starts the app in HTTPS mobile mode
|-- registered_students.csv        # Registered student records
|-- attendance.csv                 # Attendance records
|-- attendance_email_log.csv       # Email notification log
|-- certs/                         # Generated HTTPS certificate files
|-- static/                        # CSS, JavaScript, service worker, manifest, icon
|-- templates/                     # HTML templates
`-- training_img/                  # Student face training images
```

## Requirements

- Python 3.10 or newer
- Webcam or phone camera
- Windows, macOS, or Linux

Python packages used by the app:

```bash
pip install -r requirements.txt
```

The project uses `opencv-contrib-python-headless` so OpenCV face recognizer support works on Render's Linux servers without desktop GUI dependencies.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the application:

```bash
python app.py
```

4. Open the app in your browser:

```text
http://localhost:5000
```

## Mobile HTTPS Mode

Mobile browsers often require HTTPS for camera access. To start the app with a local HTTPS certificate on Windows, run:

```bash
run_mobile_https.bat
```

The script generates a certificate and starts Flask on port `5443`. It prints the local network URL, for example:

```text
https://192.168.1.10:5443
```

Open that URL from a phone connected to the same Wi-Fi network. The browser may show a certificate warning because the certificate is locally generated.

## Configuration

The app supports these environment variables:

```text
ATTENDANCE_HOST              Flask host, default 0.0.0.0
ATTENDANCE_PORT              Flask port, default 5000 or 5443 for HTTPS
ATTENDANCE_HTTPS             Set to 1/true/yes/on to enable HTTPS
ATTENDANCE_CERT              Certificate file path
ATTENDANCE_KEY               Private key file path
ATTENDANCE_LBPH_THRESHOLD    Face match threshold, default 90
ATTENDANCE_SMTP_HOST         SMTP server host
ATTENDANCE_SMTP_PORT         SMTP server port
ATTENDANCE_SMTP_USER         SMTP username
ATTENDANCE_SMTP_PASSWORD     SMTP password or app password
ATTENDANCE_SMTP_FROM         Sender email address
```

Email settings can also be placed in `email_config.py`, but environment variables are safer for passwords and deployment.

## Deploying on Render

1. Push the project to GitHub.
2. Create a new Render Web Service from the GitHub repository.
3. Use these settings:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app --timeout 120
```

4. Add email settings in Render Dashboard > Environment if you want email notifications.
5. Deploy the service.

Render provides the `PORT` environment variable automatically. The app is ready for that, and `Procfile` is included for platforms that read it.

## Usage

1. Register a student from the registration section.
2. Capture a clear face image or upload training images.
3. Use the live attendance section to detect the registered face.
4. Complete the liveness prompts shown by the app.
5. Attendance is saved to `attendance.csv`.
6. Download reports from the PDF links in the header.

## Data Files

- `registered_students.csv` stores registration details and password hashes.
- `attendance.csv` stores attendance records.
- `attendance_email_log.csv` stores email delivery results.
- `training_img/` stores student image folders used for recognition.

## Notes

- Keep training images clear, front-facing, and well lit for better recognition.
- Do not share real SMTP passwords or app passwords publicly.
- CSV files and training images contain student data, so handle them carefully.
- Debug mode is disabled by default. Set `FLASK_DEBUG=1` only for local development.
