@echo off
python generate_mobile_cert.py
set ATTENDANCE_HTTPS=1
set ATTENDANCE_HOST=0.0.0.0
set ATTENDANCE_PORT=5443
set ATTENDANCE_CERT=certs\mobile-local.crt
set ATTENDANCE_KEY=certs\mobile-local.key
python app.py
