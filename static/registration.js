const camera = document.getElementById("camera");
const canvas = document.getElementById("snapshotCanvas");
const preview = document.getElementById("snapshotPreview");
const capturedImage = document.getElementById("capturedImage");
const startButton = document.getElementById("startCamera");
const captureButton = document.getElementById("capturePhoto");
const fallback = document.getElementById("cameraFallback");
const fileInput = document.getElementById("trainingImages");
const fileSummary = document.getElementById("fileSummary");
const attendanceCamera = document.getElementById("attendanceCamera");
const attendanceCanvas = document.getElementById("attendanceCanvas");
const attendanceFallback = document.getElementById("attendanceFallback");
const startAttendanceButton = document.getElementById("startAttendance");
const stopAttendanceButton = document.getElementById("stopAttendance");
const attendanceStatus = document.getElementById("attendanceStatus");
const faceCheck = document.getElementById("faceCheck");
const blinkCheck = document.getElementById("blinkCheck");
const mouthCheck = document.getElementById("mouthCheck");
const realFaceCheck = document.getElementById("realFaceCheck");

let stream;
let attendanceStream;
let attendanceTimer;
let attendanceBusy = false;
let lastVoiceMessage = "";
let lastDetectedStudent = "";
let attendanceSessionId = "";

function isCoarsePointer() {
  return window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
}

function stopStream(mediaStream) {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }
}

function isCameraSupported() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function cameraConstraints() {
  const mobile = isCoarsePointer();
  return {
    video: {
      facingMode: "user",
      width: { ideal: mobile ? 640 : 960 },
      height: { ideal: mobile ? 854 : 720 },
      frameRate: { ideal: mobile ? 15 : 24, max: 30 }
    },
    audio: false
  };
}

async function openUserCamera() {
  if (!isCameraSupported()) {
    throw new Error("camera_not_supported");
  }
  try {
    return await navigator.mediaDevices.getUserMedia(cameraConstraints());
  } catch (error) {
    return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  }
}

function cameraHelpText() {
  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
    return `Mobile camera needs HTTPS. Restart with run_mobile_https.bat, then open https://${location.hostname}:5443 on the phone and allow camera access.`;
  }
  return "Allow camera access from the browser permission popup and try again.";
}

function attachStream(videoElement, mediaStream) {
  videoElement.setAttribute("playsinline", "");
  videoElement.playsInline = true;
  videoElement.muted = true;
  videoElement.srcObject = mediaStream;
}

async function startCamera() {
  try {
    stopAttendance(true);
    stopStream(stream);
    stream = await openUserCamera();
    attachStream(camera, stream);
    fallback.hidden = true;
  } catch (error) {
    fallback.textContent = `Camera unavailable. ${cameraHelpText()} You can still upload training images below.`;
    fallback.hidden = false;
  }
}

function capturePhoto() {
  if (!camera.videoWidth) {
    fallback.textContent = "Start the camera before capturing.";
    fallback.hidden = false;
    return;
  }

  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(camera, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL("image/jpeg", 0.95);
  capturedImage.value = imageData;
  preview.src = imageData;
  preview.hidden = false;
}

startButton.addEventListener("click", startCamera);
captureButton.addEventListener("click", capturePhoto);

fileInput.addEventListener("change", () => {
  const count = fileInput.files.length;
  fileSummary.textContent = count ? `${count} file${count === 1 ? "" : "s"} selected` : "PNG, JPG, JPEG, or WEBP files";
});

function setAttendanceStatus(message, detail = "", tone = "waiting") {
  attendanceStatus.className = `status-card ${tone}`;
  attendanceStatus.innerHTML = `<strong>${message}</strong><span>${detail}</span>`;
}

function setCheck(element, active) {
  element.classList.toggle("done", Boolean(active));
}

function formatMatchDetail(result) {
  if (result.match_distance === null || result.match_distance === undefined) {
    return "";
  }
  return ` Match distance ${result.match_distance}/${result.match_threshold}.`;
}

function speak(message) {
  if (!("speechSynthesis" in window) || !message || message === lastVoiceMessage) {
    return;
  }
  lastVoiceMessage = message;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = 0.95;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}

async function startAttendance() {
  try {
    attendanceSessionId = window.crypto && window.crypto.randomUUID ? window.crypto.randomUUID() : String(Date.now());
    lastDetectedStudent = "";
    lastVoiceMessage = "";
    [faceCheck, blinkCheck, mouthCheck, realFaceCheck].forEach((element) => setCheck(element, false));
    stopStream(stream);
    stream = null;
    camera.srcObject = null;
    stopStream(attendanceStream);
    attendanceStream = await openUserCamera();
    attachStream(attendanceCamera, attendanceStream);
    attendanceFallback.hidden = true;
    setAttendanceStatus("Looking for registered face", "After match: keep eyes visible and move your lips. Attendance is marked immediately when both checks pass.");
    clearInterval(attendanceTimer);
    attendanceTimer = setInterval(sendAttendanceFrame, 200);
  } catch (error) {
    attendanceFallback.textContent = `Camera unavailable. ${cameraHelpText()}`;
    attendanceFallback.hidden = false;
    setAttendanceStatus("Camera unavailable", "Allow camera access and try again.", "error");
  }
}

function stopAttendance(silent = false) {
  clearInterval(attendanceTimer);
  attendanceTimer = null;
  attendanceBusy = false;
  stopStream(attendanceStream);
  attendanceStream = null;
  attendanceCamera.srcObject = null;
  if (!silent) {
    setAttendanceStatus("Live attendance stopped", "Start again when the student is ready.");
  }
}

function finishAttendanceSession() {
  clearInterval(attendanceTimer);
  attendanceTimer = null;
  attendanceBusy = false;
  stopStream(attendanceStream);
  attendanceStream = null;
  attendanceCamera.srcObject = null;
}

async function sendAttendanceFrame() {
  if (attendanceBusy || !attendanceCamera.videoWidth) {
    return;
  }

  attendanceBusy = true;
  attendanceCanvas.width = attendanceCamera.videoWidth;
  attendanceCanvas.height = attendanceCamera.videoHeight;
  const context = attendanceCanvas.getContext("2d");
  context.drawImage(attendanceCamera, 0, 0, attendanceCanvas.width, attendanceCanvas.height);
  const image = attendanceCanvas.toDataURL("image/jpeg", 0.82);

  try {
    const response = await fetch("/attendance-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: attendanceSessionId, image })
    });
    const result = await response.json();
    updateAttendanceUi(result);
  } catch (error) {
    setAttendanceStatus("Live check failed", "Server did not respond. Keep the app running and try again.", "error");
  } finally {
    attendanceBusy = false;
  }
}

function updateAttendanceUi(result) {
  setCheck(faceCheck, result.status === "live_check" || result.status === "marked" || result.status === "already_marked");
  setCheck(blinkCheck, result.blink_done);
  setCheck(mouthCheck, result.mouth_done);
  setCheck(realFaceCheck, result.live_confirmed || result.status === "marked" || result.status === "already_marked");

  if (result.status === "marked") {
    const emailDetail = result.email_message ? ` ${result.email_message}` : "";
    setAttendanceStatus(result.message, `Saved successfully in the attendance CSV.${emailDetail}`, "success");
    speak(`Hi ${result.student_name}. You are present. Attendance marked.`);
    finishAttendanceSession();
    return;
  }

  if (result.status === "already_marked") {
    setAttendanceStatus(result.message, `${result.email_message || "Attendance was already saved today."} Detection stopped.`, "success");
    speak(`Hi ${result.student_name}. Attendance already taken today.`);
    finishAttendanceSession();
    return;
  }

  if (result.status === "spoof_suspected") {
    setAttendanceStatus(result.message, `Eye blink and lip movement were not confirmed quickly. Absent saved.${result.email_message ? ` ${result.email_message}` : ""}`, "error");
    speak(`${result.student_name} matched, but blink and lip movement were not confirmed. Absent marked.`);
    finishAttendanceSession();
    return;
  }

  if (result.status === "unregistered") {
    setAttendanceStatus(result.message, result.email_message || "Attendance was not saved.", "error");
    speak(`${result.student_name} is not registered. Attendance not saved.`);
    finishAttendanceSession();
    return;
  }

  if (result.status === "session_completed") {
    setAttendanceStatus(result.message, "Detection stopped. Press Start live attendance for the next person.", "success");
    finishAttendanceSession();
    return;
  }

  if (result.status === "unknown" || result.status === "no_face") {
    lastDetectedStudent = "";
    setAttendanceStatus(result.message, `Use a clear front-facing view in good light.${formatMatchDetail(result)}`, "error");
    if (result.status === "unknown") {
      speak("Unknown person. Absent. This record is not stored.");
    }
    return;
  }

  if (result.status === "confirming") {
    lastDetectedStudent = "";
    setAttendanceStatus(
      result.message,
      `Checking stable identity ${result.stable_count}/${result.stable_required}.${formatMatchDetail(result)}`,
      "waiting"
    );
    return;
  }

  const detail = result.student_name
    ? `Confidence ${result.confidence}%.${formatMatchDetail(result)} Quick check ${result.liveness_frames || 0}/${result.liveness_limit || ""}. Eye blink: ${result.blink_done ? "detected" : "checking"} (${result.eye_motion || 0}). Lip movement: ${result.mouth_done ? "detected" : "checking"} (${result.mouth_motion || 0}). Keep the face steady.`
    : "";
  if (result.student_name && result.student_name !== lastDetectedStudent) {
    lastDetectedStudent = result.student_name;
    speak(`Hi ${result.student_name}. Face detected. Complete liveness check for attendance.`);
  }
  setAttendanceStatus(result.message || "Checking liveness", detail);
}

startAttendanceButton.addEventListener("click", startAttendance);
stopAttendanceButton.addEventListener("click", stopAttendance);

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    stopAttendance(true);
    stopStream(stream);
    stream = null;
    camera.srcObject = null;
  }
});

window.addEventListener("beforeunload", () => {
  stopAttendance(true);
  stopStream(stream);
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/static/sw.js").catch(() => {});
  });
}
