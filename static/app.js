const form = document.getElementById('uploadForm');
const input = document.getElementById('imageInput');
const dropzone = document.getElementById('dropzone');
const uploadHint = document.getElementById('uploadHint');
const statusText = document.getElementById('statusText');
const modal = document.getElementById('resultModal');
const closeModalBtn = document.getElementById('closeModal');
const retryBtn = document.getElementById('retryBtn');

const uploadPanel = document.getElementById('uploadPanel');
const cameraPanel = document.getElementById('cameraPanel');
const uploadModeBtn = document.getElementById('uploadModeBtn');
const cameraModeBtn = document.getElementById('cameraModeBtn');

const startCameraBtn = document.getElementById('startCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const retakeBtn = document.getElementById('retakeBtn');
const cameraPreview = document.getElementById('cameraPreview');
const cameraCanvas = document.getElementById('cameraCanvas');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const cameraHint = document.getElementById('cameraHint');

const previewImage = document.getElementById('previewImage');
const plateNumber = document.getElementById('plateNumber');
const province = document.getElementById('province');
const vehicleType = document.getElementById('vehicleType');
const plateColor = document.getElementById('plateColor');
const usage = document.getElementById('usage');
const confidence = document.getElementById('confidence');

const supportsCamera = Boolean(navigator.mediaDevices?.getUserMedia);

let currentMode = 'upload';
let selectedFile = null;
let selectedSource = null;
let cameraStream = null;

function openModal() {
  modal.classList.remove('hidden');
}

function closeModal() {
  modal.classList.add('hidden');
}

function setStatus(text, isError = false) {
  statusText.textContent = text;
  statusText.style.color = isError ? '#bb2e2e' : '#5d7598';
}

function updatePreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function syncInputFiles(file) {
  if (!file) {
    input.value = '';
    return;
  }

  if (typeof DataTransfer === 'undefined') {
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
}

function bindFile(file, source) {
  if (!file) return;

  selectedFile = file;
  selectedSource = source;
  syncInputFiles(file);
  updatePreview(file);

  if (source === 'camera') {
    uploadHint.textContent = `ภาพล่าสุดจากกล้อง: ${file.name}`;
    setStatus('ถ่ายภาพแล้ว พร้อมตรวจสอบ');
    return;
  }

  uploadHint.textContent = `ไฟล์ที่เลือก: ${file.name}`;
  setStatus('พร้อมตรวจสอบแล้ว');
}

function clearSelection(source = null) {
  if (source && selectedSource !== source) {
    return;
  }

  selectedFile = null;
  selectedSource = null;
  input.value = '';
}

function getActiveFile() {
  if (!selectedFile || selectedSource !== currentMode) {
    return null;
  }

  return selectedFile;
}

function refreshUploadHint() {
  if (selectedSource === 'upload' && selectedFile) {
    uploadHint.textContent = `ไฟล์ที่เลือก: ${selectedFile.name}`;
    return;
  }

  uploadHint.textContent = 'รองรับ JPG, PNG, WEBP หรือคลิกเพื่อเลือกรูป';
}

function showCameraIdle() {
  cameraPlaceholder.classList.remove('hidden');
  cameraPreview.classList.add('hidden');
  cameraCanvas.classList.add('hidden');
  startCameraBtn.classList.remove('hidden');
  captureBtn.classList.add('hidden');
  retakeBtn.classList.add('hidden');
}

function showCameraLive() {
  cameraPlaceholder.classList.add('hidden');
  cameraPreview.classList.remove('hidden');
  cameraCanvas.classList.add('hidden');
  startCameraBtn.classList.add('hidden');
  captureBtn.classList.remove('hidden');
  retakeBtn.classList.add('hidden');
}

function showCameraCaptured() {
  cameraPlaceholder.classList.add('hidden');
  cameraPreview.classList.add('hidden');
  cameraCanvas.classList.remove('hidden');
  startCameraBtn.classList.add('hidden');
  captureBtn.classList.add('hidden');
  retakeBtn.classList.remove('hidden');
}

function stopCamera() {
  if (!cameraStream) return;

  cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
}

function setMode(mode) {
  currentMode = mode;

  const usingCamera = mode === 'camera';
  uploadModeBtn.classList.toggle('active', !usingCamera);
  cameraModeBtn.classList.toggle('active', usingCamera);
  uploadPanel.classList.toggle('hidden', usingCamera);
  cameraPanel.classList.toggle('hidden', !usingCamera);

  if (usingCamera) {
    if (!supportsCamera) {
      cameraModeBtn.disabled = true;
      cameraHint.textContent = 'เบราว์เซอร์นี้ไม่รองรับการเปิดกล้อง ให้ใช้อัปโหลดไฟล์แทน';
      showCameraIdle();
      setStatus('เบราว์เซอร์นี้ไม่รองรับการใช้งานกล้อง', true);
      return;
    }

    if (selectedSource === 'camera') {
      showCameraCaptured();
      setStatus('พร้อมตรวจสอบภาพจากกล้องแล้ว');
      return;
    }

    showCameraIdle();
    setStatus('เปิดกล้องเพื่อถ่ายภาพ หรือสลับกลับไปอัปโหลดไฟล์');
    return;
  }

  stopCamera();
  refreshUploadHint();
  if (selectedSource === 'upload' && selectedFile) {
    setStatus('พร้อมตรวจสอบแล้ว');
  } else {
    setStatus('');
  }
}

async function startCamera() {
  if (!supportsCamera) {
    setStatus('เบราว์เซอร์นี้ไม่รองรับการใช้งานกล้อง', true);
    return;
  }

  stopCamera();
  setStatus('กำลังเปิดกล้อง...');
  startCameraBtn.disabled = true;

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: 'environment' },
      },
      audio: false,
    });
    cameraPreview.srcObject = cameraStream;
    await cameraPreview.play();
    showCameraLive();
    setStatus('จัดป้ายให้อยู่กลางภาพแล้วกดถ่าย');
  } catch (error) {
    showCameraIdle();
    setStatus('ไม่สามารถเปิดกล้องได้ กรุณาอนุญาตการใช้งานกล้องหรือใช้อัปโหลดไฟล์แทน', true);
  } finally {
    startCameraBtn.disabled = false;
  }
}

function capturePhoto() {
  if (!cameraStream || !cameraPreview.videoWidth || !cameraPreview.videoHeight) {
    setStatus('กล้องยังไม่พร้อมสำหรับการถ่ายภาพ', true);
    return;
  }

  cameraCanvas.width = cameraPreview.videoWidth;
  cameraCanvas.height = cameraPreview.videoHeight;

  const ctx = cameraCanvas.getContext('2d');
  ctx.drawImage(cameraPreview, 0, 0, cameraCanvas.width, cameraCanvas.height);

  cameraCanvas.toBlob((blob) => {
    if (!blob) {
      setStatus('ไม่สามารถสร้างภาพจากกล้องได้', true);
      return;
    }

    const file = new File([blob], `camera-capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
    bindFile(file, 'camera');
    stopCamera();
    showCameraCaptured();
  }, 'image/jpeg', 0.92);
}

async function retakePhoto() {
  clearSelection('camera');
  refreshUploadHint();
  await startCamera();
}

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const [file] = e.dataTransfer.files;
  bindFile(file, 'upload');
});

input.addEventListener('change', () => {
  const [file] = input.files;
  bindFile(file, 'upload');
});

uploadModeBtn.addEventListener('click', () => {
  setMode('upload');
});

cameraModeBtn.addEventListener('click', () => {
  setMode('camera');
});

startCameraBtn.addEventListener('click', () => {
  startCamera();
});

captureBtn.addEventListener('click', () => {
  capturePhoto();
});

retakeBtn.addEventListener('click', () => {
  retakePhoto();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const submitBtn = form.querySelector('button[type="submit"]');
  const activeFile = getActiveFile();

  if (!activeFile) {
    setStatus('กรุณาเลือกไฟล์ภาพหรือถ่ายภาพก่อนเริ่มตรวจสอบ', true);
    return;
  }

  submitBtn.disabled = true;
  submitBtn.classList.add('loading');
  setStatus('กำลังวิเคราะห์ป้ายทะเบียน...');

  const formData = new FormData();
  formData.append('image', activeFile, activeFile.name);

  try {
    const response = await fetch('/api/detect', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || 'ไม่สามารถตรวจสอบรูปได้');
    }

    plateNumber.textContent = data.plate_number || '-';
    province.textContent = data.province || '-';
    vehicleType.textContent = data.vehicle_type || '-';
    plateColor.textContent = data.plate_color || '-';
    usage.textContent = data.usage || '-';
    confidence.textContent = data.confidence ?? 0;

    openModal();
    setStatus('ตรวจสอบสำเร็จ');
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove('loading');
  }
});

closeModalBtn.addEventListener('click', closeModal);
retryBtn.addEventListener('click', closeModal);
modal.addEventListener('click', (e) => {
  if (e.target === modal) closeModal();
});

window.addEventListener('beforeunload', () => {
  stopCamera();
});

if (!supportsCamera) {
  cameraModeBtn.disabled = true;
}

showCameraIdle();
refreshUploadHint();
setMode('upload');
