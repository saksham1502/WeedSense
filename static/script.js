const dropZone    = document.getElementById("dropZone");
const fileInput   = document.getElementById("fileInput");
const dropInner   = document.getElementById("dropInner");
const preview     = document.getElementById("preview");
const btnRun      = document.getElementById("btnRun");
const btnClear    = document.getElementById("btnClear");
const resultEmpty = document.getElementById("resultEmpty");
const classResult = document.getElementById("classResult");
const errorBox    = document.getElementById("errorBox");
const labelBadge  = document.getElementById("labelBadge");
const confText    = document.getElementById("confText");
const progressBar = document.getElementById("progressBar");
const resultDesc  = document.getElementById("resultDesc");
const resultMeta  = document.getElementById("resultMeta");

let selectedFile = null;

// ── Clipboard paste support ──────────────────────────────────────────────────
document.addEventListener("paste", e => {
  const items = e.clipboardData?.items;
  if (!items) return;
  
  for (let i = 0; i < items.length; i++) {
    if (items[i].type.indexOf("image") !== -1) {
      const blob = items[i].getAsFile();
      if (blob) {
        loadFile(blob);
        e.preventDefault();
      }
      break;
    }
  }
});

// ── Drop zone ─────────────────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

function loadFile(file) {
  selectedFile = file;
  
  // For .tif files, show filename instead of broken preview
  if (file.name.toLowerCase().endsWith('.tif') || file.name.toLowerCase().endsWith('.tiff')) {
    dropInner.innerHTML = `
      <div class="drop-icon">📄</div>
      <p style="color: var(--text); font-weight: 600;">${file.name}</p>
      <span class="drop-hint">${(file.size / 1024).toFixed(1)} KB · Ready to detect</span>
    `;
    dropInner.style.display = "flex";
    preview.style.display = "none";
  } else {
    // For JPG/PNG, show preview
    const reader = new FileReader();
    reader.onload = e => {
      preview.src = e.target.result;
      preview.style.display = "block";
      dropInner.style.display = "none";
    };
    reader.readAsDataURL(file);
  }
  
  hideResults();
}

// ── Clear ─────────────────────────────────────────────────────────────────────
btnClear.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  preview.src = "";
  preview.style.display = "none";
  dropInner.innerHTML = `
    <div class="drop-icon">↑</div>
    <p>Drag &amp; drop or <span class="link">browse</span></p>
    <span class="drop-hint">JPG, PNG, TIF — any size</span>
  `;
  dropInner.style.display = "flex";
  hideResults();
});

// ── Run ───────────────────────────────────────────────────────────────────────
btnRun.addEventListener("click", async () => {
  if (!selectedFile) { showError("Please upload an image first."); return; }
  await runClassify();
});

async function runClassify() {
  showAnalyzing();
  const fd = new FormData();
  fd.append("image", selectedFile);

  try {
    const res  = await fetch("/predict/classify", { method: "POST", body: fd });
    const data = await res.json();

    if (data.error) { showError(data.error); return; }

    const isWeed = !data.is_crop;

    // Badge
    labelBadge.textContent = data.label;
    labelBadge.className   = "result-badge " + (isWeed ? "weed" : "crop");

    // Confidence
    confText.textContent = `${data.confidence}% confidence`;

    // Progress bar
    progressBar.style.width = data.confidence + "%";
    progressBar.className   = "progress-fill " + (isWeed ? "weed" : "crop");

    // Description
    resultDesc.textContent = isWeed
      ? "Weed or non-crop vegetation detected. Consider targeted herbicide application or manual removal."
      : "Soybean crop detected. Field appears healthy with no weed presence identified.";

    // Meta
    resultMeta.innerHTML = `
      <span class="meta-item">Raw probability: <strong>${data.raw_prob}</strong></span>
      <span class="meta-item">Threshold: <strong>0.5</strong></span>
      <span class="meta-item">Model: <strong>CNN 3-layer</strong></span>
    `;

    showSection(classResult);
  } catch (err) {
    showError("Request failed: " + err.message);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
let progressInterval = null;

function showAnalyzing() {
  // Hide other sections
  resultEmpty.style.display = "none";
  errorBox.style.display = "none";
  
  // Show analyzing state in result box
  classResult.style.display = "block";
  
  labelBadge.textContent = "Analyzing";
  labelBadge.className = "result-badge analyzing";
  
  confText.textContent = "Processing image...";
  
  // Animate progress bar from 0 to ~90%
  progressBar.style.width = "0%";
  progressBar.className = "progress-fill analyzing";
  
  let progress = 0;
  progressInterval = setInterval(() => {
    progress += Math.random() * 15;
    if (progress > 90) progress = 90;
    progressBar.style.width = progress + "%";
    confText.textContent = `${Math.floor(progress)}% analyzed`;
  }, 200);
  
  resultDesc.textContent = "Running CNN inference on uploaded image. Please wait...";
  resultMeta.innerHTML = `
    <span class="meta-item">Status: <strong>Processing</strong></span>
    <span class="meta-item">Model: <strong>CNN 3-layer</strong></span>
  `;
}

function hideResults() {
  if (progressInterval) {
    clearInterval(progressInterval);
    progressInterval = null;
  }
  resultEmpty.style.display  = "flex";
  classResult.style.display  = "none";
  errorBox.style.display     = "none";
}

function showSection(el) {
  if (progressInterval) {
    clearInterval(progressInterval);
    progressInterval = null;
  }
  resultEmpty.style.display = "none";
  classResult.style.display = "none";
  errorBox.style.display    = "none";
  el.style.display = "block";
}

function showError(msg) {
  if (progressInterval) {
    clearInterval(progressInterval);
    progressInterval = null;
  }
  resultEmpty.style.display = "none";
  classResult.style.display = "none";
  errorBox.style.display    = "block";
  errorBox.textContent      = "⚠️ " + msg;
}
