/**
 * Auslan Sign Language Recognition - Frontend Application Logic
 * Problem 118 | UE24CS352A Machine Learning Mini-Project
 */

// Global State
const state = {
  currentResult: null,
  signsList: [],
  signersList: [],
  featuredSigns: [],
  activeChannelGroup: "all",
  three: {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    trajectoryLine: null,
    handMarker: null,
    points: [],
    animationId: null,
    isPlaying: false,
    currentFrame: 0,
    speed: 1.0,
    lastTime: 0,
  },
  sensorChart: null,
  ablationChart: null,
};

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) {
    lucide.createIcons();
  }

  setupTabs();
  setupThreeJS();
  setupEventListeners();

  // Load initial data from API
  await fetchAppStatus();
  await fetchSignsData();
  await fetchBenchmarksData();

  // Load a default random sample to showcase immediate interactive beauty
  await loadRandomSample();
});

// =========================================================================
// TAB NAVIGATION
// =========================================================================
function setupTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => {
        t.classList.remove("active");
        t.classList.add("text-slate-400");
      });
      tab.classList.add("active");
      tab.classList.remove("text-slate-400");

      const targetId = tab.dataset.tab;
      document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.add("hidden");
      });

      const activePane = document.getElementById(targetId);
      if (activePane) {
        activePane.classList.remove("hidden");
      }

      // Re-trigger layout for Three.js and charts
      if (targetId === "tab-classifier") {
        onWindowResizeThree();
      } else if (targetId === "tab-ablation" && state.ablationChart) {
        state.ablationChart.resize();
      }
    });
  });
}

// =========================================================================
// API CALLS
// =========================================================================
async function fetchAppStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    console.log("System status:", data);
  } catch (err) {
    console.error("Failed to fetch status:", err);
  }
}

async function fetchSignsData() {
  try {
    const res = await fetch("/api/signs");
    const data = await res.json();
    state.signsList = data.signs || [];
    state.signersList = data.signers || [];
    state.featuredSigns = data.featured || [];

    populateSignSelects();
    populateFeaturedSigns();
    populateSignsDictionary(state.signsList);
  } catch (err) {
    console.error("Failed to load signs data:", err);
  }
}

async function fetchBenchmarksData() {
  try {
    const res = await fetch("/api/benchmarks");
    const data = await res.json();
    renderClassicalTable(data.classical_models);
    renderRecurrentTable(data.recurrent_models);
    renderAblationSingleTable(data.ablation_single);
    renderAblationCumulativeTable(data.ablation_cumulative);
    renderConfusedPairsTable(data.confused_pairs);
    renderAblationChart(data.ablation_single);
  } catch (err) {
    console.error("Failed to load benchmarks:", err);
  }
}

async function loadSampleFile(filePath) {
  try {
    setLoadingState(true);
    const res = await fetch(`/api/predict-sample?file=${encodeURIComponent(filePath)}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    displayPredictionResult(data);
  } catch (err) {
    alert("Error classifying sample: " + err.message);
  } finally {
    setLoadingState(false);
  }
}

async function loadRandomSample() {
  try {
    setLoadingState(true);
    const res = await fetch("/api/random");
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    displayPredictionResult(data);
  } catch (err) {
    console.error("Error fetching random sign:", err);
  } finally {
    setLoadingState(false);
  }
}

// =========================================================================
// UI POPULATION & RENDERING
// =========================================================================
function populateSignSelects() {
  const signSelect = document.getElementById("signSelect");
  const signerSelect = document.getElementById("signerSelect");

  signSelect.innerHTML = '<option value="">-- Choose Sign --</option>';
  state.signsList.forEach((sign) => {
    const opt = document.createElement("option");
    opt.value = sign;
    opt.textContent = sign;
    signSelect.appendChild(opt);
  });

  signerSelect.innerHTML = '<option value="">Any Signer</option>';
  state.signersList.forEach((signer) => {
    const opt = document.createElement("option");
    opt.value = signer;
    opt.textContent = signer;
    signerSelect.appendChild(opt);
  });
}

function populateFeaturedSigns() {
  const container = document.getElementById("featuredSignsContainer");
  container.innerHTML = "";

  state.featuredSigns.slice(0, 14).forEach((sign) => {
    const btn = document.createElement("button");
    btn.className =
      "text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700/80 px-2.5 py-1 rounded-md transition-all";
    btn.textContent = sign;
    btn.addEventListener("click", () => {
      document.getElementById("signSelect").value = sign;
      handleSignSelection(sign);
    });
    container.appendChild(btn);
  });
}

function populateSignsDictionary(signs) {
  const grid = document.getElementById("signsGrid");
  grid.innerHTML = "";

  signs.forEach((s) => {
    const item = document.createElement("div");
    item.className =
      "bg-slate-900/80 hover:bg-blue-900/30 border border-slate-800 hover:border-blue-600/50 p-2 rounded-lg flex items-center justify-between cursor-pointer transition-all group";
    item.innerHTML = `
      <span class="text-xs font-semibold text-slate-200 group-hover:text-blue-400 truncate">${s}</span>
      <i data-lucide="chevron-right" class="w-3.5 h-3.5 text-slate-500 group-hover:text-blue-400"></i>
    `;
    item.addEventListener("click", () => {
      // Switch to classifier tab and load this sign
      const classifierTab = document.querySelector('[data-tab="tab-classifier"]');
      if (classifierTab) classifierTab.click();
      document.getElementById("signSelect").value = s;
      handleSignSelection(s);
    });
    grid.appendChild(item);
  });

  if (window.lucide) lucide.createIcons();
}

async function handleSignSelection(sign) {
  const signer = document.getElementById("signerSelect").value;
  const sampleSelect = document.getElementById("sampleSelect");
  const btnClassify = document.getElementById("btnClassifySample");

  if (!sign) {
    sampleSelect.innerHTML = '<option value="">Select a sign first</option>';
    sampleSelect.disabled = true;
    btnClassify.disabled = true;
    return;
  }

  sampleSelect.innerHTML = '<option value="">Loading samples...</option>';
  sampleSelect.disabled = true;

  try {
    let url = `/api/sample-files?sign=${encodeURIComponent(sign)}`;
    if (signer) url += `&signer=${encodeURIComponent(signer)}`;

    const res = await fetch(url);
    const data = await res.json();

    if (data.samples && data.samples.length > 0) {
      sampleSelect.innerHTML = "";
      data.samples.forEach((sample) => {
        const opt = document.createElement("option");
        opt.value = sample.path;
        opt.textContent = `${sample.signer} - ${sample.filename}`;
        sampleSelect.appendChild(opt);
      });
      sampleSelect.disabled = false;
      btnClassify.disabled = false;

      // Automatically classify the first sample
      loadSampleFile(data.samples[0].path);
    } else {
      sampleSelect.innerHTML = '<option value="">No samples for this signer</option>';
    }
  } catch (err) {
    console.error("Error loading sample files:", err);
  }
}

// =========================================================================
// DISPLAY PREDICTION & CHARTS
// =========================================================================
function displayPredictionResult(data) {
  state.currentResult = data;

  // 1. Predicted Label & Confidence
  const predLabel = document.getElementById("predLabelText");
  const predConfidenceBadge = document.getElementById("predConfidenceBadge");
  const matchStatusBadge = document.getElementById("matchStatusBadge");
  const trueLabelText = document.getElementById("trueLabelText");
  const durationFramesText = document.getElementById("durationFramesText");
  const sourceFileText = document.getElementById("sourceFileText");

  predLabel.textContent = data.predicted_label;
  const confPct = (data.confidence * 100).toFixed(1);
  predConfidenceBadge.textContent = `${confPct}% Confidence`;

  // Color code confidence badge
  if (data.confidence >= 0.6) {
    predConfidenceBadge.className =
      "badge-tag bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
  } else if (data.confidence >= 0.35) {
    predConfidenceBadge.className =
      "badge-tag bg-amber-500/20 text-amber-400 border border-amber-500/30";
  } else {
    predConfidenceBadge.className =
      "badge-tag bg-rose-500/20 text-rose-400 border border-rose-500/30";
  }

  // True Label & Match Status
  if (data.true_label) {
    trueLabelText.textContent = data.true_label;
    matchStatusBadge.classList.remove("hidden");
    if (data.match) {
      matchStatusBadge.innerHTML = `
        <span class="badge-tag bg-emerald-600/20 text-emerald-300 border border-emerald-500/40 flex items-center space-x-1.5 py-1 px-3">
          <i data-lucide="check-circle" class="w-3.5 h-3.5 text-emerald-400"></i>
          <span>MATCH [SUCCESS]</span>
        </span>
      `;
    } else {
      matchStatusBadge.innerHTML = `
        <span class="badge-tag bg-amber-600/20 text-amber-300 border border-amber-500/40 flex items-center space-x-1.5 py-1 px-3">
          <i data-lucide="alert-circle" class="w-3.5 h-3.5 text-amber-400"></i>
          <span>MISMATCH</span>
        </span>
      `;
    }
  } else {
    trueLabelText.textContent = "Unknown (Custom Input)";
    matchStatusBadge.classList.add("hidden");
  }

  durationFramesText.textContent = `${data.raw_frames} frames (~${data.duration_sec}s @ 50 Hz)`;
  sourceFileText.textContent = data.source || "Uploaded Sample";

  // 2. Render Top-5 Probabilities Breakdown
  const topList = document.getElementById("topProbList");
  topList.innerHTML = "";

  data.top_predictions.forEach((item, idx) => {
    const pct = (item.probability * 100).toFixed(1);
    const isTop = idx === 0;

    const row = document.createElement("div");
    row.className = "flex items-center text-xs space-x-3";
    row.innerHTML = `
      <span class="font-mono text-slate-500 w-4">${item.rank}.</span>
      <span class="w-24 truncate font-medium ${isTop ? "text-white font-semibold" : "text-slate-300"}">${item.label}</span>
      <div class="flex-1 bg-slate-800/80 rounded-full h-2.5 overflow-hidden border border-slate-700/50">
        <div class="prob-bar h-full rounded-full ${isTop ? "bg-gradient-to-r from-blue-500 to-indigo-500" : "bg-slate-600"}" style="width: ${pct}%"></div>
      </div>
      <span class="font-mono text-right w-12 ${isTop ? "text-blue-400 font-bold" : "text-slate-400"}">${pct}%</span>
    `;
    topList.appendChild(row);
  });

  // 3. Render 3D Trajectory in Three.js
  if (data.trajectory_3d && data.trajectory_3d.length > 0) {
    updateThreeTrajectory(data.trajectory_3d);
  }

  // 4. Render Sensor Dynamics Chart
  if (data.channels) {
    updateSensorChart(data.channels);
  }

  if (window.lucide) lucide.createIcons();
}

function setLoadingState(isLoading) {
  const btnClassify = document.getElementById("btnClassifySample");
  const btnRandom = document.getElementById("btnRandomSample");
  if (isLoading) {
    btnRandom.classList.add("opacity-50", "pointer-events-none");
    if (btnClassify) btnClassify.classList.add("opacity-50", "pointer-events-none");
  } else {
    btnRandom.classList.remove("opacity-50", "pointer-events-none");
    if (btnClassify && !document.getElementById("sampleSelect").disabled) {
      btnClassify.classList.remove("opacity-50", "pointer-events-none");
    }
  }
}

// =========================================================================
// THREE.JS 3D TRAJECTORY VISUALIZER
// =========================================================================
function setupThreeJS() {
  const container = document.getElementById("trajectoryCanvasContainer");
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0c1322);

  // Camera
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(2.5, 2.5, 3.5);

  // Renderer
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);
  renderer.domElement.id = "threeCanvas";

  // OrbitControls
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0.5, 0.5, 0.5);

  // Ambient & Directional Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0x60a5fa, 0.8);
  dirLight.position.set(5, 10, 7);
  scene.add(dirLight);

  // Bounding coordinate grid box [0, 1] for normalized space
  const gridHelper = new THREE.GridHelper(1.2, 10, 0x3b82f6, 0x1e293b);
  gridHelper.position.set(0.5, 0, 0.5);
  scene.add(gridHelper);

  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const boxEdges = new THREE.EdgesGeometry(boxGeo);
  const boxLine = new THREE.LineSegments(
    boxEdges,
    new THREE.LineBasicMaterial({ color: 0x1e293b, transparent: true, opacity: 0.6 })
  );
  boxLine.position.set(0.5, 0.5, 0.5);
  scene.add(boxLine);

  // Axes indicator at origin
  const axesHelper = new THREE.AxesHelper(0.3);
  axesHelper.position.set(0, 0, 0);
  scene.add(axesHelper);

  // Animated Hand Marker (Sphere + Orienting Ring)
  const markerGroup = new THREE.Group();

  const sphereGeo = new THREE.SphereGeometry(0.04, 16, 16);
  const sphereMat = new THREE.MeshStandardMaterial({
    color: 0x60a5fa,
    emissive: 0x2563eb,
    emissiveIntensity: 0.6,
    roughness: 0.2,
  });
  const sphere = new THREE.Mesh(sphereGeo, sphereMat);
  markerGroup.add(sphere);

  // Ring to show wrist roll
  const ringGeo = new THREE.TorusGeometry(0.07, 0.012, 8, 24);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0xa855f7 });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;
  markerGroup.add(ring);

  markerGroup.visible = false;
  scene.add(markerGroup);

  // Save to state
  state.three.scene = scene;
  state.three.camera = camera;
  state.three.renderer = renderer;
  state.three.controls = controls;
  state.three.handMarker = markerGroup;

  // Window resize handler
  window.addEventListener("resize", onWindowResizeThree);

  // Render loop
  function animate(timestamp) {
    requestAnimationFrame(animate);
    controls.update();

    if (state.three.isPlaying && state.three.points.length > 0) {
      if (!state.three.lastTime) state.three.lastTime = timestamp;
      const delta = timestamp - state.three.lastTime;
      const frameInterval = 1000 / (25 * state.three.speed); // base ~25fps playback

      if (delta >= frameInterval) {
        state.three.lastTime = timestamp;
        let nextFrame = state.three.currentFrame + 1;
        if (nextFrame >= state.three.points.length) {
          nextFrame = 0; // loop
        }
        setThreeFrame(nextFrame);
      }
    }

    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);
}

function onWindowResizeThree() {
  const container = document.getElementById("trajectoryCanvasContainer");
  if (!container || !state.three.renderer) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  state.three.camera.aspect = width / height;
  state.three.camera.updateProjectionMatrix();
  state.three.renderer.setSize(width, height);
}

function updateThreeTrajectory(trajectoryPoints) {
  state.three.points = trajectoryPoints;

  // Hide the initial prompt overlay
  const overlay = document.getElementById("canvasOverlayPrompt");
  if (overlay) overlay.classList.add("hidden");

  // Remove existing trajectory line
  if (state.three.trajectoryLine) {
    state.three.scene.remove(state.three.trajectoryLine);
    state.three.trajectoryLine.geometry.dispose();
    state.three.trajectoryLine.material.dispose();
  }

  // Build 3D curve vertices
  const vectors = trajectoryPoints.map((pt) => new THREE.Vector3(pt.x, pt.y, pt.z));
  const curve = new THREE.CatmullRomCurve3(vectors);
  const points = curve.getPoints(120);

  // Tube geometry for a thick, glowing 3D trajectory
  const tubeGeo = new THREE.TubeGeometry(curve, 100, 0.015, 8, false);

  // Vertex colors along the trajectory (start green -> end purple)
  const colors = [];
  const count = tubeGeo.attributes.position.count;
  for (let i = 0; i < count; i++) {
    const t = i / count;
    // interpolate color
    const c = new THREE.Color().lerpColors(new THREE.Color(0x10b981), new THREE.Color(0x8b5cf6), t);
    colors.push(c.r, c.g, c.b);
  }
  tubeGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const tubeMat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.3,
    metalness: 0.1,
  });

  const trajectoryMesh = new THREE.Mesh(tubeGeo, tubeMat);
  state.three.scene.add(trajectoryMesh);
  state.three.trajectoryLine = trajectoryMesh;

  // Enable controls
  document.getElementById("btnTrajectoryPlay").disabled = false;
  document.getElementById("btnTrajectoryReset").disabled = false;
  document.getElementById("trajectoryScrubber").disabled = false;
  document.getElementById("trajectoryScrubber").max = trajectoryPoints.length - 1;

  // Set initial frame
  setThreeFrame(0);
}

function setThreeFrame(frameIndex) {
  state.three.currentFrame = frameIndex;
  const points = state.three.points;
  if (!points || points.length === 0 || frameIndex >= points.length) return;

  const pt = points[frameIndex];
  const marker = state.three.handMarker;
  marker.visible = true;
  marker.position.set(pt.x, pt.y, pt.z);

  // Rotate marker based on wrist roll
  marker.rotation.z = pt.roll * Math.PI * 2;

  // Update slider & text
  document.getElementById("trajectoryScrubber").value = frameIndex;
  document.getElementById("trajectoryFrameDisplay").textContent = `Frame ${frameIndex} / ${
    points.length - 1
  }`;
}

// =========================================================================
// SENSOR DYNAMICS CHART (CHART.JS)
// =========================================================================
function updateSensorChart(channels) {
  const ctx = document.getElementById("sensorDynamicsChart").getContext("2d");
  const frames = channels.frames;

  const allDatasets = [
    { label: "X Position", data: channels.x, borderColor: "#3b82f6", group: "pos" },
    { label: "Y Position", data: channels.y, borderColor: "#06b6d4", group: "pos" },
    { label: "Z Position", data: channels.z, borderColor: "#6366f1", group: "pos" },
    { label: "Wrist Roll", data: channels.roll, borderColor: "#a855f7", group: "rot" },
    { label: "Thumb (F1)", data: channels.thumb, borderColor: "#10b981", group: "fingers" },
    { label: "Forefinger (F2)", data: channels.forefinger, borderColor: "#f59e0b", group: "fingers" },
    { label: "Middle Finger (F3)", data: channels.middle, borderColor: "#ec4899", group: "fingers" },
    { label: "Ring Finger (F4)", data: channels.ring, borderColor: "#14b8a6", group: "fingers" },
  ].map((ds) => ({
    ...ds,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.2,
    fill: false,
    hidden: state.activeChannelGroup !== "all" && ds.group !== state.activeChannelGroup,
  }));

  if (state.sensorChart) {
    state.sensorChart.data.labels = frames;
    state.sensorChart.data.datasets = allDatasets;
    state.sensorChart.update();
  } else {
    state.sensorChart = new Chart(ctx, {
      type: "line",
      data: { labels: frames, datasets: allDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: { color: "#94a3b8", boxWidth: 12, font: { size: 10 } },
          },
          tooltip: {
            backgroundColor: "#0f172a",
            borderColor: "#334155",
            borderWidth: 1,
            titleColor: "#f8fafc",
            bodyColor: "#cbd5e1",
            bodyFont: { size: 11 },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Resampled Time Frame (0 - 56)", color: "#64748b" },
            grid: { color: "rgba(51, 65, 85, 0.3)" },
            ticks: { color: "#94a3b8", font: { size: 10 } },
          },
          y: {
            title: { display: true, text: "Normalized Value [0, 1]", color: "#64748b" },
            grid: { color: "rgba(51, 65, 85, 0.3)" },
            ticks: { color: "#94a3b8", font: { size: 10 } },
            min: 0,
            max: 1,
          },
        },
      },
    });
  }
}

// =========================================================================
// BENCHMARK TABLES & ABLATION CHARTS
// =========================================================================
function renderClassicalTable(models) {
  const tbody = document.getElementById("classicalBenchmarkTable");
  tbody.innerHTML = "";

  models.forEach((m) => {
    const isChamp = m.model.includes("CHAMPION");
    const tr = document.createElement("tr");
    tr.className = isChamp ? "bg-blue-950/30 font-semibold" : "hover:bg-slate-900/40";
    tr.innerHTML = `
      <td class="py-3 px-4 flex items-center space-x-2">
        ${isChamp ? '<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>' : ""}
        <span class="${isChamp ? "text-emerald-300 font-bold" : "text-white"}">${m.model}</span>
      </td>
      <td class="py-3 px-4 font-mono text-xs text-slate-400">${m.scaling}</td>
      <td class="py-3 px-4 font-mono font-bold ${isChamp ? "text-emerald-400 text-base" : "text-slate-200"}">${(m.accuracy * 100).toFixed(1)}%</td>
      <td class="py-3 px-4 font-mono text-slate-300">${m.macro_f1.toFixed(3)}</td>
      <td class="py-3 px-4 font-mono text-slate-400">${m.paper_f1 !== "—" ? (m.paper_f1 * 100).toFixed(1) + "%" : "—"}</td>
      <td class="py-3 px-4 text-xs ${isChamp ? "text-emerald-400 font-bold" : "text-slate-400"}">
        ${isChamp ? "+21.7% over Linear SVM (+5.2% over paper)" : "Baseline"}
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderRecurrentTable(models) {
  const tbody = document.getElementById("recurrentBenchmarkTable");
  tbody.innerHTML = "";

  models.forEach((m) => {
    const isBest = m.name.includes("BEST");
    const isFailure = m.name.includes("Paper Reference");

    const tr = document.createElement("tr");
    tr.className = isBest
      ? "bg-purple-950/30 font-semibold"
      : isFailure
      ? "bg-red-950/20"
      : "hover:bg-slate-900/40";

    tr.innerHTML = `
      <td class="py-3 px-4 flex items-center space-x-2">
        <span class="${isBest ? "text-purple-300 font-bold" : isFailure ? "text-red-400" : "text-white"}">${m.name}</span>
      </td>
      <td class="py-3 px-4 font-mono text-xs text-slate-300">${m.arch}</td>
      <td class="py-3 px-4 font-mono text-xs text-slate-400">${m.units}</td>
      <td class="py-3 px-4 font-mono text-xs text-slate-400">${m.dropout}</td>
      <td class="py-3 px-4 font-mono font-bold ${isBest ? "text-purple-300 text-base" : isFailure ? "text-red-400" : "text-slate-200"}">
        ${(m.accuracy * 100).toFixed(1)}%
      </td>
      <td class="py-3 px-4 font-mono ${isBest ? "text-purple-300 font-bold" : "text-slate-300"}">
        ${m.macro_f1.toFixed(3)}
      </td>
      <td class="py-3 px-4 text-xs ${isBest ? "text-purple-300 font-semibold" : "text-slate-400"}">
        ${m.note}
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAblationSingleTable(items) {
  const tbody = document.getElementById("ablationSingleTable");
  tbody.innerHTML = "";

  items.forEach((item) => {
    const isCritical = item.removed.includes("POS");
    const tr = document.createElement("tr");
    tr.className = isCritical ? "bg-red-950/30 font-semibold" : "hover:bg-slate-900/40";
    tr.innerHTML = `
      <td class="py-2.5 px-3 ${isCritical ? "text-red-400 font-bold" : "text-slate-200"}">${item.removed}</td>
      <td class="py-2.5 px-3 font-mono font-bold ${isCritical ? "text-red-400" : "text-slate-200"}">${(item.test_error * 100).toFixed(1)}%</td>
      <td class="py-2.5 px-3 text-xs ${isCritical ? "text-red-400 font-bold" : "text-slate-400"}">${item.importance}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAblationCumulativeTable(items) {
  const tbody = document.getElementById("ablationCumulativeTable");
  tbody.innerHTML = "";

  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-900/40";
    tr.innerHTML = `
      <td class="py-2.5 px-3 text-slate-200">${item.stage}</td>
      <td class="py-2.5 px-3 font-mono text-xs text-slate-400">${item.features_remaining} / 8</td>
      <td class="py-2.5 px-3 font-mono font-bold text-slate-200">${(item.test_error * 100).toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderConfusedPairsTable(pairs) {
  const tbody = document.getElementById("confusedPairsTable");
  tbody.innerHTML = "";

  pairs.forEach((p) => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-900/40";
    tr.innerHTML = `
      <td class="py-3 px-4 font-semibold text-white">${p.true}</td>
      <td class="py-3 px-4 font-semibold text-amber-400">${p.pred}</td>
      <td class="py-3 px-4 font-mono font-bold text-slate-200">${p.count} mistakes</td>
      <td class="py-3 px-4 text-xs text-slate-400">${p.reason}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAblationChart(singleData) {
  const ctx = document.getElementById("ablationBarChart").getContext("2d");
  const labels = singleData.map((d) => d.removed.replace(/\(.*\)/, ""));
  const errors = singleData.map((d) => (d.test_error * 100).toFixed(1));
  const colors = singleData.map((d) => (d.removed.includes("POS") ? "#ef4444" : "#3b82f6"));

  state.ablationChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Test Error Rate (%)",
          data: errors,
          backgroundColor: colors,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0f172a",
          callbacks: {
            label: (ctx) => ` Test Error: ${ctx.parsed.y}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#94a3b8", font: { size: 10 } },
        },
        y: {
          grid: { color: "rgba(51, 65, 85, 0.3)" },
          ticks: { color: "#94a3b8", font: { size: 10 } },
          min: 0,
          max: 100,
          title: { display: true, text: "Error Rate (%)", color: "#64748b" },
        },
      },
    },
  });
}

// =========================================================================
// EVENT LISTENERS & INTERACTION HANDLERS
// =========================================================================
function setupEventListeners() {
  // Sign Select
  document.getElementById("signSelect").addEventListener("change", (e) => {
    handleSignSelection(e.target.value);
  });

  // Signer Select
  document.getElementById("signerSelect").addEventListener("change", () => {
    const sign = document.getElementById("signSelect").value;
    if (sign) handleSignSelection(sign);
  });

  // Classify Sample Button
  document.getElementById("btnClassifySample").addEventListener("click", () => {
    const filePath = document.getElementById("sampleSelect").value;
    if (filePath) loadSampleFile(filePath);
  });

  // Random Sign Button
  document.getElementById("btnRandomSample").addEventListener("click", loadRandomSample);

  // File Upload
  const fileInput = document.getElementById("fileUploadInput");
  fileInput.addEventListener("change", async (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const formData = new FormData();
      formData.append("file", file);

      try {
        setLoadingState(true);
        const res = await fetch("/api/upload", { method: "POST", body: formData });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        displayPredictionResult(data);
      } catch (err) {
        alert("Upload error: " + err.message);
      } finally {
        setLoadingState(false);
      }
    }
  });

  // Paste Text Modal Triggers
  const pasteModal = document.getElementById("pasteModal");
  document.getElementById("btnPasteTextModal").addEventListener("click", () => {
    pasteModal.classList.remove("hidden");
  });
  document.getElementById("btnCloseModal").addEventListener("click", () => {
    pasteModal.classList.add("hidden");
  });
  document.getElementById("btnCancelModal").addEventListener("click", () => {
    pasteModal.classList.add("hidden");
  });

  document.getElementById("btnSubmitPaste").addEventListener("click", async () => {
    const text = document.getElementById("pasteModalTextarea").value.trim();
    if (!text) {
      alert("Please paste some comma-separated sensor rows.");
      return;
    }

    try {
      setLoadingState(true);
      const res = await fetch("/api/upload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      pasteModal.classList.add("hidden");
      displayPredictionResult(data);
    } catch (err) {
      alert("Error: " + err.message);
    } finally {
      setLoadingState(false);
    }
  });

  // 3D Motion Controls
  const btnPlay = document.getElementById("btnTrajectoryPlay");
  const iconPlayPause = document.getElementById("iconPlayPause");
  const scrubber = document.getElementById("trajectoryScrubber");
  const btnReset = document.getElementById("btnTrajectoryReset");

  btnPlay.addEventListener("click", () => {
    state.three.isPlaying = !state.three.isPlaying;
    if (state.three.isPlaying) {
      btnPlay.classList.replace("bg-blue-600", "bg-amber-600");
      btnPlay.innerHTML = '<i data-lucide="pause" class="w-4 h-4 fill-white"></i>';
    } else {
      btnPlay.classList.replace("bg-amber-600", "bg-blue-600");
      btnPlay.innerHTML = '<i data-lucide="play" class="w-4 h-4 fill-white"></i>';
    }
    if (window.lucide) lucide.createIcons();
  });

  btnReset.addEventListener("click", () => {
    state.three.isPlaying = false;
    btnPlay.classList.replace("bg-amber-600", "bg-blue-600");
    btnPlay.innerHTML = '<i data-lucide="play" class="w-4 h-4 fill-white"></i>';
    if (window.lucide) lucide.createIcons();
    setThreeFrame(0);
  });

  scrubber.addEventListener("input", (e) => {
    state.three.isPlaying = false;
    btnPlay.classList.replace("bg-amber-600", "bg-blue-600");
    btnPlay.innerHTML = '<i data-lucide="play" class="w-4 h-4 fill-white"></i>';
    if (window.lucide) lucide.createIcons();
    setThreeFrame(parseInt(e.target.value, 10));
  });

  // Speed Buttons
  document.querySelectorAll(".speed-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".speed-btn").forEach((b) => {
        b.className =
          "speed-btn px-2 py-1 rounded bg-slate-800 text-slate-300 hover:bg-slate-700 text-[11px]";
      });
      btn.className = "speed-btn px-2 py-1 rounded bg-blue-600 text-white text-[11px] font-semibold";
      state.three.speed = parseFloat(btn.dataset.speed);
    });
  });

  // Sensor Channel Group Toggles
  document.querySelectorAll(".sensor-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".sensor-tab-btn").forEach((b) => {
        b.className =
          "sensor-tab-btn px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 hover:bg-slate-700 text-xs";
      });
      btn.className = "sensor-tab-btn active px-2.5 py-1 rounded-md bg-blue-600 text-white font-medium text-xs";
      state.activeChannelGroup = btn.dataset.channelGroup;

      if (state.sensorChart && state.currentResult && state.currentResult.channels) {
        updateSensorChart(state.currentResult.channels);
      }
    });
  });

  // 95 Signs Search Filter Input
  const filterInput = document.getElementById("signFilterInput");
  filterInput.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase().trim();
    const filtered = state.signsList.filter((s) => s.toLowerCase().includes(q));
    populateSignsDictionary(filtered);
  });
}
