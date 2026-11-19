// PathSnap evaluator UI (Laptop 2). Talks only to the API; never sees reference routes.
const $ = (id) => document.getElementById(id);
const DEFAULT_API = location.protocol.startsWith("http") && location.port === "8000"
  ? location.origin : "http://localhost:8000";

let API = DEFAULT_API;
try { API = localStorage.getItem("pathsnap_api") || DEFAULT_API; } catch (e) { /* storage blocked */ }
$("api-url").value = API;

const map = L.map("map", { preferCanvas: true }).setView([55.75, 37.62], 11);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
// drawing order = z-order: route first, classifier arcs above it, GPS dots on top
const layers = { hmm: L.layerGroup().addTo(map), clf: L.layerGroup().addTo(map), gps: L.layerGroup().addTo(map) };
L.control.layers(null, {
  "Raw GPS": layers.gps, "Classifier-only arcs": layers.clf, "Predicted route (HMM)": layers.hmm,
}, { collapsed: false }).addTo(map);

function log(msg, isErr = false) {
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
  if (isErr) li.className = "err";
  $("log").prepend(li);
}
const pct = (x) => (x == null ? "–" : (100 * x).toFixed(2) + "%");
const num = (x, d = 3) => (x == null ? "–" : Number(x).toFixed(d));

async function connect(ev) {
  ev?.preventDefault();
  API = $("api-url").value.trim().replace(/\/$/, "");
  try { localStorage.setItem("pathsnap_api", API); } catch (e) { /* ignore */ }
  $("api-status").textContent = "connecting…"; $("api-status").className = "status";
  try {
    const h = await (await fetch(`${API}/health`)).json();
    const r = await fetch(`${API}/trips`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const { trips } = await r.json();
    $("api-status").textContent = `connected · model ${h.model} · ${trips.length} trips`;
    $("api-status").className = "status ok";
    log(`GET /trips -> ${trips.length} trips`);
    renderTrips(trips);
  } catch (e) {
    $("api-status").textContent = "cannot reach API"; $("api-status").className = "status err";
    log(`connect failed: ${e.message}`, true);
  }
}

function renderTrips(trips) {
  const box = $("trips");
  box.innerHTML = "";
  if (!trips.length) { box.innerHTML = '<p class="hint">The API has no evaluation trips.</p>'; return; }
  trips.forEach((t) => {
    const b = document.createElement("button");
    b.className = "trip"; b.type = "button";
    b.innerHTML = `<b>${t.trip_id}</b><span>${t.n_gps.toLocaleString()} GPS points</span>`;
    b.onclick = () => runTrip(t.trip_id, b);
    box.appendChild(b);
  });
}

async function runTrip(tripId, btn) {
  document.querySelectorAll(".trip").forEach((x) => { x.classList.remove("active"); x.disabled = true; });
  btn.classList.add("active");
  $("busy").hidden = false;
  log(`POST /predict {trip_id: ${tripId}}`);
  try {
    const r = await fetch(`${API}/predict`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: tripId }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const res = await r.json();
    draw(res); showMetrics(res);
    log(`trip ${tripId}: accuracy ${pct(res.metrics.hmm.pointwise_accuracy)} in ${res.timing_s.request_total}s`);
  } catch (e) {
    log(`predict failed: ${e.message}`, true);
  } finally {
    $("busy").hidden = true;
    document.querySelectorAll(".trip").forEach((x) => { x.disabled = false; });
  }
}

function draw(res) {
  Object.values(layers).forEach((l) => l.clearLayers());
  const css = getComputedStyle(document.documentElement);
  const gpsColor = css.getPropertyValue("--gps").trim();
  // lines first, GPS dots last so they are drawn on top
  res.predicted_route.forEach((seg) =>
    L.polyline(seg, { color: css.getPropertyValue("--accent").trim(), weight: 5, opacity: 0.85 }).addTo(layers.hmm));
  res.classifier_only_arcs.forEach((seg) =>
    L.polyline(seg, { color: css.getPropertyValue("--clf").trim(), weight: 3, dashArray: "5 5", opacity: 0.9 }).addTo(layers.clf));
  const step = Math.max(1, Math.floor(res.gps.length / 1500));
  res.gps.forEach((p, i) => {
    if (i % step === 0) L.circleMarker(p, { radius: 2, stroke: false, fillColor: gpsColor, fillOpacity: 0.55 }).addTo(layers.gps);
  });
  map.fitBounds(L.latLngBounds(res.gps), { padding: [30, 30] });
}

function showMetrics(res) {
  const h = res.metrics.hmm, c = res.metrics.classifier_only;
  $("results").hidden = false;
  $("r-trip").textContent = res.trip_id;
  $("k-acc").textContent = pct(h.pointwise_accuracy);
  $("k-acc-clf").textContent = pct(c.pointwise_accuracy);
  $("k-cont").textContent = pct(h.route_continuity);
  $("k-f1").textContent = num(h.route_f1);
  const rows = [
    ["Pointwise accuracy", pct(c.pointwise_accuracy), pct(h.pointwise_accuracy)],
    ["Route continuity", pct(c.route_continuity), pct(h.route_continuity)],
    ["Breaks (disconnected steps)", c.n_breaks, h.n_breaks],
    ["Route precision", num(c.route_precision), num(h.route_precision)],
    ["Route recall", num(c.route_recall), num(h.route_recall)],
    ["Route F1", num(c.route_f1), num(h.route_f1)],
  ];
  $("metrics-body").innerHTML = rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("");
  const t = res.timing_s;
  $("r-meta").textContent = `${res.n_gps.toLocaleString()} GPS points · model ${res.model} · ` +
    `features ${t.candidates_features}s, classifier ${t.classifier}s, HMM ${t.hmm_viterbi}s, total ${t.request_total}s`;
}

$("api-form").addEventListener("submit", connect);
connect();
