import { useState } from "react";
import { useNavigate } from "react-router-dom";
import StepIndicator from "../components/StepIndicator";
import { startBalancing } from "../api/client";
import { useApp } from "../context/AppContext";

const METHODS = [
  {
    id: "smote",
    icon: "⬆",
    title: "SMOTE",
    subtitle: "Oversampling",
    desc: "Generates synthetic minority-class samples by interpolating between existing ones.",
    pros: "Increases minority-class representation without losing majority data.",
    cons: "Can introduce noise; sensitive to k_neighbors parameter.",
    when: "Use when you have a small minority class and enough samples to interpolate.",
  },
  {
    id: "nearmiss",
    icon: "⬇",
    title: "NearMiss",
    subtitle: "Undersampling",
    desc: "Removes majority-class samples that are nearest to the minority class.",
    pros: "Reduces dataset size; fast and interpretable.",
    cons: "Discards majority-class data; may lose useful information.",
    when: "Use when the majority class is very large and you can afford to remove samples.",
  },
  {
    id: "combined",
    icon: "↕",
    title: "Combined",
    subtitle: "SMOTE + NearMiss",
    desc: "Applies SMOTE to oversample the minority class, then NearMiss to clean the majority.",
    pros: "Balances both sides; often more robust than either method alone.",
    cons: "More parameters to tune; slower than individual methods.",
    when: "Use when both classes need adjustment for a balanced dataset.",
  },
];

export default function BalancingConfigPage() {
  const navigate = useNavigate();
  const { datasetId, labelCol, setCurrentRunId, runs } = useApp();
  const [method, setMethod] = useState("smote");
  const [params, setParams] = useState({ k_neighbors: 5, version: 1, n_neighbors: 3, nearmiss_version: 1 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  function setParam(key, val) {
    setParams((p) => ({ ...p, [key]: val }));
  }

  function handleRun() {
    setError(null);
    setLoading(true);
    startBalancing(datasetId, labelCol, method, params)
      .then((r) => {
        setCurrentRunId(r.data.run_id);
        navigate("/processing");
      })
      .catch((e) => { setError(e.message); setLoading(false); });
  }

  const hasPriorRun = runs.length > 0;
  const selectedMethod = METHODS.find((m) => m.id === method);

  return (
    <div className="page-container">
      <StepIndicator current={5} />
      <h1 className="page-title">Balancing configuration</h1>
      <p className="page-subtitle">Choose a balancing method and configure its parameters.</p>

      {hasPriorRun && (
        <div className="alert alert-info" style={{ marginBottom: 20 }}>
          ℹ Previous run data will be preserved for comparison in Run History.
        </div>
      )}

      <div className="method-cards">
        {METHODS.map((m) => (
          <div
            key={m.id}
            className={`method-card ${method === m.id ? "selected" : ""}`}
            onClick={() => setMethod(m.id)}
          >
            <div className="method-card-icon">{m.icon}</div>
            <div className="method-card-title">{m.title}</div>
            <div style={{ fontSize: 11, color: "var(--blue)", fontWeight: 600, marginBottom: 4 }}>{m.subtitle}</div>
            <div className="method-card-desc">{m.desc}</div>
          </div>
        ))}
      </div>

      {/* Parameter panel */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 700, marginBottom: 16 }}>Parameters for {selectedMethod.title}</div>

        {(method === "smote" || method === "combined") && (
          <div className="param-row">
            <label className="param-label">
              k_neighbors <span className="badge badge-info" style={{ fontSize: 11 }}>default: 5</span>
            </label>
            <input
              type="number"
              className="param-input"
              min={1} max={15}
              value={params.k_neighbors}
              onChange={(e) => setParam("k_neighbors", parseInt(e.target.value) || 5)}
              style={{ maxWidth: 120 }}
            />
            <span className="param-hint">Number of nearest neighbours for SMOTE interpolation (1–15).</span>
          </div>
        )}

        {(method === "nearmiss" || method === "combined") && (
          <>
            <div className="param-row">
              <label className="param-label">
                NearMiss version <span className="badge badge-info" style={{ fontSize: 11 }}>default: 1</span>
              </label>
              <select
                className="param-input"
                style={{ maxWidth: 120 }}
                value={method === "combined" ? params.nearmiss_version : params.version}
                onChange={(e) => {
                  const v = parseInt(e.target.value);
                  method === "combined" ? setParam("nearmiss_version", v) : setParam("version", v);
                }}
              >
                <option value={1}>Version 1</option>
                <option value={2}>Version 2</option>
                <option value={3}>Version 3</option>
              </select>
              <span className="param-hint">
                V1: select nearest majority to minority. V2: farthest minority. V3: 2-step selection.
              </span>
            </div>
            <div className="param-row">
              <label className="param-label">
                n_neighbors <span className="badge badge-info" style={{ fontSize: 11 }}>default: 3</span>
              </label>
              <input
                type="number"
                className="param-input"
                min={1} max={10}
                value={params.n_neighbors}
                onChange={(e) => setParam("n_neighbors", parseInt(e.target.value) || 3)}
                style={{ maxWidth: 120 }}
              />
              <span className="param-hint">Number of nearest neighbours for NearMiss selection.</span>
            </div>
          </>
        )}

        <button className="btn-link" style={{ fontSize: 13, marginTop: 4 }}
          onClick={() => setParams({ k_neighbors: 5, version: 1, n_neighbors: 3, nearmiss_version: 1 })}>
          Reset to safe defaults
        </button>
      </div>

      {/* Method info */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="collapsible-header" onClick={() => setExpanded((o) => !o)}>
          <span>About {selectedMethod.title}</span>
          <span>{expanded ? "▲" : "▼"}</span>
        </div>
        {expanded && (
          <div className="collapsible-content" style={{ fontSize: 14, color: "var(--gray-600)", lineHeight: 1.7 }}>
            <p><strong>How it works:</strong> {selectedMethod.desc}</p>
            <p style={{ marginTop: 8 }}><strong>Pros:</strong> {selectedMethod.pros}</p>
            <p style={{ marginTop: 8 }}><strong>Cons:</strong> {selectedMethod.cons}</p>
            <p style={{ marginTop: 8 }}><strong>When to use:</strong> {selectedMethod.when}</p>
          </div>
        )}
      </div>

      {error && <div className="alert alert-error" style={{ marginBottom: 16 }}>⚠ {error}</div>}

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={() => navigate("/diagnosis")}>← Back</button>
        <button className="btn btn-primary" disabled={loading} onClick={handleRun}>
          {loading ? <><span className="spinner" />Running…</> : "Run balancing →"}
        </button>
      </div>
    </div>
  );
}
