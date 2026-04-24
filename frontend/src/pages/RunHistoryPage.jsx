import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Legend, Tooltip,
} from "recharts";
import StepIndicator from "../components/StepIndicator";
import { useApp } from "../context/AppContext";

const METRIC_KEYS = [
  { key: "macro_f1", label: "Macro F1" },
  { key: "minority_recall", label: "Min. Recall" },
  { key: "minority_f1", label: "Min. F1" },
  { key: "accuracy", label: "Accuracy" },
];

const COLORS = ["#2563eb", "#7c3aed", "#db2777", "#059669"];

function fmt(v) { return v !== undefined ? (v * 100).toFixed(1) + "%" : "—"; }

export default function RunHistoryPage() {
  const navigate = useNavigate();
  const { runs, currentRunId, setCurrentRunId } = useApp();
  const [selected, setSelected] = useState([]);
  const [sortKey, setSortKey] = useState(null);
  const [sortAsc, setSortAsc] = useState(false);

  function toggleSelect(runId) {
    setSelected((prev) =>
      prev.includes(runId)
        ? prev.filter((id) => id !== runId)
        : prev.length < 4 ? [...prev, runId] : prev
    );
  }

  function handleSort(key) {
    if (sortKey === key) setSortAsc((a) => !a);
    else { setSortKey(key); setSortAsc(false); }
  }

  let displayRuns = [...runs];
  if (sortKey) {
    displayRuns.sort((a, b) => {
      const va = a.metrics_after?.[sortKey] ?? 0;
      const vb = b.metrics_after?.[sortKey] ?? 0;
      return sortAsc ? va - vb : vb - va;
    });
  }

  const selectedRuns = runs.filter((r) => selected.includes(r.run_id));

  // Radar data for comparison
  const radarData = METRIC_KEYS.map(({ key, label }) => {
    const entry = { metric: label };
    selectedRuns.forEach((r, i) => {
      entry[`Run ${i + 1}: ${r.method}`] = parseFloat(((r.metrics_after?.[key] ?? 0) * 100).toFixed(1));
    });
    return entry;
  });

  if (runs.length === 0) {
    return (
      <div className="page-container">
        <StepIndicator current={9} />
        <h1 className="page-title">Run history</h1>
        <div className="alert alert-info" style={{ marginTop: 20 }}>
          ℹ No runs yet. Complete a balancing pipeline to see results here.
        </div>
        <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => navigate("/configure")}>
          Start first run
        </button>
      </div>
    );
  }

  return (
    <div className="page-container">
      <StepIndicator current={9} />
      <h1 className="page-title">Run history</h1>
      <p className="page-subtitle">
        {runs.length} run(s) completed. {runs.length < 2 ? "Run another method to enable comparison." : "Select 2–4 runs to compare them side by side."}
      </p>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="table-scroll">
          <table className="data-table run-history-table">
            <thead>
              <tr>
                {runs.length > 1 && <th>Select</th>}
                <th>#</th>
                <th>Method</th>
                <th>IR Before</th>
                <th>IR After</th>
                {METRIC_KEYS.map(({ key, label }) => (
                  <th
                    key={key}
                    style={{ cursor: "pointer", userSelect: "none" }}
                    onClick={() => handleSort(key)}
                  >
                    {label} After {sortKey === key ? (sortAsc ? "▲" : "▼") : ""}
                  </th>
                ))}
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {displayRuns.map((run, i) => (
                <tr key={run.run_id} style={run.run_id === currentRunId ? { background: "var(--blue-light)" } : {}}>
                  {runs.length > 1 && (
                    <td>
                      <input
                        type="checkbox"
                        checked={selected.includes(run.run_id)}
                        onChange={() => toggleSelect(run.run_id)}
                        disabled={!selected.includes(run.run_id) && selected.length >= 4}
                      />
                    </td>
                  )}
                  <td>{i + 1}</td>
                  <td><strong>{run.method.toUpperCase()}</strong></td>
                  <td>{run.ir_before}</td>
                  <td>{run.ir_after}</td>
                  {METRIC_KEYS.map(({ key }) => (
                    <td key={key}>{fmt(run.metrics_after?.[key])}</td>
                  ))}
                  <td>
                    <button
                      className="btn btn-link btn-sm"
                      onClick={() => { setCurrentRunId(run.run_id); navigate("/results"); }}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selected.length >= 2 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="section-label">Comparison — {selected.length} selected runs (metrics after balancing)</div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" tick={{ fontSize: 13 }} />
              <Tooltip formatter={(v) => `${v}%`} />
              <Legend />
              {selectedRuns.map((r, i) => (
                <Radar
                  key={r.run_id}
                  name={`Run ${runs.indexOf(r) + 1}: ${r.method.toUpperCase()}`}
                  dataKey={`Run ${i + 1}: ${r.method}`}
                  stroke={COLORS[i]}
                  fill={COLORS[i]}
                  fillOpacity={0.15}
                />
              ))}
            </RadarChart>
          </ResponsiveContainer>

          <div className="table-scroll" style={{ marginTop: 16 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  {selectedRuns.map((r, i) => (
                    <th key={r.run_id} style={{ color: COLORS[i] }}>
                      Run {runs.indexOf(r) + 1}: {r.method.toUpperCase()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {METRIC_KEYS.map(({ key, label }) => (
                  <tr key={key}>
                    <td>{label}</td>
                    {selectedRuns.map((r) => (
                      <td key={r.run_id}>{fmt(r.metrics_after?.[key])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={() => navigate("/results")}>← Back to latest results</button>
        <button className="btn btn-primary" onClick={() => navigate("/configure")}>Try another method</button>
      </div>
    </div>
  );
}
