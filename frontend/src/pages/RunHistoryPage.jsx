import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
} from "recharts";
import StepIndicator from "../components/StepIndicator";
import { useApp } from "../context/AppContext";

const COLORS = ["#2563eb", "#7c3aed", "#db2777", "#059669"];

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
      const va = a[sortKey] ?? 0;
      const vb = b[sortKey] ?? 0;
      return sortAsc ? va - vb : vb - va;
    });
  }

  const selectedRuns = runs.filter((r) => selected.includes(r.run_id));

  // Grouped bar chart: IR Before + IR After per selected run
  const chartData = selectedRuns.map((r, i) => ({
    name: `Run ${runs.indexOf(r) + 1}: ${r.method.toUpperCase()}`,
    "IR Before": r.ir_before,
    "IR After":  r.ir_after,
  }));

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
        {runs.length} run(s) this session.{" "}
        {runs.length < 2 ? "Run another method to enable comparison." : "Select 2–4 runs to compare them."}
      </p>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                {runs.length > 1 && <th>Select</th>}
                <th>#</th>
                <th>Method</th>
                <th
                  style={{ cursor: "pointer", userSelect: "none" }}
                  onClick={() => handleSort("ir_before")}
                >
                  IR Before {sortKey === "ir_before" ? (sortAsc ? "▲" : "▼") : ""}
                </th>
                <th
                  style={{ cursor: "pointer", userSelect: "none" }}
                  onClick={() => handleSort("ir_after")}
                >
                  IR After {sortKey === "ir_after" ? (sortAsc ? "▲" : "▼") : ""}
                </th>
                <th>Reduction</th>
                <th>Elapsed</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {displayRuns.map((run, i) => {
                const reduction = run.ir_before > 0
                  ? `↓ ${((1 - run.ir_after / run.ir_before) * 100).toFixed(1)}%`
                  : "—";
                return (
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
                    <td>{runs.indexOf(run) + 1}</td>
                    <td><strong>{run.method.toUpperCase()}</strong></td>
                    <td>{run.ir_before}</td>
                    <td style={{ color: run.ir_after < run.ir_before ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                      {run.ir_after}
                    </td>
                    <td className={run.ir_after < run.ir_before ? "positive" : "negative"}>{reduction}</td>
                    <td>{run.elapsed_seconds}s</td>
                    <td>
                      <button
                        className="btn btn-link btn-sm"
                        onClick={() => { setCurrentRunId(run.run_id); navigate("/results"); }}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {selected.length >= 2 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="section-label">IR Comparison — {selected.length} selected runs</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} margin={{ left: 8, right: 16 }}>
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="IR Before" fill="#9ca3af" />
              <Bar dataKey="IR After"  fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>

          <div className="table-scroll" style={{ marginTop: 16 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Method</th>
                  <th>IR Before</th>
                  <th>IR After</th>
                  <th>Reduction</th>
                </tr>
              </thead>
              <tbody>
                {selectedRuns.map((r) => {
                  const reduction = r.ir_before > 0
                    ? `↓ ${((1 - r.ir_after / r.ir_before) * 100).toFixed(1)}%`
                    : "—";
                  return (
                    <tr key={r.run_id}>
                      <td>Run {runs.indexOf(r) + 1}</td>
                      <td>{r.method.toUpperCase()}</td>
                      <td>{r.ir_before}</td>
                      <td style={{ color: r.ir_after < r.ir_before ? "var(--green)" : "var(--red)", fontWeight: 600 }}>
                        {r.ir_after}
                      </td>
                      <td className={r.ir_after < r.ir_before ? "positive" : "negative"}>{reduction}</td>
                    </tr>
                  );
                })}
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
