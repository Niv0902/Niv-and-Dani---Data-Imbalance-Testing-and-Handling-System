import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import StepIndicator from "../components/StepIndicator";
import { getColumns, getPreview, getColumnSummary } from "../api/client";
import { useApp } from "../context/AppContext";

const COLORS = ["#2563eb", "#7c3aed", "#db2777", "#059669", "#d97706", "#dc2626", "#0891b2"];

export default function ColumnSelectionPage() {
  const navigate = useNavigate();
  const { datasetId, datasetMeta, labelCol, setLabelCol } = useApp();
  const [columns, setColumns] = useState([]);
  const [preview, setPreview] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!datasetId) { navigate("/"); return; }
    Promise.all([getColumns(datasetId), getPreview(datasetId)])
      .then(([colRes, previewRes]) => {
        setColumns(colRes.data);
        setPreview(previewRes.data);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [datasetId]);

  function handleColChange(col) {
    setLabelCol(col);
    setSummary(null);
    getColumnSummary(datasetId, col)
      .then((r) => setSummary(r.data))
      .catch(() => {});
  }

  const selectedColInfo = columns.find((c) => c.name === labelCol);
  const manyUnique = selectedColInfo && selectedColInfo.unique_count > 50;
  const isNumeric = selectedColInfo && selectedColInfo.dtype === "numeric";

  if (loading) return <div className="page-container"><div className="spinner" style={{ margin: "60px auto", display: "block" }} /></div>;
  if (error) return <div className="page-container"><div className="alert alert-error">{error}</div></div>;

  return (
    <div className="page-container">
      <StepIndicator current={2} />
      <h1 className="page-title">Select label column</h1>
      <p className="page-subtitle">
        Choose the column that contains the class labels for your classification task.
      </p>

      {preview && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="section-label">Dataset preview (first 5 rows)</div>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>{preview.columns.map((c) => <th key={c}>{c}</th>)}</tr>
              </thead>
              <tbody>
                {preview.rows.map((row, i) => (
                  <tr key={i}>{preview.columns.map((c) => <td key={c}>{row[c]}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="card">
        <div className="param-row">
          <label className="param-label" htmlFor="label-select">Label column</label>
          <select
            id="label-select"
            className="param-input"
            value={labelCol || ""}
            onChange={(e) => handleColChange(e.target.value)}
          >
            <option value="">— select a column —</option>
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.dtype}, {c.unique_count} unique)
              </option>
            ))}
          </select>
        </div>

        {manyUnique && (
          <div className="alert alert-warning" style={{ marginTop: 8 }}>
            ⚠ This column has many unique values ({selectedColInfo.unique_count}).
            Are you sure it contains class labels?
          </div>
        )}
        {isNumeric && (
          <div className="alert alert-info" style={{ marginTop: 8 }}>
            ℹ This column appears numeric. Label columns are typically categorical.
          </div>
        )}
      </div>

      {summary && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="section-label">Class distribution preview</div>
          <div style={{ display: "flex", gap: 24, marginBottom: 16 }}>
            <div><strong>{summary.unique_count}</strong> <span style={{ color: "var(--gray-500)", fontSize: 13 }}>unique classes</span></div>
            <div><strong>{summary.total.toLocaleString()}</strong> <span style={{ color: "var(--gray-500)", fontSize: 13 }}>samples</span></div>
          </div>
          <ResponsiveContainer width="100%" height={Math.min(40 * summary.classes.length + 40, 280)}>
            <BarChart data={summary.classes} layout="vertical" margin={{ left: 80 }}>
              <XAxis type="number" tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={80} />
              <Tooltip formatter={(v, _, p) => [`${v} (${p.payload.pct}%)`, "Count"]} />
              <Bar dataKey="count">
                {summary.classes.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={() => navigate("/")}>← Back</button>
        <button className="btn btn-primary" disabled={!labelCol} onClick={() => navigate("/validate")}>
          Continue →
        </button>
      </div>
    </div>
  );
}
