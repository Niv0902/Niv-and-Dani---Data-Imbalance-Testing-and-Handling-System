import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from "recharts";
import StepIndicator from "../components/StepIndicator";
import { getDiagnosis } from "../api/client";
import { useApp } from "../context/AppContext";

const COLORS = ["#2563eb", "#7c3aed", "#db2777", "#059669", "#d97706", "#dc2626", "#0891b2", "#92400e"];

function severityClass(s) {
  if (s === "Low") return "ir-low";
  if (s === "Medium") return "ir-medium";
  if (s === "High") return "ir-high";
  return "ir-extreme";
}

function irExplanation(diag) {
  if (!diag) return "";
  const { ir, severity, majority_class, minority_class, classes } = diag;
  const isBinary = classes.length === 2;
  if (isBinary) {
    return `Your dataset has an Imbalance Ratio of ${ir}. This means the majority class ("${majority_class}") has ${ir}× more samples than the minority class ("${minority_class}"). ${severity === "Low" ? "This is mild imbalance — balancing may not be necessary." : severity === "Medium" ? "This is moderate imbalance that may affect minority-class recall." : "This is severe imbalance. Without balancing, the model will likely ignore the minority class."}`;
  }
  return `Your dataset has ${classes.length} classes with a max pairwise Imbalance Ratio of ${ir} (${severity}). The most common class is "${majority_class}" and the rarest is "${minority_class}".`;
}

export default function DiagnosisPage() {
  const navigate = useNavigate();
  const { datasetId, labelCol, diagnosisResult, setDiagnosisResult } = useApp();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!datasetId || !labelCol) { navigate("/"); return; }
    if (diagnosisResult) return;
    setLoading(true);
    getDiagnosis(datasetId, labelCol)
      .then((r) => setDiagnosisResult(r.data))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [datasetId, labelCol]);

  const diag = diagnosisResult;

  if (loading) return <div className="page-container"><div className="spinner" style={{ margin: "80px auto", display: "block" }} /></div>;

  return (
    <div className="page-container">
      <StepIndicator current={4} />
      <h1 className="page-title">Imbalance diagnosis</h1>
      <p className="page-subtitle">Class distribution and Imbalance Ratio for your dataset.</p>

      {error && <div className="alert alert-error">{error}</div>}

      {diag && (
        <>
          <div className="stat-row">
            <div className="stat-card">
              <div className={`stat-value ${severityClass(diag.severity)}`}>{diag.ir}</div>
              <div className="stat-label">Imbalance Ratio (IR)</div>
            </div>
            <div className="stat-card">
              <div className={`stat-value badge ${statusBadge(diag.severity)}`} style={{ fontSize: 18, padding: "4px 12px", display: "inline-block" }}>
                {diag.severity}
              </div>
              <div className="stat-label" style={{ marginTop: 8 }}>Severity</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{diag.total_samples.toLocaleString()}</div>
              <div className="stat-label">Total samples</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{diag.classes.length}</div>
              <div className="stat-label">Classes</div>
            </div>
          </div>

          {diag.ir === 1.0 && (
            <div className="alert alert-success" style={{ marginBottom: 16 }}>
              ✅ Your dataset is perfectly balanced! Balancing may not be necessary.
            </div>
          )}

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="section-label">Class distribution</div>
            <ResponsiveContainer width="100%" height={Math.min(50 * diag.classes.length + 60, 340)}>
              <BarChart data={diag.classes} layout="vertical" margin={{ left: 100 }}>
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={100} />
                <Tooltip formatter={(v, _, p) => [`${v.toLocaleString()} (${p.payload.pct}%)`, "Samples"]} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {diag.classes.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="section-label">Class breakdown</div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Count</th>
                  <th>Percentage</th>
                  <th>Role</th>
                </tr>
              </thead>
              <tbody>
                {diag.classes.map((cls, i) => (
                  <tr key={cls.name}>
                    <td><strong>{cls.name}</strong></td>
                    <td>{cls.count.toLocaleString()}</td>
                    <td>{cls.pct}%</td>
                    <td>
                      {i === 0
                        ? <span className="tag tag-majority">Majority</span>
                        : i === diag.classes.length - 1
                        ? <span className="tag tag-minority">Minority</span>
                        : <span className="tag" style={{ background: "var(--gray-100)", color: "var(--gray-600)" }}>Middle</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <div className="section-label">What this means</div>
            <p style={{ fontSize: 14, color: "var(--gray-600)", lineHeight: 1.6 }}>
              {irExplanation(diag)}
            </p>
            <div style={{ marginTop: 12, fontSize: 13, color: "var(--gray-400)" }}>
              IR thresholds: <strong style={{ color: "var(--green)" }}>Low</strong> (&lt;3) &nbsp;|&nbsp;
              <strong style={{ color: "var(--yellow)" }}>Medium</strong> (3–10) &nbsp;|&nbsp;
              <strong style={{ color: "var(--orange)" }}>High</strong> (10–50) &nbsp;|&nbsp;
              <strong style={{ color: "var(--red)" }}>Extreme</strong> (&gt;50)
            </div>
          </div>
        </>
      )}

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={() => navigate("/validate")}>← Back</button>
        <div className="nav-row-right">
          <button className="btn btn-secondary" onClick={() => navigate("/results-only")}>
            Skip balancing
          </button>
          <button className="btn btn-primary" onClick={() => navigate("/configure")}>
            Continue to balancing →
          </button>
        </div>
      </div>
    </div>
  );
}

function statusBadge(s) {
  if (s === "Low") return "badge-pass";
  if (s === "Medium") return "badge-warning";
  return "badge-error";
}
