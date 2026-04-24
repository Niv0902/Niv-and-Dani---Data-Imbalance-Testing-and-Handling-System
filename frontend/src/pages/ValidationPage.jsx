import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import StepIndicator from "../components/StepIndicator";
import { validateDataset } from "../api/client";
import { useApp } from "../context/AppContext";

function statusClass(s) {
  if (s === "pass") return "badge-pass";
  if (s === "warning") return "badge-warning";
  return "badge-error";
}

function CheckRow({ check }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderBottom: "1px solid var(--gray-100)", padding: "12px 0" }}>
      <div
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center", cursor: check.details ? "pointer" : "default" }}
        onClick={() => check.details && setOpen((o) => !o)}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className={`badge ${statusClass(check.status)}`}>
            {check.status === "pass" ? "✓ Pass" : check.status === "warning" ? "⚠ Warning" : "✕ Error"}
          </span>
          <span style={{ fontWeight: 500 }}>{check.name}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 13, color: "var(--gray-500)" }}>{check.message}</span>
          {check.details && <span style={{ color: "var(--gray-400)" }}>{open ? "▲" : "▼"}</span>}
        </div>
      </div>
      {open && check.details && (
        <div style={{ marginTop: 8, padding: "8px 12px", background: "var(--gray-50)", borderRadius: 6, fontSize: 13, color: "var(--gray-600)", whiteSpace: "pre-wrap" }}>
          {check.details}
        </div>
      )}
    </div>
  );
}

export default function ValidationPage() {
  const navigate = useNavigate();
  const { datasetId, labelCol, setValidationResult, validationResult } = useApp();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!datasetId || !labelCol) { navigate("/"); return; }
    if (validationResult) return;
    setLoading(true);
    validateDataset(datasetId, labelCol)
      .then((r) => setValidationResult(r.data))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [datasetId, labelCol]);

  const result = validationResult;
  const allGood = result && !result.has_errors && result.warnings === 0;

  if (loading) return <div className="page-container"><div className="spinner" style={{ margin: "80px auto", display: "block" }} /></div>;

  return (
    <div className="page-container">
      <StepIndicator current={3} />
      <h1 className="page-title">Data validation</h1>
      <p className="page-subtitle">
        Automatic quality checks on your dataset. Fix any errors before proceeding.
      </p>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <>
          <div className="stat-row">
            <div className="stat-card">
              <div className="stat-value" style={{ color: "var(--green)" }}>{result.passed}</div>
              <div className="stat-label">Passed</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: result.warnings > 0 ? "var(--yellow)" : "var(--gray-400)" }}>{result.warnings}</div>
              <div className="stat-label">Warnings</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: result.errors > 0 ? "var(--red)" : "var(--gray-400)" }}>{result.errors}</div>
              <div className="stat-label">Errors</div>
            </div>
          </div>

          {allGood && (
            <div className="alert alert-success" style={{ marginBottom: 16 }}>
              🎉 All checks passed! Your dataset looks good. Click Continue to proceed.
            </div>
          )}

          {result.has_errors && (
            <div className="alert alert-error" style={{ marginBottom: 16 }}>
              ✕ Blocking errors found. Please go back and select a valid label column or fix your dataset.
            </div>
          )}

          <div className="card">
            {result.checks.map((c, i) => <CheckRow key={i} check={c} />)}
          </div>

          {result.warnings > 0 && !result.has_errors && (
            <div className="alert alert-warning" style={{ marginTop: 16 }}>
              ⚠ Warnings found. You can proceed, but results may be affected by missing or inconsistent data.
            </div>
          )}
        </>
      )}

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={() => { setValidationResult(null); navigate("/columns"); }}>← Back</button>
        <button
          className="btn btn-primary"
          disabled={!result || result.has_errors}
          onClick={() => navigate("/diagnosis")}
        >
          Continue →
        </button>
      </div>
    </div>
  );
}
