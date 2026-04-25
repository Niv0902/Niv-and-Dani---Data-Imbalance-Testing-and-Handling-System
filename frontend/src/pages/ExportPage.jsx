import { useState } from "react";
import { useNavigate } from "react-router-dom";
import StepIndicator from "../components/StepIndicator";
import { exportDatasetUrl, exportSummaryUrl, exportAllUrl } from "../api/client";
import { useApp } from "../context/AppContext";

function ExportCard({ icon, title, desc, url, filename }) {
  const [clicked, setClicked] = useState(false);
  function handleDownload() {
    setClicked(true);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => setClicked(false), 3000);
  }
  return (
    <div className="card export-card">
      <div style={{ fontSize: 32, marginBottom: 8 }}>{icon}</div>
      <div className="export-card-title">{title}</div>
      <div className="export-card-desc">{desc}</div>
      <button className={`btn ${clicked ? "btn-secondary" : "btn-primary"}`} style={{ marginTop: 12, alignSelf: "flex-start" }} onClick={handleDownload}>
        {clicked ? "✓ Downloaded" : "Download"}
      </button>
    </div>
  );
}

export default function ExportPage() {
  const navigate = useNavigate();
  const { currentRunId, runs, reset } = useApp();
  const run = runs.find((r) => r.run_id === currentRunId) || runs[runs.length - 1];

  if (!run) {
    return (
      <div className="page-container">
        <div className="alert alert-warning">No completed run found. Please run the pipeline first.</div>
        <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => navigate("/configure")}>Go to configuration</button>
      </div>
    );
  }

  const runId = run.run_id;

  return (
    <div className="page-container">
      <StepIndicator current={8} />
      <h1 className="page-title">Export results</h1>
      <p className="page-subtitle">Download the balanced training dataset and run summary.</p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <ExportCard
          icon="📊"
          title="Balanced Dataset"
          desc="The resampled training set as a CSV file. Contains only the training portion — the test set is kept untouched."
          url={exportDatasetUrl(runId)}
          filename={`balanced_dataset_${runId.slice(0, 8)}.csv`}
        />
        <ExportCard
          icon="📄"
          title="Run Summary"
          desc="A PDF report with project info, method, parameters, IR before/after, class distributions with counts, and timestamp."
          url={exportSummaryUrl(runId)}
          filename={`run_summary_${runId.slice(0, 8)}.pdf`}
        />
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontWeight: 700 }}>📦 Download all</div>
            <div style={{ fontSize: 13, color: "var(--gray-500)" }}>Both files packaged as a single ZIP archive.</div>
          </div>
          <a
            className="btn btn-secondary"
            href={exportAllUrl(runId)}
            download={`imbalancekit_export_${runId.slice(0, 8)}.zip`}
          >
            Download ZIP
          </a>
        </div>
      </div>

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={() => navigate("/results")}>← Back to results</button>
        <div className="nav-row-right">
          <button className="btn btn-secondary" onClick={() => navigate("/history")}>View run history</button>
          <button
            className="btn btn-primary"
            onClick={() => { reset(); navigate("/"); }}
          >
            Start new run
          </button>
        </div>
      </div>
    </div>
  );
}
