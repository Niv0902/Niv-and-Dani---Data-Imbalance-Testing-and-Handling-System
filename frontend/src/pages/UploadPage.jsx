import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import StepIndicator from "../components/StepIndicator";
import { uploadDataset } from "../api/client";
import { useApp } from "../context/AppContext";

export default function UploadPage() {
  const navigate = useNavigate();
  const { setDatasetId, setDatasetMeta, reset } = useApp();
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(null); // {filename, rows, columns, dataset_id}
  const [error, setError] = useState(null);
  const fileRef = useRef();

  function handleFile(file) {
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["csv", "xlsx", "xls"].includes(ext)) {
      setError("Unsupported file format. Please upload a CSV or XLSX file.");
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setError("File exceeds 50 MB. Please reduce the dataset size.");
      return;
    }
    setError(null);
    setUploading(true);
    setUploaded(null);
    uploadDataset(file)
      .then((res) => {
        setUploaded(res.data);
        setDatasetId(res.data.dataset_id);
        setDatasetMeta({ filename: res.data.filename, rows: res.data.rows, columns: res.data.columns });
      })
      .catch((e) => setError(e.message))
      .finally(() => setUploading(false));
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }

  function onContinue() {
    navigate("/columns");
  }

  function startOver() {
    reset();
    setUploaded(null);
    setError(null);
  }

  return (
    <div className="page-container">
      <StepIndicator current={1} />
      <h1 className="page-title">Upload your dataset</h1>
      <p className="page-subtitle">
        Upload a labeled classification dataset in CSV or XLSX format (max 50 MB).
      </p>

      <div
        className={`upload-zone ${dragOver ? "drag-over" : ""} ${uploading ? "uploading" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !uploading && fileRef.current.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <div className="upload-icon">
          {uploading ? <span className="spinner" style={{ width: 48, height: 48, borderWidth: 4 }} /> : "📂"}
        </div>
        <h3>{uploading ? "Uploading…" : "Drag & drop your file here"}</h3>
        <p>or click to browse &nbsp;·&nbsp; Supported: CSV, XLSX &nbsp;·&nbsp; Max 50 MB</p>
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginTop: 16 }}>
          <span>⚠</span> {error}
        </div>
      )}

      {uploaded && (
        <div className="card" style={{ marginTop: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
            <div>
              <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 6 }}>✅ {uploaded.filename}</div>
              <div style={{ display: "flex", gap: 24, color: "var(--gray-500)", fontSize: 14 }}>
                <span><strong style={{ color: "var(--gray-800)" }}>{uploaded.rows.toLocaleString()}</strong> rows</span>
                <span><strong style={{ color: "var(--gray-800)" }}>{uploaded.columns}</strong> columns</span>
              </div>
            </div>
            <button className="btn btn-link" onClick={startOver}>Change file</button>
          </div>
        </div>
      )}

      <div className="nav-row">
        <span />
        <button className="btn btn-primary" disabled={!uploaded || uploading} onClick={onContinue}>
          Continue →
        </button>
      </div>
    </div>
  );
}
