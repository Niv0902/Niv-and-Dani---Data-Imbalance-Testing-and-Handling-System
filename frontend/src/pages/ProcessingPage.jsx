import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getStatus, cancelRun, getResults } from "../api/client";
import { useApp } from "../context/AppContext";
import StepIndicator from "../components/StepIndicator";

const STAGES = [
  "Splitting data",
  "Applying resampling",
  "Computing statistics",
  "Finishing up...",
  "Done",
];

function stageState(currentStage, label) {
  const ci = STAGES.indexOf(currentStage);
  const li = STAGES.indexOf(label);
  if (li < ci) return "done";
  if (li === ci) return "active";
  return "pending";
}

function stageIcon(s) {
  if (s === "done")   return "✓";
  if (s === "active") return <span className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />;
  return "○";
}

export default function ProcessingPage() {
  const navigate = useNavigate();
  const { currentRunId, addRun } = useApp();
  const [status, setStatus] = useState({ status: "running", stage: "Queued", progress_pct: 0, elapsed_sec: 0 });
  const [longRunning, setLongRunning] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (!currentRunId) { navigate("/"); return; }

    const poll = async () => {
      try {
        const r = await getStatus(currentRunId);
        setStatus(prev =>
          prev.status === r.data.status && prev.stage === r.data.stage && prev.progress_pct === r.data.progress_pct
            ? prev
            : r.data
        );
        if (r.data.elapsed_sec > 60) setLongRunning(true);

        if (r.data.status === "completed") {
          clearInterval(intervalRef.current);
          setStatus(prev => ({ ...prev, stage: "Finishing up..." }));
          const res = await getResults(currentRunId);
          addRun({ run_id: currentRunId, ...res.data });
          navigate("/results");
        } else if (r.data.status === "error" || r.data.status === "cancelled") {
          clearInterval(intervalRef.current);
        }
      } catch (e) {
        clearInterval(intervalRef.current);
      }
    };

    poll();
    intervalRef.current = setInterval(poll, 2000);
    return () => clearInterval(intervalRef.current);
  }, [currentRunId]);

  async function handleCancel() {
    clearInterval(intervalRef.current);
    await cancelRun(currentRunId).catch(() => {});
    navigate("/configure");
  }

  const isError     = status.status === "error";
  const isCancelled = status.status === "cancelled";

  return (
    <div className="page-container">
      <StepIndicator current={6} />
      <h1 className="page-title">Processing</h1>
      <p className="page-subtitle">Running the balancing pipeline. Please wait…</p>

      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10, fontSize: 14, color: "var(--gray-500)" }}>
          <span>{status.stage}</span>
          <span>⏱ {status.elapsed_sec}s</span>
        </div>
        <div className="progress-bar-wrap">
          <div className="progress-bar-fill" style={{ width: `${status.progress_pct}%` }} />
        </div>
        <div style={{ textAlign: "right", fontSize: 12, color: "var(--gray-400)", marginTop: 4 }}>
          {status.progress_pct}%
        </div>
      </div>

      {longRunning && !isError && !isCancelled && (
        <div className="alert alert-info" style={{ marginBottom: 16 }}>
          ℹ Large datasets take longer. Your results are almost ready.
        </div>
      )}

      {isError && (
        <div className="alert alert-error" style={{ marginBottom: 16 }}>
          ✕ An error occurred: {status.error}
          <button className="btn btn-secondary btn-sm" style={{ marginLeft: 12 }} onClick={() => navigate("/configure")}>
            Try again
          </button>
        </div>
      )}

      {isCancelled && (
        <div className="alert alert-warning" style={{ marginBottom: 16 }}>
          Processing was cancelled.
        </div>
      )}

      <div className="card">
        <div className="section-label">Pipeline stages</div>
        <ul className="stage-list" style={{ marginTop: 12 }}>
          {STAGES.filter((s) => s !== "Done").map((s) => {
            const st = stageState(status.stage, s);
            return (
              <li key={s} className={`stage-item ${st}`}>
                <span className="stage-icon">{stageIcon(st)}</span>
                {s}
              </li>
            );
          })}
        </ul>
      </div>

      <div className="nav-row">
        {!isError && !isCancelled && (
          <button className="btn btn-danger btn-sm" onClick={handleCancel}>Cancel</button>
        )}
        {(isError || isCancelled) && (
          <button className="btn btn-secondary" onClick={() => navigate("/configure")}>← Back to config</button>
        )}
      </div>
    </div>
  );
}
