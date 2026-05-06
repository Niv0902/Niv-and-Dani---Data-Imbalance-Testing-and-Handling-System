import { useNavigate } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer,
} from "recharts";
import StepIndicator from "../components/StepIndicator";
import { useApp } from "../context/AppContext";

const BEFORE_COLOR = "#9ca3af";
const AFTER_COLOR  = "#2563eb";

function irSeverity(ir) {
  if (ir < 2)  return { label: "Low",     color: "#16a34a" };
  if (ir < 5)  return { label: "Medium",  color: "#d97706" };
  if (ir < 10) return { label: "High",    color: "#ea580c" };
  return             { label: "Extreme",  color: "#dc2626" };
}

function SingleDistChart({ dist, classNames, color, title }) {
  const data = classNames.map((name) => ({
    name: name.length > 12 ? name.slice(0, 12) + "…" : name,
    count: dist[name] || 0,
  }));
  const height = Math.min(40 * classNames.length + 60, 280);
  return (
    <div>
      <div style={{ textAlign: "center", fontWeight: 600, fontSize: 13, marginBottom: 6, color: "var(--gray-600)" }}>
        {title}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 16 }}>
          <XAxis type="number" tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={80} />
          <Tooltip />
          <Bar dataKey="count" fill={color} radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ClassCountTable({ distBefore, distAfter, classNames }) {
  const totalB = Object.values(distBefore).reduce((a, b) => a + b, 0);
  const totalA = Object.values(distAfter).reduce((a, b) => a + b, 0);
  const totalD = totalA - totalB;
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Class</th>
          <th>Original dataset</th>
          <th>After balancing</th>
          <th>Change</th>
        </tr>
      </thead>
      <tbody>
        {classNames.map((name) => {
          const b = distBefore[name] || 0;
          const a = distAfter[name]  || 0;
          const d = a - b;
          return (
            <tr key={name}>
              <td><strong>{name}</strong></td>
              <td>{b.toLocaleString()}</td>
              <td>{a.toLocaleString()}</td>
              <td className={d > 0 ? "positive" : d < 0 ? "negative" : ""}>
                {d > 0 ? `+${d}` : d}
              </td>
            </tr>
          );
        })}
        <tr style={{ fontWeight: 700, borderTop: "2px solid var(--gray-200)" }}>
          <td>Total</td>
          <td>{totalB.toLocaleString()}</td>
          <td>{totalA.toLocaleString()}</td>
          <td className={totalD > 0 ? "positive" : totalD < 0 ? "negative" : ""}>
            {totalD > 0 ? `+${totalD}` : totalD}
          </td>
        </tr>
      </tbody>
    </table>
  );
}

export default function ResultsPage() {
  const navigate = useNavigate();
  const { runs, currentRunId } = useApp();

  const run = runs.find((r) => r.run_id === currentRunId) || runs[runs.length - 1];

  if (!run) {
    return (
      <div className="page-container">
        <div className="alert alert-warning">No results available. Please run the balancing pipeline first.</div>
        <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => navigate("/configure")}>
          Go to configuration
        </button>
      </div>
    );
  }

  const {
    class_distribution_before, class_distribution_after,
    class_names, ir_before, ir_after, method, elapsed_seconds,
    total_before, held_out_size,
  } = run;

  const heldOutPct  = Math.round((held_out_size ?? 0.2) * 100);
  const trainPct    = 100 - heldOutPct;
  const heldOutRows = Math.round(total_before * (held_out_size ?? 0.2));
  const trainRows   = total_before - heldOutRows;

  const irImproved = ir_after < ir_before;
  const sevBefore  = irSeverity(ir_before);
  const sevAfter   = irSeverity(ir_after);

  return (
    <div className="page-container">
      <StepIndicator current={7} />
      <h1 className="page-title">Results & Comparison</h1>
      <p className="page-subtitle">
        Before vs. after balancing — method: <strong>{method.toUpperCase()}</strong>
        &nbsp;·&nbsp; elapsed: {elapsed_seconds}s
      </p>

      <div style={{ background: "#eff6ff", border: "1px solid #bfdbfe", borderRadius: 8, padding: "10px 14px", marginBottom: 16, fontSize: 13, color: "#1e40af" }}>
        <strong>How to read this page:</strong> "Before" = the full original dataset ({total_before.toLocaleString()} rows).
        "After" = the {trainPct}% training portion ({trainRows.toLocaleString()} rows) after balancing.
        The remaining {heldOutPct}% ({heldOutRows.toLocaleString()} rows) is the held-out set — excluded from balancing to prevent data leakage and not included in the export.
        A reduction in majority class count reflects this split, not row deletion by the method.
      </div>

      {!irImproved && (
        <div className="alert alert-warning" style={{ marginBottom: 16 }}>
          ⚠ The imbalance ratio did not improve. Try a different method or parameters.
        </div>
      )}

      {/* IR with severity labels */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-label">Imbalance Ratio</div>
        <div className="ir-compare">
          <div>
            <div className="compare-label before">Before</div>
            <div style={{ fontSize: 32, fontWeight: 700 }}>{ir_before}</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: sevBefore.color, marginTop: 4 }}>
              {sevBefore.label}
            </div>
          </div>
          <div className="ir-arrow">→</div>
          <div>
            <div className="compare-label after">After (balanced portion)</div>
            <div style={{ fontSize: 32, fontWeight: 700, color: irImproved ? "var(--green)" : "var(--red)" }}>
              {ir_after}
            </div>
            <div style={{ fontSize: 13, fontWeight: 600, color: sevAfter.color, marginTop: 4 }}>
              {sevAfter.label}
            </div>
          </div>
          <div>
            <span className={`badge ${irImproved ? "badge-pass" : "badge-error"}`}>
              {irImproved
                ? `↓ ${((1 - ir_after / ir_before) * 100).toFixed(1)}% reduction`
                : "↑ increased"}
            </span>
          </div>
        </div>
      </div>

      {/* Side-by-side distribution bar charts */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-label">Class distribution</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          <SingleDistChart
            dist={class_distribution_before}
            classNames={class_names}
            color={BEFORE_COLOR}
            title={`Original dataset (${total_before.toLocaleString()} rows)`}
          />
          <SingleDistChart
            dist={class_distribution_after}
            classNames={class_names}
            color={AFTER_COLOR}
            title={`After balancing — training ${trainPct}% (${trainRows.toLocaleString()} rows)`}
          />
        </div>
      </div>

      {/* Class count table with totals */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-label">Sample counts per class</div>
        <div className="table-scroll">
          <ClassCountTable
            distBefore={class_distribution_before}
            distAfter={class_distribution_after}
            classNames={class_names}
          />
        </div>
      </div>

      <div className="nav-row">
        <button className="btn btn-secondary" onClick={() => navigate("/configure")}>Try another method</button>
        <div className="nav-row-right">
          {runs.length > 1 && (
            <button className="btn btn-secondary" onClick={() => navigate("/history")}>Compare runs</button>
          )}
          <button className="btn btn-primary" onClick={() => navigate("/export")}>Continue to export →</button>
        </div>
      </div>
    </div>
  );
}
