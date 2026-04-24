import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  ResponsiveContainer, Cell,
} from "recharts";
import StepIndicator from "../components/StepIndicator";
import ConfusionMatrix from "../components/ConfusionMatrix";
import { useApp } from "../context/AppContext";

const BEFORE_COLOR = "#9ca3af";
const AFTER_COLOR = "#2563eb";

const METRIC_LABELS = {
  accuracy: "Accuracy",
  macro_f1: "Macro F1",
  macro_precision: "Macro Precision",
  macro_recall: "Macro Recall",
  minority_recall: "Minority Recall",
  minority_precision: "Minority Precision",
  minority_f1: "Minority F1",
};

function delta(before, after) {
  const d = after - before;
  const sign = d >= 0 ? "+" : "";
  return { text: `${sign}${(d * 100).toFixed(1)}%`, positive: d >= 0 };
}

function MetricsTable({ before, after }) {
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Before</th>
          <th>After</th>
          <th>Change</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(METRIC_LABELS).map(([key, label]) => {
          const b = before[key] ?? 0;
          const a = after[key] ?? 0;
          const d = delta(b, a);
          return (
            <tr key={key}>
              <td>{label}</td>
              <td>{(b * 100).toFixed(1)}%</td>
              <td>{(a * 100).toFixed(1)}%</td>
              <td className={d.positive ? "positive" : "negative"}>{d.text}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function DistributionCompare({ before, after, classNames }) {
  const data = classNames.map((name) => ({
    name: name.length > 12 ? name.slice(0, 12) + "…" : name,
    Before: before[name] || 0,
    After: after[name] || 0,
  }));
  return (
    <ResponsiveContainer width="100%" height={Math.min(50 * classNames.length + 80, 300)}>
      <BarChart data={data} layout="vertical" margin={{ left: 90 }}>
        <XAxis type="number" tick={{ fontSize: 12 }} />
        <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={90} />
        <Tooltip />
        <Legend />
        <Bar dataKey="Before" fill={BEFORE_COLOR} />
        <Bar dataKey="After" fill={AFTER_COLOR} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function ResultsPage() {
  const navigate = useNavigate();
  const { runs, currentRunId } = useApp();
  const [showPerClass, setShowPerClass] = useState(false);

  const run = runs.find((r) => r.run_id === currentRunId) || runs[runs.length - 1];

  if (!run) {
    return (
      <div className="page-container">
        <div className="alert alert-warning">No results available. Please run the balancing pipeline first.</div>
        <button className="btn btn-primary" style={{ marginTop: 16 }} onClick={() => navigate("/configure")}>Go to configuration</button>
      </div>
    );
  }

  const { metrics_before, metrics_after, confusion_matrix_before, confusion_matrix_after,
    class_distribution_before, class_distribution_after, class_names,
    ir_before, ir_after, method, params, elapsed_seconds } = run;

  const irImproved = ir_after < ir_before;
  const minority_recall_dropped = metrics_after.minority_recall < metrics_before.minority_recall;

  return (
    <div className="page-container">
      <StepIndicator current={7} />
      <h1 className="page-title">Results & Comparison</h1>
      <p className="page-subtitle">
        Before vs. after balancing — method: <strong>{method.toUpperCase()}</strong> &nbsp;·&nbsp; elapsed: {elapsed_seconds}s
      </p>

      {minority_recall_dropped && (
        <div className="alert alert-warning" style={{ marginBottom: 16 }}>
          ⚠ Minority recall decreased after balancing. This can occur when the method and dataset interact poorly. Try a different method or parameters.
        </div>
      )}

      {/* IR comparison */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-label">Imbalance Ratio</div>
        <div className="ir-compare">
          <div>
            <div className="compare-label before">Before</div>
            <div style={{ fontSize: 32, fontWeight: 700 }}>{ir_before}</div>
          </div>
          <div className="ir-arrow">→</div>
          <div>
            <div className="compare-label after">After (train set)</div>
            <div style={{ fontSize: 32, fontWeight: 700, color: irImproved ? "var(--green)" : "var(--red)" }}>
              {ir_after}
            </div>
          </div>
          <div>
            <span className={`badge ${irImproved ? "badge-pass" : "badge-error"}`}>
              {irImproved ? `↓ ${((1 - ir_after / ir_before) * 100).toFixed(1)}% reduction` : "↑ increased"}
            </span>
          </div>
        </div>
      </div>

      {/* Metrics table */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-label">Performance metrics</div>
        <div className="table-scroll">
          <MetricsTable before={metrics_before} after={metrics_after} />
        </div>
      </div>

      {/* Class distribution */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-label">Class distribution (training set)</div>
        <DistributionCompare
          before={class_distribution_before}
          after={class_distribution_after}
          classNames={class_names}
        />
      </div>

      {/* Confusion matrices */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-label">Confusion matrices</div>
        <div className="compare-grid">
          <div>
            <div className="compare-label before">Before balancing</div>
            <ConfusionMatrix matrix={confusion_matrix_before} classNames={class_names} />
          </div>
          <div>
            <div className="compare-label after">After balancing</div>
            <ConfusionMatrix matrix={confusion_matrix_after} classNames={class_names} />
          </div>
        </div>
      </div>

      {/* Per-class breakdown */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="collapsible-header" onClick={() => setShowPerClass((o) => !o)}>
          <span>Per-class breakdown</span>
          <span>{showPerClass ? "▲" : "▼"}</span>
        </div>
        {showPerClass && (
          <div className="collapsible-content table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision Before</th>
                  <th>Precision After</th>
                  <th>Recall Before</th>
                  <th>Recall After</th>
                  <th>F1 Before</th>
                  <th>F1 After</th>
                </tr>
              </thead>
              <tbody>
                {metrics_before.per_class.map((b, i) => {
                  const a = metrics_after.per_class[i] || {};
                  return (
                    <tr key={b.class_name}>
                      <td><strong>{b.class_name}</strong></td>
                      <td>{(b.precision * 100).toFixed(1)}%</td>
                      <td className={a.precision >= b.precision ? "positive" : "negative"}>{((a.precision || 0) * 100).toFixed(1)}%</td>
                      <td>{(b.recall * 100).toFixed(1)}%</td>
                      <td className={a.recall >= b.recall ? "positive" : "negative"}>{((a.recall || 0) * 100).toFixed(1)}%</td>
                      <td>{(b.f1 * 100).toFixed(1)}%</td>
                      <td className={a.f1 >= b.f1 ? "positive" : "negative"}>{((a.f1 || 0) * 100).toFixed(1)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
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
