const STEPS = [
  "Upload",
  "Columns",
  "Validate",
  "Diagnose",
  "Configure",
  "Processing",
  "Results",
  "Export",
  "History",
];

export default function StepIndicator({ current }) {
  return (
    <div className="step-indicator">
      {STEPS.map((label, i) => {
        const num = i + 1;
        const isDone = num < current;
        const isActive = num === current;
        return (
          <div className="step-item" key={num}>
            {i > 0 && <div className={`step-connector ${isDone ? "done" : ""}`} />}
            <div className={`step-circle ${isDone ? "done" : isActive ? "active" : ""}`}>
              {isDone ? "✓" : num}
            </div>
            <span className={`step-label ${isDone ? "done" : isActive ? "active" : ""}`}>
              {label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
