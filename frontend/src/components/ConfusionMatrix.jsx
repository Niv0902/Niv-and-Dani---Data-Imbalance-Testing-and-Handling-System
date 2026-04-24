export default function ConfusionMatrix({ matrix, classNames }) {
  if (!matrix || !matrix.length) return null;
  const n = matrix.length;
  const labels = classNames || matrix.map((_, i) => `Class ${i}`);

  // Scale cell colour by value relative to row max
  function cellBg(row, col) {
    const rowMax = Math.max(...matrix[row]);
    if (rowMax === 0) return "#fff";
    const ratio = matrix[row][col] / rowMax;
    const isDiag = row === col;
    if (isDiag) {
      const g = Math.round(240 - ratio * 100);
      return `rgb(${g}, ${Math.round(220 - ratio * 50)}, ${g})`;
    }
    const r = Math.round(255 - ratio * 40);
    return `rgb(${r}, ${Math.round(240 - ratio * 60)}, ${Math.round(240 - ratio * 60)})`;
  }

  return (
    <div className="cm-wrap">
      <table className="cm-table">
        <thead>
          <tr>
            <th style={{ background: "transparent", border: "none" }} />
            <th colSpan={n} style={{ textAlign: "center", color: "var(--gray-500)", fontWeight: 600 }}>
              Predicted
            </th>
          </tr>
          <tr>
            <th style={{ background: "transparent", border: "none" }} />
            {labels.map((l) => (
              <th key={l}>{String(l).length > 8 ? String(l).slice(0, 8) + "…" : l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, r) => (
            <tr key={r}>
              {r === 0 && (
                <th
                  rowSpan={n}
                  style={{
                    writingMode: "vertical-rl",
                    transform: "rotate(180deg)",
                    textAlign: "center",
                    color: "var(--gray-500)",
                    fontWeight: 600,
                    background: "var(--gray-50)",
                  }}
                >
                  Actual
                </th>
              )}
              {row.map((val, c) => (
                <td
                  key={c}
                  className={r === c ? "cm-diag" : ""}
                  style={{ background: cellBg(r, c), minWidth: 52 }}
                  title={`Actual: ${labels[r]}, Predicted: ${labels[c]}, Count: ${val}`}
                >
                  {val}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
