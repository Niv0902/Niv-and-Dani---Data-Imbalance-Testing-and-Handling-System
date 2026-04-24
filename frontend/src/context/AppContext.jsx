import { createContext, useContext, useState } from "react";

const AppContext = createContext(null);

export function AppProvider({ children }) {
  const [datasetId, setDatasetId] = useState(null);
  const [datasetMeta, setDatasetMeta] = useState(null); // {filename, rows, columns}
  const [labelCol, setLabelCol] = useState(null);
  const [validationResult, setValidationResult] = useState(null);
  const [diagnosisResult, setDiagnosisResult] = useState(null);
  const [currentRunId, setCurrentRunId] = useState(null);
  const [runs, setRuns] = useState([]); // [{run_id, method, result, ...}]

  function addRun(run) {
    setRuns((prev) => [...prev, run]);
  }

  function reset() {
    setDatasetId(null);
    setDatasetMeta(null);
    setLabelCol(null);
    setValidationResult(null);
    setDiagnosisResult(null);
    setCurrentRunId(null);
    setRuns([]);
  }

  return (
    <AppContext.Provider
      value={{
        datasetId, setDatasetId,
        datasetMeta, setDatasetMeta,
        labelCol, setLabelCol,
        validationResult, setValidationResult,
        diagnosisResult, setDiagnosisResult,
        currentRunId, setCurrentRunId,
        runs, addRun,
        reset,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside AppProvider");
  return ctx;
}
