import axios from "axios";

const api = axios.create({ baseURL: "http://localhost:8001/api" });

api.interceptors.response.use(
  (r) => r,
  (err) => {
    const msg =
      err.response?.data?.detail ||
      err.response?.data?.message ||
      err.message ||
      "An unexpected error occurred.";
    return Promise.reject(new Error(msg));
  }
);

// Upload
export const uploadDataset = (file) => {
  const fd = new FormData();
  fd.append("file", file);
  return api.post("/upload", fd);
};

// Dataset
export const getColumns = (datasetId) =>
  api.get("/dataset/columns", { params: { dataset_id: datasetId } });

export const getPreview = (datasetId, rows = 5) =>
  api.get("/dataset/preview", { params: { dataset_id: datasetId, rows } });

export const getColumnSummary = (datasetId, col) =>
  api.get("/dataset/column-summary", { params: { dataset_id: datasetId, col } });

// Validate
export const validateDataset = (datasetId, labelCol) =>
  api.post("/validate", { dataset_id: datasetId, label_col: labelCol });

// Diagnosis
export const getDiagnosis = (datasetId, labelCol) =>
  api.get("/diagnosis", { params: { dataset_id: datasetId, label_col: labelCol } });

// Balance
export const startBalancing = (datasetId, labelCol, method, params, heldOutSize) =>
  api.post("/balance", {
    dataset_id: datasetId,
    label_col: labelCol,
    method,
    params,
    held_out_size: heldOutSize,
  });

// Status / Cancel
export const getStatus = (runId) => api.get(`/status/${runId}`);
export const cancelRun = (runId) => api.post(`/cancel/${runId}`);

// Results
export const getResults = (runId) => api.get(`/results/${runId}`);

// Export
export const exportDatasetUrl = (runId) =>
  `http://localhost:8001/api/export/dataset/${runId}`;
export const exportSummaryUrl = (runId) =>
  `http://localhost:8001/api/export/summary/${runId}`;
export const exportAllUrl = (runId) =>
  `http://localhost:8001/api/export/all/${runId}`;
