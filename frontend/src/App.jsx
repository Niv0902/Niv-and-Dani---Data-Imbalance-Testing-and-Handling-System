import { BrowserRouter, Routes, Route, Link, useNavigate } from "react-router-dom";
import { AppProvider, useApp } from "./context/AppContext";
import UploadPage from "./pages/UploadPage";
import ColumnSelectionPage from "./pages/ColumnSelectionPage";
import ValidationPage from "./pages/ValidationPage";
import DiagnosisPage from "./pages/DiagnosisPage";
import BalancingConfigPage from "./pages/BalancingConfigPage";
import ProcessingPage from "./pages/ProcessingPage";
import ResultsPage from "./pages/ResultsPage";
import ExportPage from "./pages/ExportPage";
import RunHistoryPage from "./pages/RunHistoryPage";

function Header() {
  const { runs, reset } = useApp();
  const navigate = useNavigate();
  return (
    <header className="app-header">
      <div
        className="logo"
        onClick={() => { reset(); navigate("/"); }}
        style={{ cursor: "pointer" }}
      >
        ⚖ ImbalanceKit
      </div>
      <nav>
        {runs.length > 0 && (
          <Link to="/history">History ({runs.length})</Link>
        )}
        <a
          href="https://github.com/Niv0902"
          target="_blank"
          rel="noreferrer"
          style={{ marginLeft: 16 }}
        >
          GitHub
        </a>
      </nav>
    </header>
  );
}

function AppInner() {
  return (
    <>
      <Header />
      <main>
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/columns" element={<ColumnSelectionPage />} />
          <Route path="/validate" element={<ValidationPage />} />
          <Route path="/diagnosis" element={<DiagnosisPage />} />
          <Route path="/configure" element={<BalancingConfigPage />} />
          <Route path="/processing" element={<ProcessingPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/export" element={<ExportPage />} />
          <Route path="/history" element={<RunHistoryPage />} />
          <Route path="*" element={<UploadPage />} />
        </Routes>
      </main>
      <footer className="app-footer">
        ImbalanceKit · Braude College of Engineering · Capstone Project 26-1-D-3 ·
        Niv Oren &amp; Daniel Levovsky
      </footer>
    </>
  );
}

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <AppInner />
      </BrowserRouter>
    </AppProvider>
  );
}
