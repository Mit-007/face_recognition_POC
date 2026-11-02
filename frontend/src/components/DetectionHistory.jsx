import { useEffect, useState } from "react";
import { getDetectionHistory, clearDetectionHistory, imageUrl } from "../api";

export default function DetectionHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterStatus, setFilterStatus] = useState("all"); // 'all' | 'matched' | 'rejected'
  const [searchQuery, setSearchQuery] = useState("");
  const [clearing, setClearing] = useState(false);

  const fetchHistory = async () => {
    try {
      const res = await getDetectionHistory(100);
      setHistory(res.data.history || []);
    } catch {
      // Fallback on error
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let ignore = false;
    getDetectionHistory(100)
      .then((res) => {
        if (!ignore) {
          setHistory(res.data.history || []);
          setLoading(false);
        }
      })
      .catch(() => {
        if (!ignore) setLoading(false);
      });

    return () => {
      ignore = true;
    };
  }, []);

  const handleClear = async () => {
    if (!window.confirm("Are you sure you want to clear all detection history logs?")) return;
    setClearing(true);
    try {
      await clearDetectionHistory();
      setHistory([]);
    } catch {
      alert("Failed to clear history.");
    } finally {
      setClearing(false);
    }
  };

  // Filter & Search
  const filtered = history.filter((item) => {
    const matchesStatus =
      filterStatus === "all" ||
      (filterStatus === "matched" && item.matched === 1) ||
      (filterStatus === "rejected" && item.matched === 0);

    const name = item.name ? item.name.toLowerCase() : "";
    const msg = item.message ? item.message.toLowerCase() : "";
    const query = searchQuery.toLowerCase().trim();
    const matchesSearch = !query || name.includes(query) || msg.includes(query);

    return matchesStatus && matchesSearch;
  });

  const totalScans = history.length;
  const matchedCount = history.filter((h) => h.matched === 1).length;
  const rejectedCount = totalScans - matchedCount;
  const matchRate = totalScans > 0 ? ((matchedCount / totalScans) * 100).toFixed(0) : "0";

  return (
    <div className="card history-page-card">
      <div className="history-header">
        <div>
          <h2>Detection History Log</h2>
          <p className="subtitle">
            Audit log of all facial recognition scans, confidence scores, and timestamps.
          </p>
        </div>
        <div className="history-top-actions">
          <button
            type="button"
            className="history-btn secondary"
            onClick={fetchHistory}
            disabled={loading}
          >
            {loading ? "Refreshing..." : "↻ Refresh"}
          </button>
          {history.length > 0 && (
            <button
              type="button"
              className="history-btn danger"
              onClick={handleClear}
              disabled={clearing}
            >
              {clearing ? "Clearing..." : "Clear Log"}
            </button>
          )}
        </div>
      </div>

      {/* Analytics Summary Stats */}
      <div className="history-stats-grid">
        <div className="stat-card">
          <span className="stat-label">Total Scans</span>
          <strong className="stat-value">{totalScans}</strong>
        </div>
        <div className="stat-card success">
          <span className="stat-label">Verified Matches</span>
          <strong className="stat-value">{matchedCount}</strong>
          <span className="stat-sub">{matchRate}% success rate</span>
        </div>
        <div className="stat-card error">
          <span className="stat-label">Rejected / Unknown</span>
          <strong className="stat-value">{rejectedCount}</strong>
          <span className="stat-sub">
            {totalScans > 0 ? (100 - parseFloat(matchRate)).toFixed(0) : 0}% rejected
          </span>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="history-filters-bar">
        <div className="filter-chips">
          <button
            type="button"
            className={`filter-chip ${filterStatus === "all" ? "active" : ""}`}
            onClick={() => setFilterStatus("all")}
          >
            All ({totalScans})
          </button>
          <button
            type="button"
            className={`filter-chip success ${filterStatus === "matched" ? "active" : ""}`}
            onClick={() => setFilterStatus("matched")}
          >
            ✓ Matched ({matchedCount})
          </button>
          <button
            type="button"
            className={`filter-chip error ${filterStatus === "rejected" ? "active" : ""}`}
            onClick={() => setFilterStatus("rejected")}
          >
            ✕ Rejected ({rejectedCount})
          </button>
        </div>

        <div className="search-box">
          <input
            type="text"
            placeholder="Search by name..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* History Table */}
      <div className="history-table-container">
        {loading && history.length === 0 ? (
          <div className="history-loading">
            <span className="spinner dark" />
            <p>Loading detection records...</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="history-empty">
            <span className="empty-icon">📋</span>
            <h3>No detection records found</h3>
            <p>
              {history.length === 0
                ? "Scans performed in the 'Detect Person' tab will automatically be recorded here."
                : "No log records match your current filter."}
            </p>
          </div>
        ) : (
          <table className="history-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Status</th>
                <th>Identified Person</th>
                <th>Confidence Score</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((item) => {
                const isMatch = item.matched === 1;
                const scorePct = (item.score * 100).toFixed(1);
                const thresholdPct = (item.threshold * 100).toFixed(0);

                return (
                  <tr key={item.id} className={isMatch ? "row-matched" : "row-rejected"}>
                    <td className="col-time">
                      <span className="time-main">{item.timestamp}</span>
                    </td>
                    <td className="col-status">
                      <span className={`status-pill-badge ${isMatch ? "matched" : "rejected"}`}>
                        {isMatch ? "✓ VERIFIED" : "✕ REJECTED"}
                      </span>
                    </td>
                    <td className="col-person">
                      <div className="person-cell">
                        {isMatch && item.image_name ? (
                          <img
                            className="table-avatar"
                            src={imageUrl(item.image_name)}
                            alt={item.name || "Person"}
                          />
                        ) : (
                          <div className="table-avatar-placeholder">
                            <span>👤</span>
                          </div>
                        )}
                        <div className="person-cell-info">
                          <strong className="person-cell-name">
                            {isMatch ? item.name : "Unrecognized Person"}
                          </strong>
                          <span className="person-cell-sub">
                            {isMatch ? "Authorized Identity" : item.message || "Unknown visitor"}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td className="col-score">
                      <div className="score-cell">
                        <div className="score-text">
                          <strong>{scorePct}%</strong>
                          <span className="score-raw">({item.score.toFixed(3)})</span>
                        </div>
                        <div className="table-meter-track">
                          <div
                            className={`table-meter-bar ${isMatch ? "success" : "rejected"}`}
                            style={{ width: `${Math.min(100, Math.max(0, item.score * 100))}%` }}
                          />
                        </div>
                        <span className="score-cutoff-hint">Cutoff: {thresholdPct}%</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
