import { useState } from "react";
import { detectPerson, imageUrl } from "../api";
import CameraCapture from "./CameraCapture";
import FaceOverlay from "./FaceOverlay";
import StatusBanner from "./StatusBanner";

const DECISIONS = {
  verified: { badge: "✓ VERIFIED", title: null },
  retake: { badge: "PLEASE RETAKE", title: "Not Sure Who This Is" },
  unknown: { badge: "NOT VERIFIED", title: "Unknown Person" },
};

export default function DetectPerson({ onNavigateHistory }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const onCapture = (capturedFile, previewUrl) => {
    setFile(capturedFile);
    setPreview(previewUrl);
    setResult(null);
    setError(null);
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please capture a photo first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await detectPerson(file);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Detection failed.");
    } finally {
      setLoading(false);
    }
  };

  const overlayElement =
    result && result.bbox && result.image_dimensions ? (
      <FaceOverlay
        bbox={result.bbox}
        kps={result.kps}
        imageDimensions={result.image_dimensions}
        name={result.name}
        decision={result.decision}
        matched={result.matched}
        latencyMs={result.latency_ms}
      />
    ) : null;

  const decision = result ? DECISIONS[result.decision] : null;

  return (
    <form className="card" onSubmit={onSubmit}>
      <h2>Detect Person</h2>
      <p className="subtitle">
        Capture a live photo to verify identity using Classical Face Similarity (LBP + HOG).
      </p>

      <div className="field">
        <span>Photo to check</span>
        <CameraCapture captured={preview} onCapture={onCapture} overlay={overlayElement} />
      </div>

      <button type="submit" disabled={loading}>
        {loading && <span className="spinner" />}
        {loading ? "Comparing Face Descriptors..." : "Detect Person"}
      </button>

      {error && <StatusBanner type="error">{error}</StatusBanner>}

      {/* Verification Card */}
      {result && result.face_detected && (
        <div className={`verification-card ${result.matched ? "match-success" : "match-denied"}`}>
          <div className="verification-header">
            <span className={`badge-pill ${result.matched ? "success" : "denied"}`}>
              {decision.badge}
            </span>
            {result.latency_ms && <span className="latency-tag">⚡ {result.latency_ms} ms</span>}
          </div>

          <div className="verification-body">
            <div className="verification-title">
              <h3>{result.matched ? result.name : decision.title}</h3>
              <p>{result.message}</p>
            </div>

            {/* Feature Distance Chips: lower is closer, and these are not percentages */}
            {result.final_distance !== undefined && (
              <div className="classical-chips">
                <span className="classical-chip">
                  Closest Distance: <strong>{result.final_distance.toFixed(3)}</strong> (limit {result.cutoff.toFixed(3)})
                </span>
                {result.second_distance !== null && (
                  <span className="classical-chip">
                    Second Closest: <strong>{result.second_distance.toFixed(3)}</strong> (needs a gap of {result.margin.toFixed(3)})
                  </span>
                )}
                <span className="classical-chip">
                  LBP Texture (Chi²): <strong>{result.lbp_distance.toFixed(3)}</strong>
                </span>
                <span className="classical-chip">
                  HOG Shape (Cosine): <strong>{result.hog_distance.toFixed(3)}</strong>
                </span>
              </div>
            )}

            {/* Side-by-Side Reference */}
            {result.image_name && (
              <div className="reference-gallery">
                <p className="reference-title">Registered Reference Portrait:</p>
                <img
                  className="matched-image"
                  src={imageUrl(result.image_name)}
                  alt={result.name}
                />
              </div>
            )}

            {/* Score Analysis: every registered person, closest first */}
            {result.ranked_candidates.length > 0 && (
              <div className="comparison-section">
                <h4 className="comparison-title">
                  All Registered People ({result.ranked_candidates.length}), Closest First
                </h4>
                <p className="comparison-hint">Lower distance means more alike. These are not percentages.</p>
                <div className="comparison-table-wrap">
                  <table className="comparison-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Person</th>
                        <th>Distance</th>
                        <th>LBP</th>
                        <th>HOG</th>
                        <th>Within Limit</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.ranked_candidates.map((candidate, index) => (
                        <tr key={candidate.person_id}>
                          <td>{index + 1}</td>
                          <td>
                            <div className="comparison-person">
                              <img src={imageUrl(candidate.image_name)} alt={candidate.name} />
                              <span>{candidate.name}</span>
                            </div>
                          </td>
                          <td><strong>{candidate.final_distance.toFixed(3)}</strong></td>
                          <td>{candidate.lbp_distance.toFixed(3)}</td>
                          <td>{candidate.hog_distance.toFixed(3)}</td>
                          <td>{candidate.final_distance <= result.cutoff ? "Yes" : "No"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Failure with no face detected */}
      {result && !result.face_detected && (
        <StatusBanner type="error">{result.message}</StatusBanner>
      )}

      {result && onNavigateHistory && (
        <button
          type="button"
          className="view-history-link-btn"
          onClick={onNavigateHistory}
        >
          📋 View Full Detection History Table →
        </button>
      )}
    </form>
  );
}
