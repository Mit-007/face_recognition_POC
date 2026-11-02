import { useState } from "react";
import { detectPerson, imageUrl } from "../api";
import CameraCapture from "./CameraCapture";
import FaceOverlay from "./FaceOverlay";
import StatusBanner from "./StatusBanner";

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
        score={result.score}
        matched={result.matched}
        latencyMs={result.latency_ms}
      />
    ) : null;

  const thresholdPct = result ? (result.threshold * 100).toFixed(0) : null;
  const barStatus = result?.matched ? "success" : "denied";

  return (
    <form className="card" onSubmit={onSubmit}>
      <h2>Detect Person</h2>
      <p className="subtitle">
        Capture a live photo to verify identity and find the most similar registered person via Classical Face Similarity (LBP + HOG).
      </p>

      <div className="field">
        <span>Photo to check</span>
        <CameraCapture captured={preview} onCapture={onCapture} overlay={overlayElement} />
      </div>

      <button type="submit" disabled={loading}>
        {loading && <span className="spinner" />}
        {loading ? "Comparing Face Descriptors..." : "Detect & Find Most Similar"}
      </button>

      {error && <StatusBanner type="error">{error}</StatusBanner>}

      {/* Verification Card (matched or unrecognized face) */}
      {result && result.face_detected && (
        <div className={`verification-card ${result.matched ? "match-success" : "match-denied"}`}>
          <div className="verification-header">
            <span className={`badge-pill ${barStatus}`}>
              {result.matched ? "✓ MATCH VERIFIED" : "NO MATCH VERIFIED"}
            </span>
            {result.latency_ms && <span className="latency-tag">⚡ {result.latency_ms} ms</span>}
          </div>

          <div className="verification-body">
            <div className="verification-title">
              <h3>{result.matched ? result.name : "Unrecognized Person"}</h3>
              <p>
                {result.matched
                  ? `Top candidate matched with distance below cutoff (${result.final_distance?.toFixed(3)} ≤ ${(1 - result.threshold).toFixed(3)})`
                  : result.message}
              </p>
            </div>

            {/* Feature Distance Chips */}
            {result.final_distance !== undefined && (
              <div className="classical-chips">
                <span className="classical-chip">
                  {result.matched ? "Weighted Distance" : "Closest Distance"}: <strong>{result.final_distance?.toFixed(4)}</strong>
                </span>
                <span className="classical-chip">
                  LBP Texture (Chi²): <strong>{result.lbp_distance?.toFixed(4)}</strong>
                </span>
                <span className="classical-chip">
                  HOG Shape (Cosine): <strong>{result.hog_distance?.toFixed(4)}</strong>
                </span>
              </div>
            )}

            {/* Similarity Meter */}
            <div className="confidence-meter">
              <div className="meter-label">
                <span>{result.matched ? "Resemblance Score (1 - Distance)" : "Closest Candidate Resemblance"}</span>
                <strong>
                  {(result.score * 100).toFixed(1)}% ({result.score.toFixed(3)})
                </strong>
              </div>
              <div className="meter-track">
                <div
                  className={`meter-bar ${barStatus}`}
                  style={{ width: `${Math.min(100, Math.max(0, result.score * 100))}%` }}
                />
                <div
                  className="meter-threshold-line"
                  style={{ left: `${thresholdPct}%` }}
                  title={`Match Cutoff: ${thresholdPct}%`}
                />
              </div>
              <div className="meter-footer">
                <span>0%</span>
                <span className="cutoff-tag">Match Cutoff: {thresholdPct}%</span>
                <span>100%</span>
              </div>
            </div>

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

            {/* Ranked Candidates, also shown on denial to reveal the closest matches */}
            {result.ranked_candidates && result.ranked_candidates.length > 0 && (
              <div className="ranking-section">
                <h4 className="ranking-title">
                  Database Distance Ranking ({result.matched ? "Most Similar First" : "Closest Resemblances"})
                </h4>
                <div className="ranking-list">
                  {result.ranked_candidates.map((cand, idx) => (
                    <div key={cand.person_id || idx} className="ranking-item">
                      <span className="ranking-rank">#{idx + 1}</span>
                      <div className="ranking-person">
                        {cand.image_name && (
                          <img
                            src={imageUrl(cand.image_name)}
                            alt={cand.name}
                            className="ranking-thumb"
                          />
                        )}
                        <span className="ranking-name">{cand.name}</span>
                      </div>
                      <div className="ranking-metrics">
                        <span className="ranking-dist">dist: {cand.final_distance?.toFixed(3)}</span>
                        <span className="ranking-score">
                          {(cand.similarity_score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
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
