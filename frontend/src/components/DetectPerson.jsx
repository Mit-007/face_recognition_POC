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

  const thresholdPct = result?.threshold ? (result.threshold * 100).toFixed(0) : "50";

  return (
    <form className="card" onSubmit={onSubmit}>
      <h2>Detect Person</h2>
      <p className="subtitle">
        Capture a live photo to verify identity against the vector database.
      </p>

      <div className="field">
        <span>Photo to check</span>
        <CameraCapture captured={preview} onCapture={onCapture} overlay={overlayElement} />
      </div>

      <button type="submit" disabled={loading}>
        {loading && <span className="spinner" />}
        {loading ? "Analyzing Biometrics..." : "Detect Person"}
      </button>

      {error && <StatusBanner type="error">{error}</StatusBanner>}

      {/* Biometric Verification Card (Matched) */}
      {result && result.matched && (
        <div className="verification-card match-success">
          <div className="verification-header">
            <span className="badge-pill success">MATCH VERIFIED</span>
            {result.latency_ms && <span className="latency-tag">⚡ {result.latency_ms} ms</span>}
          </div>

          <div className="verification-body">
            <div className="verification-title">
              <h3>{result.name}</h3>
              <p>Identity confirmed in Pinecone vector index</p>
            </div>

            {/* Confidence Gauge */}
            <div className="confidence-meter">
              <div className="meter-label">
                <span>Cosine Similarity</span>
                <strong>
                  {(result.score * 100).toFixed(1)}% ({result.score.toFixed(3)})
                </strong>
              </div>
              <div className="meter-track">
                <div
                  className="meter-bar success"
                  style={{ width: `${Math.min(100, Math.max(0, result.score * 100))}%` }}
                />
                <div
                  className="meter-threshold-line"
                  style={{ left: `${thresholdPct}%` }}
                  title={`System Cutoff: ${thresholdPct}%`}
                />
              </div>
              <div className="meter-footer">
                <span>0%</span>
                <span className="cutoff-tag">System Cutoff: {thresholdPct}%</span>
                <span>100%</span>
              </div>
            </div>

            {/* Side-by-Side Reference */}
            {result.image_name && (
              <div className="reference-gallery">
                <p className="reference-title">Registered Reference Photo:</p>
                <img
                  className="matched-image"
                  src={imageUrl(result.image_name)}
                  alt={result.name}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Biometric Verification Card (Unrecognized Face Detected) */}
      {result && !result.matched && result.face_detected && (
        <div className="verification-card match-denied">
          <div className="verification-header">
            <span className="badge-pill denied">ACCESS DENIED</span>
            {result.latency_ms && <span className="latency-tag">⚡ {result.latency_ms} ms</span>}
          </div>

          <div className="verification-body">
            <div className="verification-title">
              <h3>Unrecognized Person</h3>
              <p>{result.message}</p>
            </div>

            {/* Confidence Gauge showing below threshold */}
            <div className="confidence-meter">
              <div className="meter-label">
                <span>Best Candidate Score</span>
                <strong>
                  {(result.score * 100).toFixed(1)}% ({result.score.toFixed(3)})
                </strong>
              </div>
              <div className="meter-track">
                <div
                  className="meter-bar denied"
                  style={{ width: `${Math.min(100, Math.max(0, result.score * 100))}%` }}
                />
                <div
                  className="meter-threshold-line"
                  style={{ left: `${thresholdPct}%` }}
                  title={`System Cutoff: ${thresholdPct}%`}
                />
              </div>
              <div className="meter-footer">
                <span>0%</span>
                <span className="cutoff-tag">Required Cutoff: {thresholdPct}%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Failure with no face detected */}
      {result && !result.matched && !result.face_detected && (
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
