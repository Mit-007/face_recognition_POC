import { useState } from "react";
import { detectPerson, imageUrl } from "../api";
import CameraCapture from "./CameraCapture";
import StatusBanner from "./StatusBanner";

export default function DetectPerson() {
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

  return (
    <form className="card" onSubmit={onSubmit}>
      <h2>Detect Person</h2>
      <p className="subtitle">Capture a live photo to check it against registered people.</p>

      <div className="field">
        <span>Photo to check</span>
        <CameraCapture captured={preview} onCapture={onCapture} />
      </div>

      <button type="submit" disabled={loading}>
        {loading && <span className="spinner" />}
        {loading ? "Checking..." : "Detect Person"}
      </button>

      {error && <StatusBanner type="error">{error}</StatusBanner>}

      {result && result.matched && (
        <div className="result">
          <StatusBanner type="success">
            Match found: <strong>{result.name}</strong> (score: {result.score.toFixed(3)})
          </StatusBanner>
          <img
            className="matched-image"
            src={imageUrl(result.image_name)}
            alt={result.name}
          />
        </div>
      )}

      {result && !result.matched && (
        <StatusBanner type="error">{result.message}</StatusBanner>
      )}
    </form>
  );
}
