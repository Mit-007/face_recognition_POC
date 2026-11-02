import { useState } from "react";
import { registerPerson } from "../api";
import CameraCapture from "./CameraCapture";
import StatusBanner from "./StatusBanner";

export default function RegisterPerson({ onNavigateUsers }) {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  const onCapture = (capturedFile, previewUrl) => {
    setFile(capturedFile);
    setPreview(previewUrl);
    setStatus(null);
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!firstName.trim()) {
      setStatus({ type: "error", message: "Please enter a first name." });
      return;
    }
    if (!file) {
      setStatus({ type: "error", message: "Please capture a face photo." });
      return;
    }

    setLoading(true);
    setStatus(null);
    try {
      const res = await registerPerson(firstName.trim(), lastName.trim(), file);
      setStatus({
        type: "success",
        message: `Registered "${res.data.name}" successfully into the database.`,
      });
      setFirstName("");
      setLastName("");
      setFile(null);
      setPreview(null);
    } catch (err) {
      setStatus({
        type: "error",
        message: err.response?.data?.detail || "Registration failed.",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="card register-card" onSubmit={onSubmit}>
      <div className="register-header">
        <h2>Register New Person</h2>
        <p className="subtitle">
          Add identity credentials and capture biometric facial data for the database.
        </p>
      </div>

      <div className="register-split-grid">
        {/* Left Side: Field Inputs & Submission */}
        <div className="register-left-col">
          <div className="fields-group">
            <label className="field">
              <span>
                First Name <em className="required">*</em>
              </span>
              <input
                type="text"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                placeholder="e.g. Alex"
                required
              />
            </label>

            <label className="field">
              <span>Last Name</span>
              <input
                type="text"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                placeholder="e.g. Morgan"
              />
            </label>
          </div>

          <div className="register-guidelines">
            <span className="guidelines-title">Enrollment Checklist:</span>
            <ul>
              <li>Face centered in frame</li>
              <li>Good frontal lighting (no harsh shadows)</li>
              <li>Remove dark sunglasses or face masks</li>
            </ul>
          </div>

          <button type="submit" disabled={loading} className="register-submit-btn">
            {loading && <span className="spinner" />}
            {loading ? "Registering Biometrics..." : "Register Person"}
          </button>

          {status && <StatusBanner type={status.type}>{status.message}</StatusBanner>}

          {status && status.type === "success" && onNavigateUsers && (
            <button
              type="button"
              className="view-history-link-btn"
              onClick={onNavigateUsers}
              style={{ marginTop: "4px" }}
            >
              👥 View in Registered Users Tab →
            </button>
          )}
        </div>

        {/* Right Side: Photo Capture */}
        <div className="register-right-col">
          <div className="field">
            <span>
              Face Photo Capture <em className="required">*</em>
            </span>
            <CameraCapture captured={preview} onCapture={onCapture} />
          </div>
        </div>
      </div>
    </form>
  );
}
