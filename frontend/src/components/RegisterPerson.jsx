import { useState } from "react";
import { registerPerson } from "../api";
import CameraCapture from "./CameraCapture";
import StatusBanner from "./StatusBanner";

export default function RegisterPerson() {
  const [name, setName] = useState("");
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
    if (!name || !file) {
      setStatus({ type: "error", message: "Please enter a name and capture a photo." });
      return;
    }

    setLoading(true);
    setStatus(null);
    try {
      const res = await registerPerson(name, file);
      setStatus({
        type: "success",
        message: `Registered "${res.data.name}" successfully.`,
      });
      setName("");
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
    <form className="card" onSubmit={onSubmit}>
      <h2>Register New Person</h2>
      <p className="subtitle">Add someone to the database with their name and a live camera photo.</p>

      <label className="field">
        <span>Name</span>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. John Doe"
        />
      </label>

      <div className="field">
        <span>Face Photo</span>
        <CameraCapture captured={preview} onCapture={onCapture} />
      </div>

      <button type="submit" disabled={loading}>
        {loading && <span className="spinner" />}
        {loading ? "Registering..." : "Register Person"}
      </button>

      {status && <StatusBanner type={status.type}>{status.message}</StatusBanner>}
    </form>
  );
}
