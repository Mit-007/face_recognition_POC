import { useEffect, useRef, useState } from "react";

export default function CameraCapture({ captured, onCapture }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const [active, setActive] = useState(false);
  const [error, setError] = useState(null);

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setActive(false);
  };

  useEffect(() => stopCamera, []);

  // The <video> element only mounts once `active` is true, so the ref isn't
  // available yet inside startCamera() right after setActive(true). Attach
  // the stream here instead, once the element actually exists.
  useEffect(() => {
    if (active && videoRef.current && streamRef.current) {
      const video = videoRef.current;
      video.srcObject = streamRef.current;
      video.play().catch(() => {});
    }
  }, [active]);

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
      });
      streamRef.current = stream;
      setActive(true);
    } catch (err) {
      setError("Could not access camera. Please allow camera permission and try again.");
    }
  };

  const capture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    if (!video.videoWidth || !video.videoHeight) {
      setError("Camera feed isn't ready yet. Wait a moment and try again.");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    // Mirror the capture to match the mirrored live preview the user saw.
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" });
        onCapture(file, canvas.toDataURL("image/jpeg"));
        stopCamera();
      },
      "image/jpeg",
      0.92
    );
  };

  const retake = () => {
    onCapture(null, null);
    startCamera();
  };

  return (
    <div className="camera-box">
      {!captured && !active && (
        <button type="button" className="camera-btn" onClick={startCamera}>
          Open Camera
        </button>
      )}

      {!captured && active && (
        <div className="camera-live">
          <div className="camera-frame">
            <video ref={videoRef} autoPlay playsInline muted />
            <div className="camera-guide" />
          </div>
          <p className="camera-hint">Center your face in the outline, then capture</p>
          <button type="button" className="camera-btn" onClick={capture}>
            Capture Photo
          </button>
        </div>
      )}

      {captured && (
        <div className="camera-preview">
          <img src={captured} alt="captured" />
          <button type="button" className="camera-btn secondary" onClick={retake}>
            Retake
          </button>
        </div>
      )}

      <canvas ref={canvasRef} style={{ display: "none" }} />

      {error && <p className="status error">{error}</p>}
    </div>
  );
}
