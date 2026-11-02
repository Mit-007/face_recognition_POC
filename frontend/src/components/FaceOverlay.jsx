export default function FaceOverlay({
  bbox,
  kps = [],
  imageDimensions,
  name,
  score,
  matched,
  latencyMs,
}) {
  if (!bbox || !imageDimensions) return null;

  const [x1, y1, x2, y2] = bbox;
  const { width, height } = imageDimensions;

  const boxW = Math.max(0, x2 - x1);
  const boxH = Math.max(0, y2 - y1);
  const corner = Math.min(boxW, boxH) * 0.22;

  const color = matched ? "#10b981" : "#ef4444";
  const fillColor = matched ? "rgba(16, 185, 129, 0.12)" : "rgba(239, 68, 68, 0.14)";
  const textColor = "#ffffff";

  // Label text
  let labelText = "Face Detected";
  if (matched && name) {
    labelText = score !== undefined ? `${name} · ${(score * 100).toFixed(1)}%` : name;
  } else if (!matched && score !== undefined && score > 0) {
    labelText = `Unknown · ${(score * 100).toFixed(1)}%`;
  } else if (!matched) {
    labelText = "Unrecognized";
  }

  // Label badge placement
  const labelY = y1 > 32 ? y1 - 26 : y1 + 6;
  const labelX = Math.max(8, Math.min(x1, width - 180));
  const badgeWidth = Math.min(220, Math.max(120, labelText.length * 8.5 + 24));

  return (
    <svg
      className="face-overlay"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 4,
      }}
    >
      <defs>
        <filter id="glow-filter" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Semi-transparent bounding box body */}
      <rect
        x={x1}
        y={y1}
        width={boxW}
        height={boxH}
        fill={fillColor}
        stroke={color}
        strokeWidth="1.5"
        strokeDasharray="4 3"
        rx="4"
      />

      {/* Corner Brackets */}
      <g stroke={color} strokeWidth="3.5" fill="none" strokeLinecap="round">
        {/* Top-Left */}
        <path d={`M ${x1} ${y1 + corner} L ${x1} ${y1} L ${x1 + corner} ${y1}`} />
        {/* Top-Right */}
        <path d={`M ${x2 - corner} ${y1} L ${x2} ${y1} L ${x2} ${y1 + corner}`} />
        {/* Bottom-Left */}
        <path d={`M ${x1} ${y2 - corner} L ${x1} ${y2} L ${x1 + corner} ${y2}`} />
        {/* Bottom-Right */}
        <path d={`M ${x2 - corner} ${y2} L ${x2} ${y2} L ${x2} ${y2 - corner}`} />
      </g>

      {/* Floating Identity Badge */}
      <g filter="url(#glow-filter)">
        <rect
          x={labelX}
          y={labelY}
          width={badgeWidth}
          height="22"
          rx="5"
          fill={color}
          opacity="0.95"
        />
        {/* Status dot */}
        <circle cx={labelX + 11} cy={labelY + 11} r="3.5" fill="#ffffff" />
        <text
          x={labelX + 22}
          y={labelY + 15}
          fill={textColor}
          fontSize="11.5"
          fontWeight="700"
          fontFamily="system-ui, -apple-system, sans-serif"
          letterSpacing="0.2px"
        >
          {labelText}
        </text>
      </g>

      {/* Facial Keypoints (Landmarks) */}
      {kps && kps.length >= 2 && (
        <g opacity="0.9">
          {/* Subtle line between eyes */}
          <line
            x1={kps[0][0]}
            y1={kps[0][1]}
            x2={kps[1][0]}
            y2={kps[1][1]}
            stroke={color}
            strokeWidth="1"
            strokeDasharray="2 2"
          />
          {kps.map((pt, idx) => (
            <g key={idx}>
              <circle cx={pt[0]} cy={pt[1]} r="5" fill="none" stroke={color} strokeWidth="1.2" />
              <circle cx={pt[0]} cy={pt[1]} r="2.2" fill="#ffffff" />
            </g>
          ))}
        </g>
      )}

      {/* Latency badge bottom-right */}
      {latencyMs && (
        <g>
          <rect
            x={width - 70}
            y={height - 24}
            width="62"
            height="18"
            rx="4"
            fill="rgba(0,0,0,0.65)"
          />
          <text
            x={width - 39}
            y={height - 11}
            fill="#e2e8f0"
            fontSize="10"
            fontFamily="monospace"
            textAnchor="middle"
          >
            {latencyMs}ms
          </text>
        </g>
      )}
    </svg>
  );
}
