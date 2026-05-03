function buildPath(points, width, height, maxValue) {
  if (!points.length) return "";
  return points
    .map((point, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * width;
      const y = height - (point.value / maxValue) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

export default function EmotionHistoryChart({ history, accent }) {
  const width = 360;
  const height = 120;
  const points = history.slice(-60);
  const path = buildPath(points, width, height, 1);

  return (
    <div className="rounded-3xl border border-white/10 bg-black/20 p-5 backdrop-blur-xl">
      <div className="mb-4 flex items-center justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.25em] text-white/50">Confidence trend</p>
          <h3 className="text-lg font-semibold text-white">Last 30 seconds</h3>
        </div>
        <div className="text-right text-sm text-white/65">
          <p>{points.length ? points[points.length - 1].emotion : "Waiting..."}</p>
          <p>{points.length ? `${Math.round(points[points.length - 1].value * 100)}%` : "--"}</p>
        </div>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-32 w-full overflow-visible">
        <defs>
          <linearGradient id="emotionGradient" x1="0%" x2="100%">
            <stop offset="0%" stopColor={accent} stopOpacity="0.15" />
            <stop offset="100%" stopColor={accent} stopOpacity="0.95" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75, 1].map((tick) => (
          <line
            key={tick}
            x1="0"
            x2={width}
            y1={height - tick * height}
            y2={height - tick * height}
            stroke="rgba(255,255,255,0.08)"
            strokeDasharray="4 6"
          />
        ))}
        {path ? <path d={path} fill="none" stroke="url(#emotionGradient)" strokeWidth="4" strokeLinecap="round" /> : null}
      </svg>
    </div>
  );
}
