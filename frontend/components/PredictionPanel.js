export default function PredictionPanel({ state, accent }) {
  const confidence = Math.round((state.confidence || 0) * 100);

  return (
    <div className="rounded-3xl border border-white/10 bg-white/8 p-6 backdrop-blur-xl animate-slideUp">
      <p className="text-sm uppercase tracking-[0.25em] text-white/50">Live prediction</p>
      <div className="mt-4 flex items-end justify-between gap-4">
        <div>
          <h2 className="text-4xl font-semibold capitalize text-white">{state.emotion.replace("_", " ")}</h2>
          <p className="mt-2 text-sm text-white/70">{state.message || "Tracking the strongest visible face in frame."}</p>
        </div>
        <div
          className="rounded-2xl px-4 py-3 text-right text-white shadow-aura"
          style={{ backgroundColor: `${accent}33`, border: `1px solid ${accent}66` }}
        >
          <p className="text-xs uppercase tracking-[0.2em] text-white/60">Confidence</p>
          <p className="text-2xl font-semibold">{confidence}%</p>
        </div>
      </div>
      <div className="mt-5 h-3 overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${confidence}%`, background: `linear-gradient(90deg, ${accent}, #ffffff)` }}
        />
      </div>
      <div className="mt-5 flex items-center justify-between text-sm text-white/70">
        <span>{state.face_detected ? "Face detected" : "Waiting for face"}</span>
        <span>{state.face_count || 0} face(s)</span>
      </div>
    </div>
  );
}
