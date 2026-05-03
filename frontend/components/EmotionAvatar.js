const AVATARS = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😄",
  neutral: "🙂",
  sad: "😢",
  surprise: "😲",
  uncertain: "🫤",
  no_face: "🫥"
};

export default function EmotionAvatar({ emotion, accent }) {
  return (
    <div
      className="relative flex h-36 w-36 items-center justify-center rounded-full border border-white/20 text-7xl shadow-aura transition duration-500 animate-float"
      style={{ background: `radial-gradient(circle at 30% 30%, ${accent}55, transparent 65%)` }}
    >
      <div
        className="absolute inset-0 rounded-full blur-2xl opacity-80 animate-pulseGlow"
        style={{ background: `${accent}33` }}
      />
      <span className="relative">{AVATARS[emotion] || "🙂"}</span>
    </div>
  );
}
