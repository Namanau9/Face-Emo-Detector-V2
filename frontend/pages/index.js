import { useEffect, useMemo, useRef, useState } from "react";

import EmotionAvatar from "../components/EmotionAvatar";
import EmotionHistoryChart from "../components/EmotionHistoryChart";
import PredictionPanel from "../components/PredictionPanel";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const HISTORY_LIMIT = 60;
const LOG_LIMIT = 300;

const EMOTION_THEME = {
  angry: { accent: "#ff6b6b", surface: "#40181d", glow: "rgba(255,107,107,0.45)" },
  disgust: { accent: "#9acd32", surface: "#213418", glow: "rgba(154,205,50,0.4)" },
  fear: { accent: "#7c83fd", surface: "#191d44", glow: "rgba(124,131,253,0.42)" },
  happy: { accent: "#ffd166", surface: "#463714", glow: "rgba(255,209,102,0.42)" },
  neutral: { accent: "#7bdff2", surface: "#17343f", glow: "rgba(123,223,242,0.38)" },
  sad: { accent: "#70a1ff", surface: "#172845", glow: "rgba(112,161,255,0.42)" },
  surprise: { accent: "#f78fb3", surface: "#45203a", glow: "rgba(247,143,179,0.42)" },
  uncertain: { accent: "#c8d6e5", surface: "#2d3641", glow: "rgba(200,214,229,0.35)" },
  no_face: { accent: "#94a3b8", surface: "#26313f", glow: "rgba(148,163,184,0.28)" }
};

const INITIAL_STATE = {
  emotion: "no_face",
  confidence: 0,
  face_detected: false,
  face_count: 0,
  message: "Camera is starting.",
  box: null
};

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captureLoop = useRef(null);
  const requestInFlightRef = useRef(false);
  const smoothingWindowRef = useRef([]);
  const logsRef = useRef([]);

  const [prediction, setPrediction] = useState(INITIAL_STATE);
  const [history, setHistory] = useState([]);
  const [logs, setLogs] = useState([]);
  const [snapshotUrl, setSnapshotUrl] = useState("");
  const [darkMode, setDarkMode] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [apiStatus, setApiStatus] = useState("checking");
  const [modelInfo, setModelInfo] = useState({ architecture: "loading", image_size: "--" });

  const theme = useMemo(() => EMOTION_THEME[prediction.emotion] || EMOTION_THEME.neutral, [prediction.emotion]);

  useEffect(() => {
    let mounted = true;
    fetch(`${API_BASE}/health`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("Health check failed");
        }
        return response.json();
      })
      .then((payload) => {
        if (!mounted) return;
        setApiStatus("online");
        setModelInfo({
          architecture: payload.architecture || "unknown",
          image_size: payload.image_size || "--"
        });
      })
      .catch(() => mounted && setApiStatus("offline"));
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false
      });
      if (!videoRef.current) return;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setIsStreaming(true);
      startCaptureLoop();
    } catch (error) {
      setPrediction({
        ...INITIAL_STATE,
        message: "Camera permission denied or unavailable."
      });
    }
  }

  function stopCamera() {
    if (captureLoop.current) {
      clearInterval(captureLoop.current);
      captureLoop.current = null;
    }
    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    setIsStreaming(false);
  }

  function startCaptureLoop() {
    if (captureLoop.current) clearInterval(captureLoop.current);
    captureLoop.current = setInterval(captureFrame, 500);
  }

  function smoothPrediction(nextState) {
    const window = smoothingWindowRef.current;
    window.push(nextState);
    if (window.length > 5) window.shift();

    const emotionScoreMap = {};
    window.forEach((entry) => {
      emotionScoreMap[entry.emotion] = (emotionScoreMap[entry.emotion] || 0) + entry.confidence;
    });

    const smoothedEmotion = Object.entries(emotionScoreMap).sort((a, b) => b[1] - a[1])[0]?.[0] || nextState.emotion;
    const filtered = window.filter((entry) => entry.emotion === smoothedEmotion);
    const avgConfidence =
      filtered.reduce((total, entry) => total + entry.confidence, 0) / Math.max(filtered.length, 1);
    const latest = filtered[filtered.length - 1] || nextState;

    return {
      ...latest,
      emotion: smoothedEmotion,
      confidence: Number(avgConfidence.toFixed(4))
    };
  }

  async function captureFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || requestInFlightRef.current) return;

    const size = 256;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, size, size);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
    if (!blob) return;

    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      requestInFlightRef.current = true;
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        throw new Error("Prediction request failed");
      }
      const raw = await response.json();
      const nextState = smoothPrediction(raw);

      setPrediction(nextState);
      setHistory((current) => {
        const next = [...current, { value: nextState.confidence, emotion: nextState.emotion, ts: Date.now() }];
        return next.slice(-HISTORY_LIMIT);
      });

      const logEntry = {
        timestamp: new Date().toISOString(),
        emotion: nextState.emotion,
        confidence: nextState.confidence,
        face_detected: nextState.face_detected,
        face_count: nextState.face_count
      };
      logsRef.current = [...logsRef.current, logEntry].slice(-LOG_LIMIT);
      setLogs(logsRef.current);
    } catch (error) {
      setPrediction({
        ...INITIAL_STATE,
        message: "Backend unavailable. Check the API service."
      });
    } finally {
      requestInFlightRef.current = false;
    }
  }

  function takeScreenshot() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    setSnapshotUrl(canvas.toDataURL("image/png"));
  }

  function downloadLogs() {
    if (!logsRef.current.length) return;
    const header = ["timestamp", "emotion", "confidence", "face_detected", "face_count"];
    const rows = logsRef.current.map((entry) => header.map((key) => entry[key]).join(","));
    const csv = [header.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "emotion_logs.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  }

  return (
    <main
      className={`min-h-screen transition-colors duration-700 ${darkMode ? "text-white" : "text-slate-900"}`}
      style={{
        background: darkMode
          ? `radial-gradient(circle at top left, ${theme.glow}, transparent 35%), linear-gradient(135deg, #07111f 0%, ${theme.surface} 58%, #05070b 100%)`
          : `radial-gradient(circle at top left, ${theme.glow}, transparent 35%), linear-gradient(135deg, #edf2f7 0%, #ffffff 45%, #dfe7f3 100%)`
      }}
    >
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-6 py-8 lg:px-10">
        <header className="flex flex-col gap-4 rounded-[2rem] border border-white/10 bg-white/8 p-6 backdrop-blur-xl lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.35em] text-white/55">Emotion studio</p>
            <h1 className="mt-2 text-4xl font-semibold tracking-tight">Real-time face emotion recognition</h1>
            <p className="mt-3 max-w-2xl text-sm text-white/70">
              EfficientNetB0 + attention on the backend, a low-latency webcam loop on the frontend, and live visual feedback that reacts with your expression.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => setDarkMode((value) => !value)}
              className="rounded-full border border-white/15 px-4 py-2 text-sm transition hover:bg-white/10"
            >
              {darkMode ? "Light mode" : "Dark mode"}
            </button>
            <button
              onClick={takeScreenshot}
              className="rounded-full border border-white/15 px-4 py-2 text-sm transition hover:bg-white/10"
            >
              Screenshot
            </button>
            <button
              onClick={downloadLogs}
              className="rounded-full border border-white/15 px-4 py-2 text-sm transition hover:bg-white/10"
            >
              Download CSV
            </button>
          </div>
        </header>

        <section className="mt-8 grid flex-1 gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="space-y-6">
            <div className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-black/30 p-4 shadow-aura">
              <div
                className="pointer-events-none absolute inset-0 opacity-50 blur-3xl transition duration-500"
                style={{ background: `radial-gradient(circle at 20% 20%, ${theme.glow}, transparent 40%)` }}
              />
              <video ref={videoRef} className="relative aspect-video w-full rounded-[1.5rem] object-cover" muted playsInline />
              <canvas ref={canvasRef} className="hidden" />
              <div className="relative mt-4 flex flex-wrap items-center justify-between gap-3 text-sm text-white/75">
                <span>{isStreaming ? "Camera live" : "Camera offline"}</span>
                <span>API: {apiStatus}</span>
                <span>Model: {modelInfo.architecture}</span>
                <span>Input: {modelInfo.image_size}px</span>
                <span>Sampling every 500ms</span>
              </div>
            </div>

            <PredictionPanel state={prediction} accent={theme.accent} />
            <EmotionHistoryChart history={history} accent={theme.accent} />
          </div>

          <div className="space-y-6">
            <div className="flex items-center justify-center rounded-[2rem] border border-white/10 bg-white/8 p-8 backdrop-blur-xl">
              <EmotionAvatar emotion={prediction.emotion} accent={theme.accent} />
            </div>

            <div className="rounded-[2rem] border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
              <p className="text-sm uppercase tracking-[0.25em] text-white/50">Recent log</p>
              <div className="mt-4 space-y-3">
                {logs.slice(-6).reverse().map((entry) => (
                  <div key={`${entry.timestamp}-${entry.emotion}`} className="rounded-2xl border border-white/10 bg-black/15 px-4 py-3 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="capitalize text-white">{entry.emotion.replace("_", " ")}</span>
                      <span className="text-white/60">{Math.round(entry.confidence * 100)}%</span>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-white/50">
                      <span>{new Date(entry.timestamp).toLocaleTimeString()}</span>
                      <span>{entry.face_count} face(s)</span>
                    </div>
                  </div>
                ))}
                {!logs.length ? <p className="text-sm text-white/55">Predictions will appear here after the first captured frame.</p> : null}
              </div>
            </div>

            {snapshotUrl ? (
              <div className="rounded-[2rem] border border-white/10 bg-white/8 p-6 backdrop-blur-xl">
                <p className="text-sm uppercase tracking-[0.25em] text-white/50">Latest screenshot</p>
                <img src={snapshotUrl} alt="Latest camera frame" className="mt-4 w-full rounded-[1.5rem]" />
              </div>
            ) : null}
          </div>
        </section>
      </div>
    </main>
  );
}
