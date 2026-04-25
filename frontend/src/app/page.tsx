"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import styles from "./page.module.css";

// ─── Types ───────────────────────────────────────────────
interface DetectorScores {
  spatial: number;
  frequency: number;
  lipsync?: number;
}

interface FrameResult {
  frame_index: number;
  spatial_score: number;
  frequency_score: number;
  ensemble: {
    final_score: number;
    verdict: string;
    confidence: number;
    confidence_label: string;
    per_detector: DetectorScores;
    weights_used: Record<string, number>;
  };
}

interface FrameInfo {
  url: string;
  heatmap_url?: string | null;
  index: number;
}

interface AnalysisResult {
  type: "image" | "video";
  overall: {
    final_score: number;
    verdict: string;
    confidence: number;
    confidence_label: string;
    per_detector: Record<string, any>;
    frame_count?: number;
    frame_scores?: number[];
    temporal_consistency?: number;
  };
  lipsync?: { score: number; correlation: number | null; analysis_type: string };
  frame_results?: FrameResult[];
  frames?: FrameInfo[];
  metadata?: Record<string, any>;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ─── Main Page ───────────────────────────────────────────
export default function Home() {
  // State
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressStatus, setProgressStatus] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  // Frame player state
  const [currentFrame, setCurrentFrame] = useState(0);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ─── File Handling ────────────────────────────────────
  const handleFileSelect = useCallback((selectedFile: File) => {
    const ext = "." + selectedFile.name.split(".").pop()?.toLowerCase();
    const allowed = new Set([".mp4",".avi",".mov",".mkv",".webm",".jpg",".jpeg",".png",".bmp"]);
    if (!allowed.has(ext)) {
      setError(`Unsupported format: ${ext}`);
      return;
    }
    setFile(selectedFile);
    setError(null);
    setResult(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    },
    [handleFileSelect]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setDragging(false), []);

  // ─── Analysis ─────────────────────────────────────────
  const startAnalysis = useCallback(async () => {
    if (!file) return;
    setAnalyzing(true);
    setProgress(0);
    setError(null);
    setResult(null);
    setProgressStatus("Uploading file...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Submit for analysis
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const { job_id } = await res.json();
      setProgress(5);
      setProgressStatus("Processing with AI models...");

      // Poll for results
      let attempts = 0;
      const maxAttempts = 600;

      while (attempts < maxAttempts) {
        await new Promise((r) => setTimeout(r, 1000));
        attempts++;

        const pollRes = await fetch(`${API_BASE}/api/results/${job_id}`);
        const pollData = await pollRes.json();

        setProgress(Math.round(pollData.progress * 100));

        if (pollData.progress < 0.2) {
          setProgressStatus("Extracting frames and detecting faces...");
        } else if (pollData.progress < 0.5) {
          setProgressStatus("Running spatial artifact detection (XceptionNet)...");
        } else if (pollData.progress < 0.7) {
          setProgressStatus("Analyzing frequency domain (FFT)...");
        } else if (pollData.progress < 0.85) {
          setProgressStatus("Checking lip-sync consistency...");
        } else if (pollData.progress < 0.95) {
          setProgressStatus("Computing ensemble scores...");
        } else {
          setProgressStatus("Generating heatmaps and finalizing...");
        }

        if (pollData.status === "completed") {
          setResult(pollData.result);
          setCurrentFrame(0);
          break;
        } else if (pollData.status === "failed") {
          throw new Error(pollData.error || "Analysis failed");
        }
      }
    } catch (err: any) {
      setError(err.message || "An error occurred");
    } finally {
      setAnalyzing(false);
    }
  }, [file]);

  // ─── Frame Player Controls ────────────────────────────
  const totalFrames = result?.frames?.length || 0;

  const nextFrame = useCallback(() => {
    setCurrentFrame((prev) => Math.min(prev + 1, totalFrames - 1));
  }, [totalFrames]);

  const prevFrame = useCallback(() => {
    setCurrentFrame((prev) => Math.max(prev - 1, 0));
  }, []);

  const togglePlay = useCallback(() => {
    if (isPlaying) {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      playIntervalRef.current = setInterval(() => {
        setCurrentFrame((prev) => {
          if (prev >= totalFrames - 1) {
            if (playIntervalRef.current) clearInterval(playIntervalRef.current);
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 200);
    }
  }, [isPlaying, totalFrames]);

  useEffect(() => {
    return () => {
      if (playIntervalRef.current) clearInterval(playIntervalRef.current);
    };
  }, []);

  // ─── Reset ────────────────────────────────────────────
  const resetAnalysis = useCallback(() => {
    setFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
    setCurrentFrame(0);
    setIsPlaying(false);
    if (playIntervalRef.current) clearInterval(playIntervalRef.current);
  }, []);

  // ─── Computed Values ──────────────────────────────────
  const currentFrameData = result?.frames?.[currentFrame];
  const currentFrameResult = result?.frame_results?.[currentFrame];
  const isFake = result?.overall?.verdict === "FAKE";
  const verdictClass = isFake ? styles.fake : styles.real;

  // Gauge calculation
  const score = result?.overall?.final_score || 0;
  const circumference = 2 * Math.PI * 45;
  const dashOffset = circumference - score * circumference;

  return (
    <div className={styles.page}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.logo}>
          <div className={styles.logoIcon}>🔬</div>
          <span className={styles.logoText}>DeepScan AI</span>
        </div>
        <p className={styles.subtitle}>
          Multi-model ensemble deepfake detection powered by spatial analysis,
          frequency forensics, and lip-sync verification
        </p>
      </header>

      {/* Main Content */}
      <main className={styles.main}>
        <div className={`${styles.contentGrid} ${result ? styles.hasResults : ""}`}>
          {/* Left Column */}
          <div>
            {/* Upload Zone */}
            {!analyzing && !result && (
              <div className={styles.uploadContainer}>
                <div
                  className={`${styles.uploadZone} glass-card ${
                    dragging ? styles.uploadZoneDragging : ""
                  }`}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <span className={styles.uploadIcon}>🎯</span>
                  <h2 className={styles.uploadTitle}>
                    Drop your media here or click to browse
                  </h2>
                  <p className={styles.uploadSubtitle}>
                    Support for images and videos up to 500MB
                  </p>
                  <div className={styles.uploadFormats}>
                    {["MP4", "AVI", "MOV", "WEBM", "JPG", "PNG"].map((fmt) => (
                      <span key={fmt} className={styles.formatBadge}>{fmt}</span>
                    ))}
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    className={styles.fileInput}
                    accept="video/*,image/*"
                    onChange={(e) => {
                      if (e.target.files?.[0]) handleFileSelect(e.target.files[0]);
                    }}
                  />
                </div>

                {file && (
                  <>
                    <div className={styles.filePreview}>
                      <span className={styles.fileIcon}>
                        {file.type.startsWith("video") ? "🎬" : "🖼️"}
                      </span>
                      <div className={styles.fileInfo}>
                        <div className={styles.fileName}>{file.name}</div>
                        <div className={styles.fileSize}>
                          {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </div>
                      </div>
                      <button className={styles.removeBtn} onClick={(e) => { e.stopPropagation(); setFile(null); }}>
                        ✕ Remove
                      </button>
                    </div>

                    <button
                      className={styles.analyzeBtn}
                      onClick={startAnalysis}
                      disabled={analyzing}
                    >
                      🔍 Analyze for Deepfakes
                    </button>
                  </>
                )}

                {error && (
                  <div style={{
                    marginTop: "1rem",
                    padding: "1rem",
                    background: "rgba(239, 68, 68, 0.1)",
                    border: "1px solid rgba(239, 68, 68, 0.3)",
                    borderRadius: "var(--radius-md)",
                    color: "var(--accent-red)",
                    fontSize: "0.9rem",
                  }}>
                    ⚠️ {error}
                  </div>
                )}
              </div>
            )}

            {/* Progress */}
            {analyzing && (
              <div className={styles.progressContainer}>
                <div className={`${styles.progressCard} glass-card`}>
                  <div className={styles.progressHeader}>
                    <span className={styles.progressTitle}>
                      <span className={styles.spinner}></span>
                      Analyzing...
                    </span>
                    <span className={styles.progressPercent}>{progress}%</span>
                  </div>
                  <div className={styles.progressBarOuter}>
                    <div
                      className={styles.progressBarInner}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className={styles.progressStatus}>{progressStatus}</p>
                </div>
              </div>
            )}

            {/* Frame Player */}
            {result && result.frames && result.frames.length > 0 && (
              <div className={styles.playerCard + " glass-card"}>
                <h3 className={styles.sectionTitle}>
                  <span className={styles.icon}>🎞️</span>
                  Frame-by-Frame Analysis
                </h3>

                <div className={styles.playerViewport}>
                  {currentFrameData && (
                    <>
                      <img
                        className={styles.playerImage}
                        src={`${API_BASE}${currentFrameData.url}`}
                        alt={`Frame ${currentFrame}`}
                      />
                      {showHeatmap && currentFrameData.heatmap_url && (
                        <img
                          className={styles.heatmapOverlay}
                          src={`${API_BASE}${currentFrameData.heatmap_url}`}
                          alt="Heatmap overlay"
                        />
                      )}
                    </>
                  )}
                </div>

                <div className={styles.playerControls}>
                  <button className={styles.playerBtn} onClick={prevFrame} disabled={currentFrame === 0}>
                    ⏮
                  </button>
                  <button className={styles.playerBtn} onClick={togglePlay}>
                    {isPlaying ? "⏸" : "▶"}
                  </button>
                  <button
                    className={styles.playerBtn}
                    onClick={nextFrame}
                    disabled={currentFrame >= totalFrames - 1}
                  >
                    ⏭
                  </button>

                  <input
                    type="range"
                    className={styles.frameSlider}
                    min={0}
                    max={Math.max(0, totalFrames - 1)}
                    value={currentFrame}
                    onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
                  />

                  <span className={styles.frameInfo}>
                    {currentFrame + 1}/{totalFrames}
                  </span>
                </div>

                <div className={styles.heatmapToggle}>
                  <button
                    className={`${styles.toggle} ${showHeatmap ? styles.active : ""}`}
                    onClick={() => setShowHeatmap(!showHeatmap)}
                  >
                    <span className={styles.toggleKnob}></span>
                  </button>
                  <span className={styles.toggleLabel}>Show Heatmap Overlay</span>
                </div>

                {/* Per-frame score display */}
                {currentFrameResult && (
                  <div style={{ marginTop: "1rem", padding: "0.75rem", background: "var(--bg-glass)", borderRadius: "var(--radius-sm)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem" }}>
                      <span style={{ color: "var(--text-secondary)" }}>Frame Score:</span>
                      <span style={{
                        fontFamily: "var(--font-mono)",
                        fontWeight: 600,
                        color: currentFrameResult.ensemble.final_score >= 0.5 ? "var(--accent-red)" : "var(--accent-green)"
                      }}>
                        {(currentFrameResult.ensemble.final_score * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Timeline */}
            {result && result.overall?.frame_scores && result.overall.frame_scores.length > 1 && (
              <div className={`${styles.timelineCard} glass-card`}>
                <h3 className={styles.sectionTitle}>
                  <span className={styles.icon}>📊</span>
                  Score Timeline
                </h3>
                <div className={styles.timeline}>
                  {result.overall.frame_scores.map((score, i) => (
                    <div
                      key={i}
                      className={`${styles.timelineBar} ${currentFrame === i ? styles.active : ""}`}
                      style={{
                        height: `${Math.max(4, score * 100)}%`,
                        backgroundColor: score >= 0.5
                          ? `rgba(239, 68, 68, ${0.4 + score * 0.6})`
                          : `rgba(16, 185, 129, ${0.4 + (1 - score) * 0.6})`,
                        color: score >= 0.5 ? "var(--accent-red)" : "var(--accent-green)",
                      }}
                      onClick={() => setCurrentFrame(i)}
                    />
                  ))}
                </div>
                <div className={styles.timelineLabels}>
                  <span className={styles.timelineLabel}>Frame 1</span>
                  <span className={styles.timelineLabel}>
                    Frame {result.overall.frame_scores.length}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results Dashboard */}
          {result && (
            <div className={styles.resultsContainer}>
              {/* Verdict Card */}
              <div className={`${styles.verdictCard} glass-card ${verdictClass}`}>
                {/* Confidence Gauge */}
                <div className={styles.gaugeContainer}>
                  <svg className={styles.gauge} width="130" height="130" viewBox="0 0 100 100">
                    <circle className={styles.gaugeBackground} cx="50" cy="50" r="45" />
                    <circle
                      className={`${styles.gaugeProgress} ${verdictClass}`}
                      cx="50"
                      cy="50"
                      r="45"
                      strokeDasharray={circumference}
                      strokeDashoffset={dashOffset}
                    />
                  </svg>
                  <div className={styles.gaugeCenter}>
                    <div className={`${styles.gaugeScore} ${verdictClass}`}>
                      {(score * 100).toFixed(0)}
                    </div>
                    <div className={styles.gaugeLabel}>Score</div>
                  </div>
                </div>

                <div className={`${styles.verdictBadge} ${verdictClass}`}>
                  {isFake ? "⚠️" : "✅"} {result.overall.verdict}
                </div>

                <span className={styles.confidenceLabel}>
                  {result.overall.confidence_label} •
                  Confidence: {(result.overall.confidence * 100).toFixed(0)}%
                </span>
              </div>

              {/* Detector Breakdown */}
              <div className={`${styles.breakdownCard} glass-card`}>
                <h3 className={styles.sectionTitle}>
                  <span className={styles.icon}>🧠</span>
                  Detector Breakdown
                </h3>

                {/* Spatial */}
                <DetectorRow
                  icon="🔍"
                  iconClass={styles.spatial}
                  name="Spatial Analysis"
                  subtitle="XceptionNet CNN"
                  score={getDetectorScore(result.overall.per_detector, "spatial")}
                  weight={0.45}
                />

                {/* Frequency */}
                <DetectorRow
                  icon="〰️"
                  iconClass={styles.frequency}
                  name="Frequency Domain"
                  subtitle="FFT Analysis"
                  score={getDetectorScore(result.overall.per_detector, "frequency")}
                  weight={0.30}
                />

                {/* Lip-sync */}
                {result.lipsync && (
                  <DetectorRow
                    icon="🗣️"
                    iconClass={styles.lipsync}
                    name="Lip-Sync Check"
                    subtitle="Audio-Visual"
                    score={result.lipsync.score}
                    weight={0.25}
                  />
                )}
              </div>

              {/* Metadata */}
              {result.metadata && (
                <div className={`${styles.breakdownCard} glass-card`}>
                  <h3 className={styles.sectionTitle}>
                    <span className={styles.icon}>ℹ️</span>
                    Media Info
                  </h3>
                  <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                    {result.metadata.duration && (
                      <div style={{ display: "flex", justifyContent: "space-between", padding: "0.4rem 0" }}>
                        <span>Duration</span>
                        <span style={{ fontFamily: "var(--font-mono)" }}>
                          {result.metadata.duration.toFixed(1)}s
                        </span>
                      </div>
                    )}
                    {result.metadata.fps && (
                      <div style={{ display: "flex", justifyContent: "space-between", padding: "0.4rem 0" }}>
                        <span>FPS</span>
                        <span style={{ fontFamily: "var(--font-mono)" }}>
                          {result.metadata.fps.toFixed(0)}
                        </span>
                      </div>
                    )}
                    {result.metadata.width && (
                      <div style={{ display: "flex", justifyContent: "space-between", padding: "0.4rem 0" }}>
                        <span>Resolution</span>
                        <span style={{ fontFamily: "var(--font-mono)" }}>
                          {result.metadata.width}×{result.metadata.height}
                        </span>
                      </div>
                    )}
                    {result.overall.frame_count && (
                      <div style={{ display: "flex", justifyContent: "space-between", padding: "0.4rem 0" }}>
                        <span>Frames Analyzed</span>
                        <span style={{ fontFamily: "var(--font-mono)" }}>
                          {result.overall.frame_count}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <button className={styles.newAnalysisBtn} onClick={resetAnalysis}>
                🔄 New Analysis
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

// ─── Helper Components ───────────────────────────────────
function DetectorRow({
  icon,
  iconClass,
  name,
  subtitle,
  score,
  weight,
}: {
  icon: string;
  iconClass: string;
  name: string;
  subtitle: string;
  score: number;
  weight: number;
}) {
  const barColor =
    score >= 0.7
      ? "var(--accent-red)"
      : score >= 0.4
      ? "var(--accent-orange)"
      : "var(--accent-green)";

  return (
    <div className={styles.detectorRow}>
      <div className={styles.detectorInfo}>
        <div className={`${styles.detectorIcon} ${iconClass}`}>{icon}</div>
        <div>
          <div className={styles.detectorName}>{name}</div>
          <div className={styles.detectorWeight}>
            {subtitle} • w={weight}
          </div>
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center" }}>
        <span className={styles.detectorScore} style={{ color: barColor }}>
          {(score * 100).toFixed(1)}%
        </span>
        <div className={styles.detectorBarOuter}>
          <div
            className={styles.detectorBarInner}
            style={{ width: `${score * 100}%`, backgroundColor: barColor }}
          />
        </div>
      </div>
    </div>
  );
}

function getDetectorScore(perDetector: Record<string, any>, key: string): number {
  if (!perDetector) return 0;
  const val = perDetector[key];
  if (typeof val === "number") return val;
  if (val && typeof val === "object" && "mean" in val) return val.mean;
  return 0;
}
