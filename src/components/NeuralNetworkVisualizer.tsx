import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import {
  ConnectionColorMode,
  NetworkConnection,
  NetworkNode,
  NetworkPrediction,
  Step5Result,
} from '../types';
import {
  extractInputFeatures,
  generateNetworkGraphData,
  HIDDEN_DESCRIPTIONS,
  OUTPUT_DIGIT_COLORS,
  runForwardPass,
  SPECTRUM_COLORS,
} from '../utils/neuralNetworkEngine';
import { MultiDigitResult, RecognizedDigit } from '../utils/digitSegmentation';
import {
  Activity,
  Award,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Eye,
  Info,
  Layers,
  Palette,
  Sparkles,
  Volume2,
  VolumeX,
  Zap,
  Check,
  TrendingUp,
} from 'lucide-react';

interface NeuralNetworkVisualizerProps {
  step5Result?: Step5Result | null;
  hasContent?: boolean;
  multiDigitResult?: MultiDigitResult | null;
  selectedDigitIndex?: number;
  onSelectDigitIndex?: (index: number) => void;
}

export const NeuralNetworkVisualizer: React.FC<NeuralNetworkVisualizerProps> = ({
  step5Result,
  hasContent = false,
  multiDigitResult,
  selectedDigitIndex = 0,
  onSelectDigitIndex,
}) => {
  const [viewMode, setViewMode] = useState<'synaptic' | 'simplified'>('synaptic');
  const [colorMode, setColorMode] = useState<ConnectionColorMode>('spectrum');
  const [lineStyle, setLineStyle] = useState<'flowing' | 'solid'>('flowing');
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showOnlyActive, setShowOnlyActive] = useState<boolean>(false);
  const [showLayerDetails, setShowLayerDetails] = useState<boolean>(false);
  const [autoSpeak, setAutoSpeak] = useState<boolean>(true);

  // 1. Compute active step5 data: if multi-digit is present and index is selected, use that
  const activeStep5 = useMemo(() => {
    if (
      multiDigitResult &&
      multiDigitResult.hasContent &&
      multiDigitResult.digits[selectedDigitIndex]?.step5Result
    ) {
      return multiDigitResult.digits[selectedDigitIndex].step5Result;
    }
    return step5Result;
  }, [multiDigitResult, selectedDigitIndex, step5Result]);

  // 2. Compute input features (16 spatial sectors)
  const inputFeatures = useMemo(() => {
    if (!activeStep5 || !activeStep5.data) {
      return new Array(16).fill(0);
    }
    return extractInputFeatures(activeStep5.data, activeStep5.resolution);
  }, [activeStep5]);

  // 3. Compute forward pass through Neural Network
  const prediction: NetworkPrediction = useMemo(() => {
    return runForwardPass(inputFeatures);
  }, [inputFeatures]);

  // Active status
  const isActuallyPredicting =
    (hasContent && step5Result && step5Result.data) ||
    (multiDigitResult && multiDigitResult.hasContent);

  // 4. Text-to-Speech logic
  const lastSpokenRef = useRef<string>('');

  const speakText = useCallback((text: string) => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.92;
        utterance.pitch = 1.0;
        window.speechSynthesis.speak(utterance);
      } catch (err) {
        console.warn('Speech synthesis error:', err);
      }
    }
  }, []);

  const currentSpokenText = useMemo(() => {
    if (!isActuallyPredicting) return '';
    if (multiDigitResult && multiDigitResult.hasContent && multiDigitResult.digitCount > 1) {
      return `Recognized number ${multiDigitResult.digits.map((d) => d.digit).join(' ')}`;
    }
    return `Recognized digit ${prediction.predictedDigit}`;
  }, [isActuallyPredicting, multiDigitResult, prediction.predictedDigit]);

  // Auto-speak on new prediction if enabled
  useEffect(() => {
    if (autoSpeak && isActuallyPredicting && currentSpokenText) {
      if (currentSpokenText !== lastSpokenRef.current) {
        lastSpokenRef.current = currentSpokenText;
        const timer = setTimeout(() => {
          speakText(currentSpokenText);
        }, 150);
        return () => clearTimeout(timer);
      }
    }
  }, [autoSpeak, isActuallyPredicting, currentSpokenText, speakText]);

  // 5. Compute visual nodes and colored connection lines for graph
  const graphWidth = 740;
  const graphHeight = 440;

  const { nodes, connections } = useMemo(() => {
    return generateNetworkGraphData(
      inputFeatures,
      prediction,
      colorMode,
      selectedNodeId,
      graphWidth,
      graphHeight
    );
  }, [inputFeatures, prediction, colorMode, selectedNodeId]);

  // Filter connections if user toggled "Show Active Only"
  const visibleConnections = useMemo(() => {
    if (selectedNodeId) {
      return connections.filter(
        (c) => c.fromId === selectedNodeId || c.toId === selectedNodeId
      );
    }
    if (showOnlyActive) {
      return connections.filter(
        (c) => Math.abs(c.activation) > 0.15 || c.opacity > 0.55
      );
    }
    return connections;
  }, [connections, selectedNodeId, showOnlyActive]);

  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    return nodes.find((n) => n.id === selectedNodeId) || null;
  }, [selectedNodeId, nodes]);

  const topPredictedColor = OUTPUT_DIGIT_COLORS[prediction.predictedDigit];

  return (
    <div
      id="neural-network-visualizer"
      className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl space-y-5"
    >
      {/* Top Header Bar & Mode Toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </span>
            <span className="text-xs font-mono font-semibold uppercase tracking-wider text-indigo-400">
              Neural Network Classifier
            </span>
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono border border-slate-700">
              16 In &rarr; 14 Hidden &rarr; 10 Output
            </span>
            {multiDigitResult && multiDigitResult.hasContent && multiDigitResult.digitCount > 1 && (
              <span className="text-[11px] px-2 py-0.5 rounded-full bg-amber-950/80 text-amber-300 font-mono border border-amber-800/60 font-semibold">
                {multiDigitResult.digitCount} Digits Detected
              </span>
            )}
          </div>
          <h3 className="text-base font-bold text-slate-100 mt-1 flex items-center gap-2">
            Classification &amp; Synaptic Activation Flow
          </h3>
          <p className="text-xs text-slate-400">
            Real-time multilayer perceptron inference with instant speech output and synaptic tracing.
          </p>
        </div>

        {/* View Mode Switcher: Full Synaptic Graph (First Preference) vs Simplified Output */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800">
            <button
              id="view-mode-synaptic"
              type="button"
              onClick={() => setViewMode('synaptic')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition flex items-center gap-1.5 ${
                viewMode === 'synaptic'
                  ? 'bg-indigo-600 text-white font-semibold shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5 text-amber-300" />
              Synaptic Graph
            </button>
            <button
              id="view-mode-simplified"
              type="button"
              onClick={() => setViewMode('simplified')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition flex items-center gap-1.5 ${
                viewMode === 'simplified'
                  ? 'bg-indigo-600 text-white font-semibold shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Award className="w-3.5 h-3.5" />
              Direct Output
            </button>
          </div>
        </div>
      </div>

      {/* Prominent Output Showcase (Always visible, highlighted) */}
      <div className="bg-slate-950 rounded-xl border border-slate-800 p-4 sm:p-5 shadow-inner space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300">
              Recognition Output Result
            </span>
          </div>

          {/* Voice / Speak Controls */}
          <div className="flex items-center gap-2.5">
            <button
              id="btn-speak-output"
              type="button"
              disabled={!isActuallyPredicting}
              onClick={() => {
                if (currentSpokenText) speakText(currentSpokenText);
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition ${
                isActuallyPredicting
                  ? 'bg-emerald-600 hover:bg-emerald-500 text-white border-emerald-500 shadow-md active:scale-95'
                  : 'bg-slate-900 text-slate-500 border-slate-800 cursor-not-allowed'
              }`}
              title="Speak the recognized number out loud"
            >
              <Volume2 className="w-3.5 h-3.5 text-white" />
              <span>Say Output</span>
            </button>

            <button
              id="btn-toggle-autospeak"
              type="button"
              onClick={() => setAutoSpeak(!autoSpeak)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border transition ${
                autoSpeak
                  ? 'bg-emerald-950/90 text-emerald-300 border-emerald-600 shadow-sm'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-300'
              }`}
              title={autoSpeak ? 'Auto-speak enabled on new drawing (Click to turn off)' : 'Enable auto-speak'}
            >
              {autoSpeak ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <Volume2 className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="font-semibold text-emerald-300">Auto-Say: ON</span>
                </>
              ) : (
                <>
                  <VolumeX className="w-3.5 h-3.5 text-slate-500" />
                  <span>Auto-Say: OFF</span>
                </>
              )}
            </button>
          </div>
        </div>

        {isActuallyPredicting ? (
          <div>
            {/* Multi-digit display (e.g. 4 digits drawn side-by-side) */}
            {multiDigitResult && multiDigitResult.hasContent && multiDigitResult.digitCount > 1 ? (
              <div className="space-y-3">
                <div className="bg-slate-900/90 border border-indigo-500/30 rounded-xl p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 shadow-lg">
                  <div>
                    <div className="text-[11px] font-mono uppercase text-indigo-400 font-semibold mb-1">
                      Multi-Digit Sequence ({multiDigitResult.digitCount} Digits Detected)
                    </div>
                    <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
                      {multiDigitResult.digits.map((item, idx) => {
                        const isSelected = idx === selectedDigitIndex;
                        const digitColor = OUTPUT_DIGIT_COLORS[item.digit];
                        return (
                          <button
                            key={idx}
                            id={`multi-digit-box-${idx}`}
                            type="button"
                            onClick={() => onSelectDigitIndex?.(idx)}
                            className={`flex flex-col items-center justify-center min-w-[56px] px-3 py-2 rounded-xl border font-mono transition-all transform ${
                              isSelected
                                ? 'scale-105 shadow-lg ring-2 ring-indigo-400'
                                : 'hover:scale-102 hover:bg-slate-800/80 opacity-80'
                            }`}
                            style={{
                              backgroundColor: `${digitColor}22`,
                              borderColor: isSelected ? digitColor : `${digitColor}60`,
                            }}
                          >
                            <span
                              className="text-3xl font-extrabold text-white"
                              style={{ color: digitColor }}
                            >
                              {item.digit}
                            </span>
                            <span className="text-[10px] text-slate-300 font-semibold">
                              {(item.confidence * 100).toFixed(0)}%
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <div className="sm:border-l sm:border-slate-800 sm:pl-6 text-left sm:text-right">
                    <div className="text-[10px] font-mono text-slate-400 uppercase">
                      Full Number Output
                    </div>
                    <div className="text-4xl font-extrabold font-mono text-white tracking-widest text-emerald-400">
                      {multiDigitResult.fullNumber}
                    </div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      Click any digit to view synaptic paths
                    </div>
                  </div>
                </div>

                {/* Selected digit details pill */}
                <div className="flex items-center justify-between bg-slate-900/60 px-3.5 py-2 rounded-lg border border-slate-800/70 text-xs">
                  <span className="text-slate-300 font-mono">
                    Active Inspection: <b className="text-indigo-300">Digit #{selectedDigitIndex + 1} ({multiDigitResult.digits[selectedDigitIndex]?.digit})</b>
                  </span>
                  <span className="text-emerald-400 font-mono font-semibold">
                    Confidence: {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ) : (
              /* Single digit display */
              <div
                className="bg-slate-900/90 border rounded-xl p-5 flex flex-col sm:flex-row items-center justify-between gap-5 shadow-lg"
                style={{
                  backgroundColor: `${topPredictedColor}12`,
                  borderColor: `${topPredictedColor}50`,
                }}
              >
                <div className="flex items-center gap-5">
                  {/* Huge Digit Badge */}
                  <div
                    id="primary-digit-badge"
                    className="w-20 h-20 sm:w-24 sm:h-24 rounded-2xl flex items-center justify-center text-5xl sm:text-6xl font-extrabold font-mono text-white shadow-2xl ring-4 ring-white/10"
                    style={{ backgroundColor: topPredictedColor }}
                  >
                    {prediction.predictedDigit}
                  </div>

                  <div>
                    <div className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400">
                      Predicted Handwritten Digit
                    </div>
                    <div className="text-2xl sm:text-3xl font-extrabold text-white flex items-center gap-2 mt-0.5">
                      <span>Digit {prediction.predictedDigit}</span>
                      <span className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800 font-mono font-bold">
                        <Check className="w-3 h-3 text-emerald-400" />
                        {(prediction.confidence * 100).toFixed(1)}% Match
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">
                      Calibrated with 16 sector spatial pooling and ReLU activation detector.
                    </p>
                  </div>
                </div>

                <div className="w-full sm:w-auto flex sm:flex-col items-center sm:items-end justify-between gap-2 border-t sm:border-t-0 sm:border-l border-slate-800/80 pt-3 sm:pt-0 sm:pl-5">
                  <div className="text-[11px] font-mono text-slate-400">Winning Class</div>
                  <div className="text-xl font-bold font-mono text-emerald-400">
                    P({prediction.predictedDigit}) = {(prediction.confidence * 100).toFixed(1)}%
                  </div>
                  <button
                    type="button"
                    onClick={() => speakText(currentSpokenText)}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition shadow"
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                    Speak Digit
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* Empty / Waiting state */
          <div className="p-8 text-center rounded-xl bg-slate-900/50 border border-slate-800/80 space-y-3">
            <div className="w-12 h-12 mx-auto rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-500 font-mono text-xl font-bold">
              ?
            </div>
            <div>
              <h4 className="text-sm font-semibold text-slate-300">
                Awaiting Drawing Input
              </h4>
              <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1">
                Draw any digit (0–9) or multiple digits (e.g., 4 digits) on the left drawing canvas to trigger neural classification.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Simplified Output View: Bar charts, statistics, and layer flow summary */}
      {viewMode === 'simplified' ? (
        <div className="space-y-4">
          {/* Class Distribution Bars */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-emerald-400" />
                Softmax Class Probabilities (Digits 0 to 9)
              </span>
              <span className="text-[11px] text-slate-400 font-mono">
                &sum; P(k) = 1.000
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
              {prediction.probabilities.map((prob, digit) => {
                const isWinner = digit === prediction.predictedDigit;
                const barColor = OUTPUT_DIGIT_COLORS[digit];

                return (
                  <div
                    key={digit}
                    id={`prob-card-${digit}`}
                    className={`p-2.5 rounded-lg border transition-all ${
                      isWinner
                        ? 'bg-slate-900 shadow-md ring-2'
                        : 'bg-slate-900/40 border-slate-800/80 hover:bg-slate-900/80'
                    }`}
                    style={{
                      borderColor: isWinner ? barColor : undefined,
                    }}
                  >
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span
                        className="font-mono font-bold px-2 py-0.5 rounded text-[11px]"
                        style={{
                          color: barColor,
                          backgroundColor: `${barColor}20`,
                        }}
                      >
                        Digit {digit}
                      </span>
                      <span
                        className={`font-mono text-xs ${
                          isWinner ? 'text-white font-extrabold' : 'text-slate-400'
                        }`}
                      >
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>

                    {/* Progress bar */}
                    <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.max(3, prob * 100)}%`,
                          backgroundColor: barColor,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Quick toggle to view synaptic graph */}
          <div className="flex items-center justify-between bg-slate-950 px-4 py-3 rounded-xl border border-slate-800 text-xs">
            <span className="text-slate-400">
              Want to see internal hidden neuron activations and synapses?
            </span>
            <button
              type="button"
              onClick={() => setViewMode('synaptic')}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 transition"
            >
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              Open Synaptic Graph
            </button>
          </div>
        </div>
      ) : (
        /* Full Synaptic Graph View */
        <div className="space-y-4">
          {/* Control Bar: Line Colors, Active Line Style & Filter Controls */}
          <div className="flex flex-wrap items-center justify-between gap-2.5 bg-slate-950 p-2.5 rounded-lg border border-slate-800 text-xs">
            {/* Line Active Style & Color Mode Selectors */}
            <div className="flex flex-wrap items-center gap-3">
              {/* Active Line Style: Flowing vs Solid */}
              <div className="flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-slate-400 font-medium">Line Activity:</span>
                <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded border border-slate-800">
                  <button
                    id="line-style-flowing"
                    type="button"
                    onClick={() => setLineStyle('flowing')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition flex items-center gap-1 ${
                      lineStyle === 'flowing'
                        ? 'bg-gradient-to-r from-emerald-600 to-teal-600 text-white font-semibold shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                    title="All lines actively streaming electrical signal pulses"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-300 animate-pulse" />
                    All Active (Flowing)
                  </button>
                  <button
                    id="line-style-solid"
                    type="button"
                    onClick={() => setLineStyle('solid')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition flex items-center gap-1 ${
                      lineStyle === 'solid'
                        ? 'bg-slate-700 text-white font-semibold shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Glowing Solid
                  </button>
                </div>
              </div>

              {/* Color Mode: Rainbow Spectrum vs Polarity */}
              <div className="flex items-center gap-1.5">
                <Palette className="w-3.5 h-3.5 text-indigo-400" />
                <span className="text-slate-400 font-medium">Colors:</span>
                <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded border border-slate-800">
                  <button
                    id="color-mode-spectrum"
                    type="button"
                    onClick={() => setColorMode('spectrum')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition flex items-center gap-1 ${
                      colorMode === 'spectrum'
                        ? 'bg-gradient-to-r from-cyan-600 to-indigo-600 text-white font-semibold shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <Sparkles className="w-2.5 h-2.5 text-amber-300" />
                    Spectrum
                  </button>
                  <button
                    id="color-mode-polarity"
                    type="button"
                    onClick={() => setColorMode('polarity')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition flex items-center gap-1 ${
                      colorMode === 'polarity'
                        ? 'bg-slate-700 text-white font-semibold shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Polarity (+ / -)
                  </button>
                  <button
                    id="color-mode-layer"
                    type="button"
                    onClick={() => setColorMode('layer')}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition flex items-center gap-1 ${
                      colorMode === 'layer'
                        ? 'bg-slate-700 text-white font-semibold shadow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Layer Flow
                  </button>
                </div>
              </div>
            </div>

            {/* Filter / Inspection options */}
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1.5 cursor-pointer text-slate-300 select-none">
                <input
                  id="chk-show-active-only"
                  type="checkbox"
                  checked={showOnlyActive}
                  onChange={(e) => setShowOnlyActive(e.target.checked)}
                  className="accent-indigo-500 rounded"
                />
                <span className="text-[11px]">Active Pathways Only</span>
              </label>

              {selectedNodeId && (
                <button
                  id="btn-clear-selection"
                  type="button"
                  onClick={() => setSelectedNodeId(null)}
                  className="text-[11px] text-amber-400 hover:text-amber-300 font-mono bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20"
                >
                  Reset Focus ({selectedNodeId})
                </button>
              )}
            </div>
          </div>

          {/* SVG Neural Network Architecture Visualizer with Colored Lines */}
          <div className="relative bg-slate-950 rounded-xl border border-slate-800/90 overflow-hidden shadow-inner p-2">
            {/* Layer Header Indicators */}
            <div className="absolute top-3 inset-x-4 flex justify-between pointer-events-none z-10 text-[11px] font-mono">
              <div className="bg-slate-900/90 px-3 py-1 rounded-md border border-slate-800 text-sky-400 font-semibold shadow">
                Input Layer (16 Receptive Fields)
              </div>
              <div className="bg-slate-900/90 px-3 py-1 rounded-md border border-slate-800 text-purple-400 font-semibold shadow">
                Hidden Layer (14 Feature Detectors)
              </div>
              <div className="bg-slate-900/90 px-3 py-1 rounded-md border border-slate-800 text-emerald-400 font-semibold shadow">
                Output Layer (10 Digits &bull; Softmax)
              </div>
            </div>

            {/* Responsive SVG Canvas */}
            <div className="w-full overflow-x-auto">
              <svg
                viewBox={`0 0 ${graphWidth} ${graphHeight}`}
                className="w-full h-auto min-w-[640px] select-none block"
                style={{ maxHeight: '460px' }}
              >
                <defs>
                  <style>{`
                    @keyframes synapticActiveFlow {
                      from {
                        stroke-dashoffset: 24;
                      }
                      to {
                        stroke-dashoffset: 0;
                      }
                    }
                    .synapse-flowing-active {
                      stroke-dasharray: 5 3;
                      animation: synapticActiveFlow 1.8s linear infinite;
                    }
                    .synapse-firing-active {
                      stroke-dasharray: 6 3;
                      animation: synapticActiveFlow 1.1s linear infinite;
                    }
                  `}</style>
                  {/* Radial gradient glow for active output */}
                  <radialGradient id="outputGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor={topPredictedColor} stopOpacity="0.8" />
                    <stop offset="100%" stopColor={topPredictedColor} stopOpacity="0" />
                  </radialGradient>
                </defs>

                {/* Background subtle column bands */}
                <rect
                  x={graphWidth * 0.08}
                  y="20"
                  width={graphWidth * 0.08}
                  height={graphHeight - 40}
                  fill="rgba(56, 189, 248, 0.03)"
                  rx="12"
                />
                <rect
                  x={graphWidth * 0.46}
                  y="20"
                  width={graphWidth * 0.08}
                  height={graphHeight - 40}
                  fill="rgba(168, 85, 247, 0.03)"
                  rx="12"
                />
                <rect
                  x={graphWidth * 0.84}
                  y="20"
                  width={graphWidth * 0.08}
                  height={graphHeight - 40}
                  fill="rgba(16, 185, 129, 0.03)"
                  rx="12"
                />

                {/* Synaptic Connections (Lines) */}
                <g id="synaptic-connections-group">
                  {visibleConnections.map((c) => {
                    const isWinningConnection =
                      c.toId === `out-${prediction.predictedDigit}` &&
                      isActuallyPredicting;
                    const strokeWidthVal = isWinningConnection
                      ? Math.max(2.2, c.width * 1.5)
                      : c.width;
                    const strokeOpacityVal = isWinningConnection
                      ? Math.min(1.0, c.opacity + 0.35)
                      : c.opacity;

                    return (
                      <line
                        key={c.id}
                        x1={c.fromX}
                        y1={c.fromY}
                        x2={c.toX}
                        y2={c.toY}
                        stroke={c.color}
                        strokeWidth={strokeWidthVal}
                        strokeOpacity={strokeOpacityVal}
                        className={
                          lineStyle === 'flowing'
                            ? isWinningConnection
                              ? 'synapse-firing-active'
                              : 'synapse-flowing-active'
                            : ''
                        }
                      />
                    );
                  })}
                </g>

                {/* Neurons (Nodes) */}
                <g id="network-nodes-group">
                  {nodes.map((node) => {
                    const isSelected = node.id === selectedNodeId;
                    const isWinningOutput =
                      node.layer === 'output' &&
                      node.index === prediction.predictedDigit &&
                      isActuallyPredicting;

                    let radius = 10;
                    if (node.layer === 'input') radius = 8;
                    if (node.layer === 'hidden') radius = 9;
                    if (node.layer === 'output') radius = 13;
                    if (isWinningOutput) radius = 16;
                    if (isSelected) radius += 2;

                    let fill = '#1e293b';
                    let stroke = '#475569';

                    if (node.layer === 'input') {
                      const act = node.activation;
                      fill = act > 0.1 ? `rgba(56, 189, 248, ${0.2 + act * 0.8})` : '#0f172a';
                      stroke = act > 0.1 ? '#38bdf8' : '#334155';
                    } else if (node.layer === 'hidden') {
                      const act = Math.min(1.0, node.activation);
                      fill = act > 0.05 ? `rgba(168, 85, 247, ${0.2 + act * 0.8})` : '#0f172a';
                      stroke = act > 0.05 ? '#c084fc' : '#334155';
                    } else if (node.layer === 'output') {
                      const prob = node.activation;
                      const color = OUTPUT_DIGIT_COLORS[node.index];
                      if (isWinningOutput) {
                        fill = color;
                        stroke = '#ffffff';
                      } else {
                        fill = prob > 0.08 ? `${color}33` : '#0f172a';
                        stroke = prob > 0.08 ? color : '#334155';
                      }
                    }

                    return (
                      <g
                        key={node.id}
                        id={`node-group-${node.id}`}
                        className="cursor-pointer transition-transform duration-150"
                        onClick={() =>
                          setSelectedNodeId(selectedNodeId === node.id ? null : node.id)
                        }
                      >
                        {/* Winning glow */}
                        {isWinningOutput && (
                          <circle
                            cx={node.x}
                            cy={node.y}
                            r={radius + 10}
                            fill="url(#outputGlow)"
                            className="animate-pulse"
                          />
                        )}

                        {/* Node circle */}
                        <circle
                          cx={node.x}
                          cy={node.y}
                          r={radius}
                          fill={fill}
                          stroke={isSelected ? '#f59e0b' : stroke}
                          strokeWidth={isSelected ? 3 : isWinningOutput ? 2.5 : 1.5}
                        />

                        {/* Node label */}
                        {node.layer === 'output' && (
                          <text
                            x={node.x}
                            y={node.y + 4.5}
                            textAnchor="middle"
                            fill={isWinningOutput ? '#ffffff' : '#cbd5e1'}
                            fontSize={isWinningOutput ? '13' : '11'}
                            fontWeight="bold"
                            fontFamily="monospace"
                            className="pointer-events-none select-none"
                          >
                            {node.index}
                          </text>
                        )}
                      </g>
                    );
                  })}
                </g>
              </svg>
            </div>

            {/* Selected Node Details Bar */}
            {selectedNode && (
              <div className="mt-2 bg-slate-900 border border-amber-500/30 rounded-lg p-2.5 text-xs text-slate-300 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-400" />
                  <span className="font-mono text-amber-300 font-bold">{selectedNode.label}</span>
                  <span className="text-slate-400">
                    Activation:{' '}
                    <b className="text-white font-mono">{selectedNode.activation.toFixed(3)}</b>
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedNodeId(null)}
                  className="text-[11px] text-amber-400 hover:underline"
                >
                  Clear focus
                </button>
              </div>
            )}
          </div>

          {/* Softmax probabilities bars in graph view as well */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-emerald-400" />
                Softmax Class Distribution
              </span>
              <span className="text-[11px] text-slate-400 font-mono">
                &sum; P(k) = 1.000
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
              {prediction.probabilities.map((prob, digit) => {
                const isWinner = digit === prediction.predictedDigit;
                const barColor = OUTPUT_DIGIT_COLORS[digit];

                return (
                  <div
                    key={digit}
                    className={`p-2 rounded-lg border transition-all ${
                      isWinner ? 'bg-slate-900 ring-1' : 'bg-slate-900/40 border-slate-800/80'
                    }`}
                    style={{
                      borderColor: isWinner ? barColor : undefined,
                    }}
                  >
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span
                        className="font-mono font-bold px-1.5 py-0.2 rounded text-[11px]"
                        style={{ color: barColor }}
                      >
                        Digit {digit}
                      </span>
                      <span
                        className={`font-mono text-[11px] ${
                          isWinner ? 'text-white font-bold' : 'text-slate-400'
                        }`}
                      >
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-300"
                        style={{
                          width: `${Math.max(3, prob * 100)}%`,
                          backgroundColor: barColor,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Collapsible Layer Insights & Deep Learning Details */}
      <div className="border border-slate-800/80 rounded-xl overflow-hidden">
        <button
          id="btn-toggle-layer-details"
          type="button"
          onClick={() => setShowLayerDetails(!showLayerDetails)}
          className="w-full flex items-center justify-between px-4 py-3 bg-slate-950 hover:bg-slate-900 text-xs font-medium text-slate-300 transition"
        >
          <span className="flex items-center gap-2">
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            Neural Network Layer Specifications &amp; Feature Descriptions
          </span>
          {showLayerDetails ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </button>

        {showLayerDetails && (
          <div className="p-4 bg-slate-900/60 border-t border-slate-800 text-xs text-slate-400 space-y-3 font-mono">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span className="text-sky-400 font-bold block mb-1">1. Input Layer (16 Neurons)</span>
                <p className="text-[11px] text-slate-500 font-sans leading-relaxed">
                  Extracts 16 pooled spatial sectors from the recentered matrix. Values represent stroke intensity in [0.0, 1.0].
                </p>
              </div>

              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span className="text-purple-400 font-bold block mb-1">2. Hidden Layer (14 Neurons)</span>
                <p className="text-[11px] text-slate-500 font-sans leading-relaxed">
                  Applies ReLU activation: <code className="text-slate-300">h = max(0, W1 &middot; x + b1)</code>. Neurons detect horizontal bars, vertical spines, loops, and diagonal crossings.
                </p>
              </div>

              <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                <span className="text-emerald-400 font-bold block mb-1">3. Output Layer (10 Neurons)</span>
                <p className="text-[11px] text-slate-500 font-sans leading-relaxed">
                  Computes logits <code className="text-slate-300">z = W2 &middot; h + b2</code> and normalizes with Softmax to produce probabilities for digits 0&ndash;9.
                </p>
              </div>
            </div>

            {/* Hidden Feature Catalog */}
            <div className="pt-2">
              <span className="text-slate-300 font-semibold block mb-1.5 font-sans">
                Hidden Neuron Feature Detectors:
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-[10px]">
                {HIDDEN_DESCRIPTIONS.map((desc, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-1.5 bg-slate-950 rounded border border-slate-800/80"
                  >
                    <span className="text-purple-400 font-bold">h[{i}]:</span>
                    <span className="text-slate-300 text-right">{desc}</span>
                    <span className="text-emerald-400 font-bold ml-2">
                      {prediction.hiddenActivations[i]?.toFixed(2) ?? '0.00'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
