import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { DrawingCanvas } from './components/DrawingCanvas';
import { PipelineVisualizer } from './components/PipelineVisualizer';
import { HeatmapMatrixView } from './components/HeatmapMatrixView';
import { NeuralNetworkVisualizer } from './components/NeuralNetworkVisualizer';
import { PythonScriptModal } from './components/PythonScriptModal';
import { PipelineResult, PreprocessOptions } from './types';
import { runMnistPipeline } from './utils/imageProcessing';
import { segmentAndRecognizeCanvas, MultiDigitResult } from './utils/digitSegmentation';
import { Code2, GitMerge, Layers, Sparkles } from 'lucide-react';

export default function App() {
  const [strokeWidth, setStrokeWidth] = useState<number>(18);
  const [options, setOptions] = useState<PreprocessOptions>({
    targetResolution: 28,
    binarizationThreshold: 30,
    squarePaddingFactor: 1.28,
    recenterEnabled: true,
  });

  const [pipelineResult, setPipelineResult] = useState<PipelineResult>({
    hasContent: false,
    message: 'Canvas is empty. Draw a digit (0–9) on the canvas to begin preprocessing.',
  });

  const [multiDigitResult, setMultiDigitResult] = useState<MultiDigitResult | null>(null);
  const [selectedDigitIndex, setSelectedDigitIndex] = useState<number>(0);

  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);
  const [isPythonModalOpen, setIsPythonModalOpen] = useState<boolean>(false);

  // Keep a reference to the active canvas for re-running when options change
  const activeCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const handleCanvasChange = useCallback(
    (canvas: HTMLCanvasElement) => {
      activeCanvasRef.current = canvas;
      const result = runMnistPipeline(canvas, options);
      const multi = segmentAndRecognizeCanvas(canvas, options);
      setPipelineResult(result);
      setMultiDigitResult(multi);
      setSelectedDigitIndex(0);
    },
    [options]
  );

  const handleClear = useCallback(() => {
    setPipelineResult({
      hasContent: false,
      message: 'Canvas is empty. Draw a digit on the canvas to begin preprocessing.',
    });
    setMultiDigitResult(null);
    setSelectedDigitIndex(0);
  }, []);

  // When options (targetResolution, threshold, etc.) change, re-run pipeline on current canvas
  useEffect(() => {
    if (activeCanvasRef.current) {
      const result = runMnistPipeline(activeCanvasRef.current, options);
      const multi = segmentAndRecognizeCanvas(activeCanvasRef.current, options);
      setPipelineResult(result);
      setMultiDigitResult(multi);
    }
  }, [options]);

  // If user selected a specific digit in a multi-digit drawing, show that digit's step5 in the pipeline
  const activeStep5 =
    multiDigitResult &&
    multiDigitResult.hasContent &&
    multiDigitResult.digits[selectedDigitIndex]?.step5Result
      ? multiDigitResult.digits[selectedDigitIndex].step5Result
      : pipelineResult.step5;

  return (
    <div id="mnist-preprocessor-app" className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 antialiased font-sans">
      {/* Streamlit Sidebar */}
      <Sidebar
        strokeWidth={strokeWidth}
        setStrokeWidth={setStrokeWidth}
        options={options}
        setOptions={setOptions}
        onReset={() => {
          const btn = document.getElementById('btn-clear-canvas');
          if (btn) btn.click();
        }}
        onOpenPythonCode={() => setIsPythonModalOpen(true)}
        isCollapsed={isSidebarCollapsed}
        setIsCollapsed={setIsSidebarCollapsed}
      />

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col h-screen overflow-y-auto">
        {/* Streamlit Top Header Bar */}
        <header id="app-header" className="sticky top-0 z-10 bg-slate-900/90 backdrop-blur border-b border-slate-800 px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-sm shadow-red-500/50" />
            <div>
              <h1 className="text-sm sm:text-base font-bold text-slate-100 flex items-center gap-2">
                MNIST Digit Preprocessing &amp; Neural Network Pipeline
                <span className="hidden md:inline-block text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800/60">
                  AI &amp; DL TF-09Aug(E)
                </span>
              </h1>
              <p className="text-xs text-slate-400">
                Crop &bull; Aspect Pad &bull; Area Downsample &bull; Recenter &bull; Input &rarr; Hidden &rarr; Output Network (Colored Synapses)
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              id="btn-header-py-code"
              type="button"
              onClick={() => setIsPythonModalOpen(true)}
              className="flex items-center gap-1.5 py-1.5 px-3 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
            >
              <Code2 className="w-3.5 h-3.5 text-amber-400" />
              <span className="hidden sm:inline">Streamlit Code</span> (app.py)
            </button>
          </div>
        </header>

        {/* Primary Interactive Section: Drawing Canvas (Left) and Synaptic Connections & Activation Flow (Right) */}
        <div className="flex-1 p-4 sm:p-6 max-w-7xl mx-auto w-full space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Drawing Canvas & Controls */}
            <div className="lg:col-span-5 xl:col-span-4 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-semibold uppercase tracking-wider text-slate-400">
                  Input &bull; Drawing Canvas
                </span>
                <span className="text-[11px] text-emerald-400 font-medium flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                  Live Reactive
                </span>
              </div>

              {/* Drawing Canvas Component */}
              <DrawingCanvas
                strokeWidth={strokeWidth}
                onCanvasChange={handleCanvasChange}
                onClear={handleClear}
              />

              {/* Pipeline summary metadata card */}
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs text-slate-400 space-y-2.5 shadow">
                <div className="flex items-center justify-between text-slate-300 font-medium">
                  <span className="flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                    System Specifications
                  </span>
                  <span className="font-mono text-indigo-400">
                    {options.targetResolution}&times;{options.targetResolution} &rarr; MLP
                  </span>
                </div>
                <ul className="space-y-1.5 text-[11px] font-mono text-slate-400">
                  <li className="flex items-center justify-between">
                    <span>Input Canvas:</span>
                    <span className="text-slate-200">280 &times; 280 px</span>
                  </li>
                  <li className="flex items-center justify-between">
                    <span>Downsample Method:</span>
                    <span className="text-slate-200">cv2.INTER_AREA</span>
                  </li>
                  <li className="flex items-center justify-between">
                    <span>Recenter:</span>
                    <span className="text-emerald-400 font-semibold">
                      {options.recenterEnabled ? 'Center of Mass' : 'Disabled'}
                    </span>
                  </li>
                  <li className="flex items-center justify-between">
                    <span>Network Topology:</span>
                    <span className="text-sky-300 font-semibold">16 In &rarr; 14 Hid &rarr; 10 Out</span>
                  </li>
                  <li className="flex items-center justify-between">
                    <span>Active Synapses:</span>
                    <span className="text-emerald-400 font-semibold">364 Pathways Firing</span>
                  </li>
                </ul>
              </div>
            </div>

            {/* Right Column: Synaptic Connections & Activation Flow (Placed directly to the right of Drawing Canvas) */}
            <div className="lg:col-span-7 xl:col-span-8 space-y-4">
              <NeuralNetworkVisualizer
                step5Result={activeStep5}
                hasContent={pipelineResult.hasContent || (multiDigitResult?.hasContent ?? false)}
                multiDigitResult={multiDigitResult}
                selectedDigitIndex={selectedDigitIndex}
                onSelectDigitIndex={setSelectedDigitIndex}
              />
            </div>
          </div>

          {/* Secondary Section: 5-Step Image Processing Pipeline & Detailed Heatmap/Matrix */}
          <div className="pt-2 border-t border-slate-800/80 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-indigo-400" />
                  Intermediate Preprocessing Pipeline (Steps 1 &ndash; 5)
                </h2>
                <p className="text-xs text-slate-400">
                  OpenCV digit transformations: Grayscale &rarr; Bounding Box Crop &rarr; Aspect-Preserving Pad &rarr; Area Downsample &rarr; Center-of-Mass Recenter.
                </p>
              </div>

              {pipelineResult.hasContent && (
                <span className="text-xs text-slate-400 font-mono self-start sm:self-auto bg-slate-900 px-3 py-1 rounded-lg border border-slate-800">
                  Resolution: <b className="text-indigo-400">{options.targetResolution}&times;{options.targetResolution}</b> ({options.targetResolution * options.targetResolution} pixels)
                </span>
              )}
            </div>

            {/* Step 1 through 5 Visualizer */}
            <PipelineVisualizer pipelineResult={pipelineResult} />

            {/* Pixel Heatmap, 2D Numeric Matrix & 1D Vector Deep Dive */}
            {(pipelineResult.hasContent || (multiDigitResult?.hasContent ?? false)) && activeStep5 && (
              <div className="pt-2">
                <HeatmapMatrixView step5Result={activeStep5} />
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Streamlit Python Code Modal */}
      <PythonScriptModal
        isOpen={isPythonModalOpen}
        onClose={() => setIsPythonModalOpen(false)}
      />
    </div>
  );
}
