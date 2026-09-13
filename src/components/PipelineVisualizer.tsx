import React, { useEffect, useRef } from 'react';
import {
  PipelineResult,
  Step1Result,
  Step2Result,
  Step3Result,
  Step4Result,
  Step5Result,
} from '../types';
import { ArrowRight, CheckCircle2, Crosshair, HelpCircle, Layers, Move } from 'lucide-react';

interface PipelineVisualizerProps {
  pipelineResult: PipelineResult;
}

// Canvas rendering helper for raw Uint8Array / Float32Array image buffers
const StepCanvas: React.FC<{
  data: Uint8Array | Float32Array;
  width: number;
  height: number;
  displaySize?: number;
  isFloat?: boolean;
  pixelated?: boolean;
  overlayBox?: { minX: number; minY: number; width: number; height: number };
  crosshair?: { cx: number; cy: number; color?: string; label?: string };
}> = ({
  data,
  width,
  height,
  displaySize = 130,
  isFloat = false,
  pixelated = true,
  overlayBox,
  crosshair,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw base grayscale pixels
    const imgData = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      let val = 0;
      if (isFloat) {
        val = Math.round((data as Float32Array)[i] * 255);
      } else {
        val = (data as Uint8Array)[i];
      }
      val = Math.max(0, Math.min(255, val));
      const idx = i * 4;
      imgData.data[idx] = val;     // R
      imgData.data[idx + 1] = val; // G
      imgData.data[idx + 2] = val; // B
      imgData.data[idx + 3] = 255; // A
    }
    ctx.putImageData(imgData, 0, 0);

    // Overlay bounding box if provided
    if (overlayBox) {
      ctx.strokeStyle = '#38bdf8'; // Sky blue
      ctx.lineWidth = 2;
      ctx.strokeRect(
        overlayBox.minX,
        overlayBox.minY,
        overlayBox.width,
        overlayBox.height
      );
    }

    // Overlay center of mass reticle if provided
    if (crosshair) {
      const color = crosshair.color || '#ef4444';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      const size = Math.max(3, Math.floor(width * 0.12));

      // Draw crosshair
      ctx.beginPath();
      ctx.arc(crosshair.cx, crosshair.cy, size * 0.6, 0, Math.PI * 2);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(crosshair.cx - size, crosshair.cy);
      ctx.lineTo(crosshair.cx + size, crosshair.cy);
      ctx.moveTo(crosshair.cx, crosshair.cy - size);
      ctx.lineTo(crosshair.cx, crosshair.cy + size);
      ctx.stroke();
    }
  }, [data, width, height, isFloat, overlayBox, crosshair]);

  return (
    <div
      className="relative flex items-center justify-center bg-black rounded border border-slate-700 shadow-inner overflow-hidden"
      style={{ width: displaySize, height: displaySize }}
    >
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          width: '100%',
          height: '100%',
          imageRendering: pixelated ? 'pixelated' : 'auto',
        }}
      />
    </div>
  );
};

export const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({
  pipelineResult,
}) => {
  const { hasContent, message, step1, step2, step3, step4, step5 } = pipelineResult;

  if (!hasContent || !step1 || !step2 || !step3 || !step4 || !step5) {
    return (
      <div id="pipeline-empty-notice" className="bg-slate-900 border border-amber-500/30 rounded-xl p-8 text-center shadow-lg">
        <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-amber-500/10 flex items-center justify-center border border-amber-500/20 text-amber-400">
          <HelpCircle className="w-6 h-6" />
        </div>
        <h3 className="text-base font-semibold text-slate-200">
          Canvas Awaiting Input
        </h3>
        <p className="text-xs text-slate-400 max-w-md mx-auto mt-1 leading-relaxed">
          {message || 'Draw a digit (0–9) on the canvas to trigger the 5-step MNIST computer vision normalization pipeline in real time.'}
        </p>
      </div>
    );
  }

  return (
    <div id="pipeline-visualizer-section" className="space-y-4">
      {/* 4-Step Pipeline Flow Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Step 1: Grayscale & Binarization */}
        <div id="pipeline-step-1" className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex flex-col items-center text-center shadow">
          <div className="flex items-center justify-between w-full mb-2">
            <span className="text-[11px] font-mono text-emerald-400 font-semibold bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/40">
              Step 1
            </span>
            <span className="text-[11px] text-slate-400 font-mono">280&times;280</span>
          </div>

          <h4 className="text-xs font-semibold text-slate-200 mb-1.5">
            Grayscale &amp; Threshold
          </h4>

          <div className="my-1.5">
            <StepCanvas
              data={step1.binary}
              width={step1.width}
              height={step1.height}
              displaySize={110}
            />
          </div>

          <div className="text-[11px] text-slate-400 font-mono mt-2 space-y-0.5">
            <div>Stroke Pixels: <span className="text-slate-200 font-semibold">{step1.nonZeroCount}</span></div>
            <div className="text-slate-500">Threshold: &ge; 25</div>
          </div>
        </div>

        {/* Step 2: Bounding Box & Tight Crop */}
        <div id="pipeline-step-2" className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex flex-col items-center text-center shadow">
          <div className="flex items-center justify-between w-full mb-2">
            <span className="text-[11px] font-mono text-sky-400 font-semibold bg-sky-950/60 px-2 py-0.5 rounded border border-sky-800/40">
              Step 2
            </span>
            <span className="text-[11px] text-slate-400 font-mono">
              {step2.croppedWidth}&times;{step2.croppedHeight}
            </span>
          </div>

          <h4 className="text-xs font-semibold text-slate-200 mb-1.5">
            Bounding Box Crop
          </h4>

          <div className="my-1.5">
            <StepCanvas
              data={step2.data}
              width={step2.croppedWidth}
              height={step2.croppedHeight}
              displaySize={110}
            />
          </div>

          <div className="text-[11px] text-slate-400 font-mono mt-2 space-y-0.5">
            <div>Box: ({step2.box.minX}, {step2.box.minY})</div>
            <div className="text-slate-500">W: {step2.croppedWidth}px, H: {step2.croppedHeight}px</div>
          </div>
        </div>

        {/* Step 3: Square Aspect Padding */}
        <div id="pipeline-step-3" className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex flex-col items-center text-center shadow">
          <div className="flex items-center justify-between w-full mb-2">
            <span className="text-[11px] font-mono text-purple-400 font-semibold bg-purple-950/60 px-2 py-0.5 rounded border border-purple-800/40">
              Step 3
            </span>
            <span className="text-[11px] text-slate-400 font-mono">
              {step3.size}&times;{step3.size}
            </span>
          </div>

          <h4 className="text-xs font-semibold text-slate-200 mb-1.5">
            Square Padding
          </h4>

          <div className="my-1.5">
            <StepCanvas
              data={step3.data}
              width={step3.size}
              height={step3.size}
              displaySize={110}
            />
          </div>

          <div className="text-[11px] text-slate-400 font-mono mt-2 space-y-0.5">
            <div>Pad: 1:1 Aspect</div>
            <div className="text-slate-500">Offset: (+{step3.offsetX}, +{step3.offsetY})</div>
          </div>
        </div>

        {/* Step 4: Downscale / Pixelation */}
        <div id="pipeline-step-4" className="bg-slate-900 border border-slate-800 rounded-xl p-3.5 flex flex-col items-center text-center shadow">
          <div className="flex items-center justify-between w-full mb-2">
            <span className="text-[11px] font-mono text-amber-400 font-semibold bg-amber-950/60 px-2 py-0.5 rounded border border-amber-800/40">
              Step 4
            </span>
            <span className="text-[11px] text-slate-400 font-mono">
              {step4.resolution}&times;{step4.resolution}
            </span>
          </div>

          <h4 className="text-xs font-semibold text-slate-200 mb-1.5">
            Area Downsample
          </h4>

          <div className="my-1.5">
            <StepCanvas
              data={step4.data}
              width={step4.resolution}
              height={step4.resolution}
              displaySize={110}
              isFloat
              pixelated
            />
          </div>

          <div className="text-[11px] text-slate-400 font-mono mt-2 space-y-0.5">
            <div>cv2.INTER_AREA</div>
            <div className="text-slate-500">{step4.resolution * step4.resolution} Matrix Grid</div>
          </div>
        </div>
      </div>

      {/* Step 5: Center of Mass Recenter Banner */}
      <div id="pipeline-step-5" className="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-3 border-b border-slate-800">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-indigo-400 font-semibold bg-indigo-950/60 px-2.5 py-0.5 rounded border border-indigo-800/40">
                Step 5: Final MNIST Normalization
              </span>
              <span className="text-xs text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Ready for ML Model
              </span>
            </div>
            <h3 className="text-sm font-semibold text-slate-100 mt-1">
              Center of Mass Adjustment (Recentering)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Calculates image moments <span className="font-mono text-slate-300">cx = M10/M00, cy = M01/M00</span> and applies subpixel translation to center the digit mass at ({(step5.resolution - 1) / 2}, {(step5.resolution - 1) / 2}).
            </p>
          </div>

          <div className="flex items-center gap-3 bg-slate-950/80 px-3 py-2 rounded-lg border border-slate-800 text-xs font-mono">
            <div>
              <span className="text-slate-500 block text-[10px]">SHIFT (&Delta;x, &Delta;y)</span>
              <span className="text-indigo-300 font-semibold">
                {step5.shift.dx > 0 ? `+${step5.shift.dx.toFixed(2)}` : step5.shift.dx.toFixed(2)}, {step5.shift.dy > 0 ? `+${step5.shift.dy.toFixed(2)}` : step5.shift.dy.toFixed(2)}
              </span>
            </div>
            <div className="h-6 w-px bg-slate-800" />
            <div>
              <span className="text-slate-500 block text-[10px]">NEW CENTER</span>
              <span className="text-emerald-400 font-semibold">
                ({step5.finalCom.cx.toFixed(1)}, {step5.finalCom.cy.toFixed(1)})
              </span>
            </div>
          </div>
        </div>

        {/* Before vs After Comparison */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
          <div className="bg-slate-950 p-3.5 rounded-lg border border-slate-800/80 flex items-center gap-4">
            <StepCanvas
              data={step4.data}
              width={step4.resolution}
              height={step4.resolution}
              displaySize={96}
              isFloat
              crosshair={{
                cx: step5.initialCom.cx,
                cy: step5.initialCom.cy,
                color: '#f87171',
              }}
            />
            <div className="text-xs">
              <span className="text-red-400 font-medium flex items-center gap-1">
                <Crosshair className="w-3.5 h-3.5" /> Initial Center of Mass
              </span>
              <div className="font-mono text-slate-300 mt-1">
                cx: {step5.initialCom.cx.toFixed(2)}, cy: {step5.initialCom.cy.toFixed(2)}
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                Natural drawn off-center bias before correction.
              </p>
            </div>
          </div>

          <div className="bg-slate-950 p-3.5 rounded-lg border border-indigo-500/20 flex items-center gap-4">
            <StepCanvas
              data={step5.data}
              width={step5.resolution}
              height={step5.resolution}
              displaySize={96}
              isFloat
              crosshair={{
                cx: step5.finalCom.cx,
                cy: step5.finalCom.cy,
                color: '#34d399',
              }}
            />
            <div className="text-xs">
              <span className="text-emerald-400 font-medium flex items-center gap-1">
                <Crosshair className="w-3.5 h-3.5" /> Recentered Matrix Output
              </span>
              <div className="font-mono text-slate-300 mt-1">
                cx: {step5.finalCom.cx.toFixed(2)}, cy: {step5.finalCom.cy.toFixed(2)}
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                Target center ({(step5.resolution - 1) / 2}, {(step5.resolution - 1) / 2}) aligned.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
