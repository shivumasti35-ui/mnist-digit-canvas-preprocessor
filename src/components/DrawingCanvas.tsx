import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Eraser, RotateCcw, Sparkles } from 'lucide-react';
import { PRESET_DIGITS, drawPresetOnCanvas } from '../utils/sampleDigits';

interface DrawingCanvasProps {
  strokeWidth: number;
  onCanvasChange: (canvas: HTMLCanvasElement) => void;
  onClear: () => void;
}

export const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  strokeWidth,
  onCanvasChange,
  onClear,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [strokeHistory, setStrokeHistory] = useState<ImageData[]>([]);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);

  // Initialize canvas with black background
  const resetCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setStrokeHistory([]);
    onClear();
  }, [onClear]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Default load digit 3 preset on initial mount for instant preview
    const preset3 = PRESET_DIGITS[0];
    drawPresetOnCanvas(canvas, preset3, strokeWidth);
    onCanvasChange(canvas);
  }, []);

  const saveState = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setStrokeHistory((prev) => [...prev.slice(-10), imgData]);
  };

  const handleUndo = () => {
    const canvas = canvasRef.current;
    if (!canvas || strokeHistory.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const newHistory = [...strokeHistory];
    newHistory.pop(); // Remove current
    setStrokeHistory(newHistory);

    if (newHistory.length > 0) {
      const prev = newHistory[newHistory.length - 1];
      ctx.putImageData(prev, 0, 0);
    } else {
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    onCanvasChange(canvas);
  };

  const getCanvasCoords = (
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ): { x: number; y: number } | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    let clientX = 0;
    let clientY = 0;

    if ('touches' in e) {
      if (e.touches.length === 0) return null;
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  };

  const startDrawing = (
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    e.preventDefault();
    const coords = getCanvasCoords(e);
    if (!coords) return;

    saveState();
    setIsDrawing(true);
    lastPointRef.current = coords;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Draw single point/dot
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(coords.x, coords.y, strokeWidth / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  const draw = (
    e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>
  ) => {
    if (!isDrawing) return;
    e.preventDefault();
    const coords = getCanvasCoords(e);
    if (!coords || !lastPointRef.current) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(lastPointRef.current.x, lastPointRef.current.y);

    const midX = (lastPointRef.current.x + coords.x) / 2;
    const midY = (lastPointRef.current.y + coords.y) / 2;
    ctx.quadraticCurveTo(lastPointRef.current.x, lastPointRef.current.y, midX, midY);
    ctx.stroke();

    lastPointRef.current = coords;
  };

  const stopDrawing = () => {
    if (!isDrawing) return;
    setIsDrawing(false);
    lastPointRef.current = null;

    const canvas = canvasRef.current;
    if (canvas) {
      onCanvasChange(canvas);
    }
  };

  const handleLoadPreset = (digit: (typeof PRESET_DIGITS)[0]) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    saveState();
    drawPresetOnCanvas(canvas, digit, strokeWidth);
    onCanvasChange(canvas);
  };

  return (
    <div id="drawing-canvas-card" className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
            Drawing Canvas
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            280 &times; 280 px &bull; Black background, White stroke
          </p>
        </div>
        <span className="text-xs font-mono bg-slate-800 text-slate-300 px-2 py-0.5 rounded border border-slate-700">
          Stroke: {strokeWidth}px
        </span>
      </div>

      {/* The 280x280 Canvas container */}
      <div className="flex justify-center my-3">
        <div className="relative p-1 bg-gradient-to-b from-slate-700 to-slate-800 rounded-lg shadow-inner">
          <canvas
            id="mnist-drawing-canvas"
            ref={canvasRef}
            width={280}
            height={280}
            onMouseDown={startDrawing}
            onMouseMove={draw}
            onMouseUp={stopDrawing}
            onMouseLeave={stopDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={stopDrawing}
            onTouchCancel={stopDrawing}
            className="w-[280px] h-[280px] rounded cursor-crosshair touch-none select-none block"
            style={{
              boxShadow: '0 0 20px rgba(0, 0, 0, 0.8)',
            }}
          />

          {/* Target grid overlay guide (subtle 28x28 faint guide when idle) */}
          <div className="absolute inset-1 pointer-events-none opacity-10 border border-dashed border-white" />
        </div>
      </div>

      {/* Canvas Actions */}
      <div className="flex items-center gap-2 mt-4">
        <button
          id="btn-clear-canvas"
          type="button"
          onClick={resetCanvas}
          className="flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs font-medium bg-red-500/10 hover:bg-red-500/20 text-red-300 border border-red-500/30 transition-colors"
        >
          <Eraser className="w-3.5 h-3.5" />
          Clear Canvas
        </button>

        <button
          id="btn-undo-stroke"
          type="button"
          onClick={handleUndo}
          disabled={strokeHistory.length === 0}
          className="flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-slate-200 border border-slate-700 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Undo
        </button>
      </div>

      {/* Preset Digits Quick Bar */}
      <div className="mt-4 pt-3 border-t border-slate-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-medium text-slate-400 flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-amber-400" />
            Quick Presets:
          </span>
          <span className="text-[11px] text-slate-500">Test handwritten digits</span>
        </div>
        <div className="grid grid-cols-5 gap-1.5">
          {PRESET_DIGITS.map((p) => (
            <button
              key={p.digit}
              id={`preset-digit-${p.digit}`}
              type="button"
              onClick={() => handleLoadPreset(p)}
              className="py-1.5 px-1 rounded-md bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-xs font-mono font-medium text-slate-200 hover:text-white transition text-center"
              title={p.description}
            >
              Digit {p.digit}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
