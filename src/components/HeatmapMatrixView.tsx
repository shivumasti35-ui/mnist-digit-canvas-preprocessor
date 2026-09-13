import React, { useState } from 'react';
import { Step5Result } from '../types';
import { Check, Copy, Grid3X3, Info, Sliders, Table2 } from 'lucide-react';

interface HeatmapMatrixViewProps {
  step5Result: Step5Result;
}

type ColormapType = 'mnist-gray' | 'viridis' | 'inferno' | 'cyan';

// Color map calculation helpers
function getCellColor(value: number, colormap: ColormapType): string {
  // value is in range [0, 1]
  const v = Math.max(0, Math.min(1, value));

  if (colormap === 'mnist-gray') {
    const byte = Math.round(v * 255);
    return `rgb(${byte}, ${byte}, ${byte})`;
  }

  if (colormap === 'cyan') {
    // 0 is black, 1 is bright electric cyan
    const r = Math.round(v * 34);
    const g = Math.round(v * 211);
    const b = Math.round(v * 238);
    return `rgb(${r}, ${g}, ${b})`;
  }

  if (colormap === 'inferno') {
    // simplified perceptually uniform warm ramp: black -> purple -> orange -> yellow
    if (v < 0.25) {
      const t = v / 0.25;
      return `rgb(${Math.round(60 * t)}, 0, ${Math.round(110 * t)})`;
    } else if (v < 0.5) {
      const t = (v - 0.25) / 0.25;
      return `rgb(${Math.round(60 + 120 * t)}, ${Math.round(10 * t)}, ${Math.round(110 - 40 * t)})`;
    } else if (v < 0.75) {
      const t = (v - 0.5) / 0.25;
      return `rgb(${Math.round(180 + 55 * t)}, ${Math.round(10 + 130 * t)}, ${Math.round(70 - 70 * t)})`;
    } else {
      const t = (v - 0.75) / 0.25;
      return `rgb(255, ${Math.round(140 + 115 * t)}, ${Math.round(200 * t)})`;
    }
  }

  // default 'viridis' style: deep purple -> blue-teal -> green -> yellow
  if (v < 0.33) {
    const t = v / 0.33;
    return `rgb(${Math.round(68 - 30 * t)}, ${Math.round(1 + 90 * t)}, ${Math.round(84 + 60 * t)})`;
  } else if (v < 0.66) {
    const t = (v - 0.33) / 0.33;
    return `rgb(${Math.round(38 - 5 * t)}, ${Math.round(91 + 75 * t)}, ${Math.round(144 - 30 * t)})`;
  } else {
    const t = (v - 0.66) / 0.34;
    return `rgb(${Math.round(33 + 220 * t)}, ${Math.round(166 + 80 * t)}, ${Math.round(114 - 80 * t)})`;
  }
}

export const HeatmapMatrixView: React.FC<HeatmapMatrixViewProps> = ({
  step5Result,
}) => {
  const { resolution, data, rawBytes } = step5Result;
  const [activeTab, setActiveTab] = useState<'heatmap' | 'matrix' | 'vector'>('heatmap');
  const [colormap, setColormap] = useState<ColormapType>('viridis');
  const [hoveredPixel, setHoveredPixel] = useState<{
    x: number;
    y: number;
    raw: number;
    normalized: number;
  } | null>(null);
  const [copied, setCopied] = useState(false);

  // Generate python vector snippet
  const pythonVectorSnippet = `import numpy as np\n\n# Flattened 1D normalized input vector (${resolution * resolution} floats)\nx_sample = np.array([\n  ${Array.from(data, (v: number) => v.toFixed(3)).join(', ')}\n], dtype=np.float32)`;

  const handleCopyVector = () => {
    navigator.clipboard.writeText(pythonVectorSnippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div id="heatmap-matrix-container" className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
      {/* Top Header & Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
        <div>
          <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Grid3X3 className="w-4 h-4 text-amber-400" />
            Final {resolution} &times; {resolution} Representation
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Explore the final pixel tensor, 2D matrix intensities, and flattened ML vector.
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 self-start sm:self-auto">
          <button
            id="tab-btn-heatmap"
            type="button"
            onClick={() => setActiveTab('heatmap')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition ${
              activeTab === 'heatmap'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Heatmap
          </button>
          <button
            id="tab-btn-matrix"
            type="button"
            onClick={() => setActiveTab('matrix')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition ${
              activeTab === 'matrix'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            2D Matrix
          </button>
          <button
            id="tab-btn-vector"
            type="button"
            onClick={() => setActiveTab('vector')}
            className={`px-3 py-1.5 rounded text-xs font-medium transition ${
              activeTab === 'vector'
                ? 'bg-indigo-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            1D Vector ({resolution * resolution})
          </button>
        </div>
      </div>

      {/* Tab 1: Heatmap View */}
      {activeTab === 'heatmap' && (
        <div id="tab-content-heatmap" className="pt-4">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Sliders className="w-3.5 h-3.5" />
              <span>Colormap:</span>
              <div className="flex items-center gap-1 bg-slate-950 p-0.5 rounded border border-slate-800">
                {(
                  [
                    { id: 'viridis', label: 'Viridis' },
                    { id: 'inferno', label: 'Inferno' },
                    { id: 'cyan', label: 'Cyan' },
                    { id: 'mnist-gray', label: 'Grayscale' },
                  ] as const
                ).map((c) => (
                  <button
                    key={c.id}
                    id={`colormap-${c.id}`}
                    type="button"
                    onClick={() => setColormap(c.id)}
                    className={`px-2 py-0.5 rounded text-[11px] font-medium transition ${
                      colormap === c.id
                        ? 'bg-slate-700 text-white font-semibold'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Hover coordinate inspector readout */}
            <div className="text-xs font-mono bg-slate-950 px-3 py-1 rounded border border-slate-800 text-slate-300">
              {hoveredPixel ? (
                <span>
                  Pixel [Row: <b className="text-amber-400">{hoveredPixel.y}</b>, Col: <b className="text-amber-400">{hoveredPixel.x}</b>] &bull; Byte: <b className="text-emerald-400">{hoveredPixel.raw}</b> &bull; Float: <b className="text-sky-400">{hoveredPixel.normalized.toFixed(3)}</b>
                </span>
              ) : (
                <span className="text-slate-500">
                  Hover over cells to inspect intensity values
                </span>
              )}
            </div>
          </div>

          {/* Responsive Interactive Heatmap Grid */}
          <div className="flex flex-col items-center justify-center p-3 bg-black/90 rounded-lg border border-slate-800">
            <div
              className="grid gap-[1px] bg-slate-800/40 p-1 rounded max-w-full overflow-auto"
              style={{
                gridTemplateColumns: `repeat(${resolution}, minmax(0, 1fr))`,
                width: resolution <= 28 ? 'min(100%, 360px)' : 'min(100%, 460px)',
                aspectRatio: '1 / 1',
              }}
              onMouseLeave={() => setHoveredPixel(null)}
            >
              {Array.from({ length: resolution * resolution }).map((_, idx) => {
                const x = idx % resolution;
                const y = Math.floor(idx / resolution);
                const normVal = data[idx];
                const rawVal = rawBytes[idx];
                const bg = getCellColor(normVal, colormap);

                return (
                  <div
                    key={idx}
                    className="relative cursor-pointer transition-transform hover:scale-125 hover:z-20 hover:ring-1 hover:ring-white rounded-[1px]"
                    style={{ backgroundColor: bg }}
                    onMouseEnter={() =>
                      setHoveredPixel({
                        x,
                        y,
                        raw: rawVal,
                        normalized: normVal,
                      })
                    }
                  />
                );
              })}
            </div>

            {/* Gradient scale legend */}
            <div className="flex items-center gap-2 mt-3 text-[11px] font-mono text-slate-400">
              <span>0 (0.00)</span>
              <div
                className="w-44 h-2.5 rounded-full border border-slate-700"
                style={{
                  background:
                    colormap === 'viridis'
                      ? 'linear-gradient(to right, rgb(68,1,84), rgb(38,91,144), rgb(33,166,114), rgb(253,231,37))'
                      : colormap === 'inferno'
                      ? 'linear-gradient(to right, rgb(0,0,0), rgb(60,0,110), rgb(180,10,70), rgb(255,255,200))'
                      : colormap === 'cyan'
                      ? 'linear-gradient(to right, rgb(0,0,0), rgb(34,211,238))'
                      : 'linear-gradient(to right, rgb(0,0,0), rgb(255,255,255))',
                }}
              />
              <span>255 (1.00)</span>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: 2D Numeric Matrix Table */}
      {activeTab === 'matrix' && (
        <div id="tab-content-matrix" className="pt-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <Table2 className="w-3.5 h-3.5 text-indigo-400" />
              Raw integer pixel intensities [0 &ndash; 255]:
            </span>
            <span className="font-mono text-slate-500">Shape: ({resolution}, {resolution})</span>
          </div>

          <div className="max-h-72 overflow-auto bg-slate-950 rounded-lg border border-slate-800 p-2 text-[10px] font-mono">
            <table className="w-full text-center border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-500 sticky top-0 bg-slate-950">
                  <th className="p-1 text-left">r\c</th>
                  {Array.from({ length: resolution }).map((_, c) => (
                    <th key={c} className="p-1 min-w-[24px]">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: resolution }).map((_, r) => (
                  <tr key={r} className="border-b border-slate-900 hover:bg-slate-900/60">
                    <td className="p-1 text-slate-500 font-semibold text-left">{r}</td>
                    {Array.from({ length: resolution }).map((_, c) => {
                      const val = rawBytes[r * resolution + c];
                      return (
                        <td
                          key={c}
                          className={`p-1 font-mono ${
                            val > 0
                              ? val > 128
                                ? 'text-amber-300 font-bold bg-amber-500/10'
                                : 'text-slate-200'
                              : 'text-slate-700'
                          }`}
                        >
                          {val}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 3: Flattened 1D Vector */}
      {activeTab === 'vector' && (
        <div id="tab-content-vector" className="pt-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="text-xs text-slate-300 flex items-center gap-2">
              <Info className="w-3.5 h-3.5 text-sky-400" />
              <span>
                Standard ML Input Shape: <b className="font-mono text-sky-300">(1, {resolution * resolution})</b> float32 in [0.0, 1.0]
              </span>
            </div>
            <button
              id="btn-copy-vector"
              type="button"
              onClick={handleCopyVector}
              className="flex items-center gap-1.5 py-1.5 px-3 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-200 hover:text-white transition"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  Copied NumPy Vector!
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  Copy NumPy Array
                </>
              )}
            </button>
          </div>

          <div className="relative bg-slate-950 rounded-lg border border-slate-800 p-3 max-h-56 overflow-auto">
            <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap leading-relaxed">
              {pythonVectorSnippet}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
