import React from 'react';
import { PreprocessOptions, TargetResolution } from '../types';
import {
  Code2,
  Cpu,
  Info,
  Layers,
  Palette,
  RotateCcw,
  Sliders,
  Sparkles,
} from 'lucide-react';

interface SidebarProps {
  strokeWidth: number;
  setStrokeWidth: (val: number) => void;
  options: PreprocessOptions;
  setOptions: React.Dispatch<React.SetStateAction<PreprocessOptions>>;
  onReset: () => void;
  onOpenPythonCode: () => void;
  isCollapsed: boolean;
  setIsCollapsed: (collapsed: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  strokeWidth,
  setStrokeWidth,
  options,
  setOptions,
  onReset,
  onOpenPythonCode,
  isCollapsed,
  setIsCollapsed,
}) => {
  return (
    <aside
      id="streamlit-sidebar"
      className={`bg-slate-900 border-r border-slate-800 transition-all duration-300 flex flex-col z-20 shrink-0 ${
        isCollapsed ? 'w-14' : 'w-72 sm:w-80'
      }`}
    >
      {/* Sidebar Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        {!isCollapsed && (
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-500">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-mono uppercase tracking-wider text-red-400 font-semibold block">
                Streamlit App
              </span>
              <h1 className="text-sm font-bold text-slate-100 leading-none">
                MNIST Canvas Settings
              </h1>
            </div>
          </div>
        )}
        <button
          id="btn-toggle-sidebar"
          type="button"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition mx-auto"
          title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          <Sliders className="w-4 h-4" />
        </button>
      </div>

      {!isCollapsed && (
        <div className="p-4 space-y-6 overflow-y-auto flex-1">
          {/* Section 1: Canvas Drawing Controls */}
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
              <span className="flex items-center gap-1.5">
                <Palette className="w-3.5 h-3.5 text-emerald-400" />
                Drawing Controls
              </span>
              <span className="font-mono text-emerald-400">{strokeWidth}px</span>
            </div>

            <div>
              <label htmlFor="stroke-width-slider" className="text-xs text-slate-400 block mb-1.5">
                Stroke Width
              </label>
              <input
                id="stroke-width-slider"
                type="range"
                min={8}
                max={32}
                step={2}
                value={strokeWidth}
                onChange={(e) => setStrokeWidth(Number(e.target.value))}
                className="w-full accent-emerald-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
                <span>Fine (8px)</span>
                <span>Default (18px)</span>
                <span>Thick (32px)</span>
              </div>
            </div>

            <div className="pt-1">
              <button
                id="btn-sidebar-clear"
                type="button"
                onClick={onReset}
                className="w-full flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
              >
                <RotateCcw className="w-3.5 h-3.5 text-red-400" />
                Clear / Reset Canvas
              </button>
            </div>
          </div>

          <div className="h-px bg-slate-800" />

          {/* Section 2: Image Processing Hyperparameters */}
          <div className="space-y-4">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-300">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              Pipeline Parameters
            </div>

            {/* Target Resolution Selector */}
            <div>
              <label className="text-xs text-slate-400 block mb-1.5">
                Target Resolution (Grid Size)
              </label>
              <div className="grid grid-cols-3 gap-1.5">
                {([14, 28, 56] as TargetResolution[]).map((res) => (
                  <button
                    key={res}
                    id={`btn-res-${res}`}
                    type="button"
                    onClick={() =>
                      setOptions((prev) => ({ ...prev, targetResolution: res }))
                    }
                    className={`py-1.5 px-2 rounded-lg text-xs font-mono font-medium border transition ${
                      options.targetResolution === res
                        ? 'bg-indigo-600 border-indigo-500 text-white shadow'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                    }`}
                  >
                    {res}&times;{res}
                    {res === 28 && (
                      <span className="block text-[9px] text-indigo-200 uppercase font-sans">
                        MNIST
                      </span>
                    )}
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-slate-500 mt-1">
                Standard MNIST benchmark is 28&times;28 (784 features).
              </p>
            </div>

            {/* Binarization Threshold */}
            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                <label htmlFor="threshold-slider">Binarization Threshold</label>
                <span className="font-mono text-indigo-400 font-medium">
                  {options.binarizationThreshold}
                </span>
              </div>
              <input
                id="threshold-slider"
                type="range"
                min={10}
                max={120}
                step={5}
                value={options.binarizationThreshold}
                onChange={(e) =>
                  setOptions((prev) => ({
                    ...prev,
                    binarizationThreshold: Number(e.target.value),
                  }))
                }
                className="w-full accent-indigo-500 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
                <span>Sensitive (10)</span>
                <span>Strict (120)</span>
              </div>
            </div>

            {/* Recenter Toggle */}
            <div className="pt-1">
              <label className="flex items-start gap-2.5 cursor-pointer bg-slate-950/80 p-2.5 rounded-lg border border-slate-800 hover:border-slate-700 transition">
                <input
                  id="checkbox-recenter"
                  type="checkbox"
                  checked={options.recenterEnabled}
                  onChange={(e) =>
                    setOptions((prev) => ({
                      ...prev,
                      recenterEnabled: e.target.checked,
                    }))
                  }
                  className="mt-0.5 accent-indigo-600 rounded cursor-pointer"
                />
                <div className="text-xs">
                  <span className="font-medium text-slate-200 block">
                    Center of Mass Recenter
                  </span>
                  <span className="text-[11px] text-slate-400 leading-tight block mt-0.5">
                    Aligns center of gravity with grid center (LeCun MNIST standard)
                  </span>
                </div>
              </label>
            </div>

            {/* Neural Network Layer Status */}
            <div className="bg-slate-950/90 p-3 rounded-lg border border-indigo-900/40 text-[11px] space-y-1.5">
              <div className="flex items-center justify-between text-slate-200 font-semibold">
                <span className="text-indigo-400 flex items-center gap-1">
                  <Cpu className="w-3 h-3" /> Classifier Topology
                </span>
                <span className="text-emerald-400 text-[10px] font-mono font-bold">16 &rarr; 14 &rarr; 10</span>
              </div>
              <div className="text-slate-400 text-[10px] space-y-0.5">
                <div>&bull; <b className="text-sky-300">Input:</b> 16 Spatial Sectors</div>
                <div>&bull; <b className="text-purple-300">Hidden:</b> 14 Feature Neurons</div>
                <div>&bull; <b className="text-emerald-300">Output:</b> Digits 0&ndash;9 (Softmax)</div>
                <div>&bull; <b className="text-amber-300">Lines:</b> Multi-Color Synaptic Spectrum</div>
              </div>
            </div>
          </div>

          <div className="h-px bg-slate-800" />

          {/* Python Streamlit Script Trigger */}
          <div className="space-y-2">
            <button
              id="btn-view-python-code"
              type="button"
              onClick={onOpenPythonCode}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-3 rounded-lg text-xs font-semibold bg-gradient-to-r from-red-600 to-amber-600 hover:from-red-500 hover:to-amber-500 text-white shadow-lg transition"
            >
              <Code2 className="w-4 h-4" />
              View Python Streamlit Code
            </button>
            <p className="text-[11px] text-slate-500 text-center">
              Exact <code className="text-slate-400">app.py</code> script for local execution.
            </p>
          </div>

          {/* Reference note */}
          <div className="bg-slate-950 p-3 rounded-lg border border-slate-800/80 text-[11px] text-slate-400 space-y-1">
            <div className="flex items-center gap-1 text-slate-300 font-medium">
              <Info className="w-3.5 h-3.5 text-sky-400 shrink-0" />
              <span>Assignment Spec</span>
            </div>
            <p className="text-slate-500 leading-relaxed">
              AI and DL using TF-09Aug(E) Page 9. Pipeline processes 280&times;280 white-on-black input through opencv area downsampling and center-of-mass shift.
            </p>
          </div>
        </div>
      )}
    </aside>
  );
};
