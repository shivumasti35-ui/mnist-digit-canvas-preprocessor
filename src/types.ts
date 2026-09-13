/**
 * Type definitions for MNIST Digit Preprocessor pipeline
 */

export type TargetResolution = 14 | 28 | 56;

export interface PreprocessOptions {
  targetResolution: TargetResolution;
  binarizationThreshold: number; // 0 - 255
  squarePaddingFactor: number;   // e.g. 1.25 gives ~20% border margin
  recenterEnabled: boolean;
}

export interface BoundingBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
}

export interface CenterOfMass {
  cx: number;
  cy: number;
  totalMass: number;
}

export interface Step1Result {
  width: number;
  height: number;
  grayscale: Uint8Array; // 280 x 280 (0-255)
  binary: Uint8Array;    // 280 x 280 (0 or 255)
  nonZeroCount: number;
}

export interface Step2Result {
  box: BoundingBox;
  croppedWidth: number;
  croppedHeight: number;
  data: Uint8Array; // tight cropped
}

export interface Step3Result {
  size: number;
  data: Uint8Array; // size x size square
  offsetX: number;
  offsetY: number;
}

export interface Step4Result {
  resolution: TargetResolution;
  data: Float32Array; // targetResolution x targetResolution (0.0 to 1.0)
  rawBytes: Uint8Array; // (0 to 255)
}

export interface Step5Result {
  resolution: TargetResolution;
  data: Float32Array; // targetResolution x targetResolution (0.0 to 1.0)
  rawBytes: Uint8Array; // (0 to 255)
  initialCom: CenterOfMass;
  targetCom: { cx: number; cy: number };
  shift: { dx: number; dy: number };
  finalCom: CenterOfMass;
}

export interface PipelineResult {
  hasContent: boolean;
  message?: string;
  step1?: Step1Result;
  step2?: Step2Result;
  step3?: Step3Result;
  step4?: Step4Result;
  step5?: Step5Result;
}

export type ConnectionColorMode = 'spectrum' | 'polarity' | 'layer';

export interface NetworkNode {
  id: string;
  layer: 'input' | 'hidden' | 'output';
  index: number;
  label: string;
  activation: number; // 0.0 to 1.0
  x: number;
  y: number;
}

export interface NetworkConnection {
  id: string;
  fromId: string;
  toId: string;
  fromX: number;
  fromY: number;
  toX: number;
  toY: number;
  weight: number;
  activation: number;
  color: string;
  width: number;
  opacity: number;
}

export interface NetworkPrediction {
  probabilities: number[];
  predictedDigit: number;
  confidence: number;
  hiddenActivations: number[];
}
