/**
 * Digit Segmentation and Multi-Digit Recognition Engine
 * Detects whether the user drew a single digit (0-9) or a multi-digit number (e.g., 2, 3, 4 digits),
 * segments each digit from left to right, processes through the MNIST pipeline,
 * and runs forward inference on each digit.
 */

import { BoundingBox, PreprocessOptions, Step5Result } from '../types';
import {
  processStep1,
  processStep2,
  processStep3,
  processStep4,
  processStep5,
} from './imageProcessing';
import { extractInputFeatures, runForwardPass } from './neuralNetworkEngine';

export interface RecognizedDigit {
  index: number;
  digit: number;
  confidence: number;
  probabilities: number[];
  boundingBox: BoundingBox;
  step5Result: Step5Result;
  inputFeatures: number[];
}

export interface MultiDigitResult {
  hasContent: boolean;
  fullNumber: string;
  digitCount: number;
  digits: RecognizedDigit[];
  primaryIndex: number;
}

/**
 * Segment and recognize all digits drawn on a canvas
 */
export function segmentAndRecognizeCanvas(
  canvas: HTMLCanvasElement,
  options: PreprocessOptions
): MultiDigitResult {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    return {
      hasContent: false,
      fullNumber: '',
      digitCount: 0,
      digits: [],
      primaryIndex: 0,
    };
  }

  const { width, height } = canvas;
  const imageData = ctx.getImageData(0, 0, width, height);
  const step1 = processStep1(imageData, options.binarizationThreshold);

  if (step1.nonZeroCount < 30) {
    return {
      hasContent: false,
      fullNumber: '',
      digitCount: 0,
      digits: [],
      primaryIndex: 0,
    };
  }

  // 1. Column projection to find horizontal digit segments
  const colInk = new Int32Array(width);
  for (let x = 0; x < width; x++) {
    let count = 0;
    for (let y = 0; y < height; y++) {
      if (step1.binary[y * width + x] > 0) count++;
    }
    colInk[x] = count;
  }

  // Gap threshold between adjacent digits in 280px canvas:
  // A gap of >= 5 pixels of zero ink indicates a separate digit
  const GAP_THRESHOLD = 5;
  const rawSegments: Array<{ minX: number; maxX: number }> = [];
  let inSegment = false;
  let startX = 0;
  let emptyRun = 0;

  for (let x = 0; x < width; x++) {
    if (colInk[x] > 0) {
      if (!inSegment) {
        inSegment = true;
        startX = x;
      }
      emptyRun = 0;
    } else {
      if (inSegment) {
        emptyRun++;
        if (emptyRun >= GAP_THRESHOLD || x === width - 1) {
          rawSegments.push({ minX: startX, maxX: x - emptyRun });
          inSegment = false;
          emptyRun = 0;
        }
      }
    }
  }
  if (inSegment) {
    rawSegments.push({ minX: startX, maxX: width - 1 });
  }

  // Filter out tiny noise segments (< 4px wide)
  const validSegments = rawSegments.filter(s => (s.maxX - s.minX + 1) >= 4);

  // If no valid segments or just 1 segment that spans the whole drawing, treat as 1 digit
  if (validSegments.length === 0) {
    return {
      hasContent: false,
      fullNumber: '',
      digitCount: 0,
      digits: [],
      primaryIndex: 0,
    };
  }

  const digits: RecognizedDigit[] = [];

  for (let segIdx = 0; segIdx < validSegments.length; segIdx++) {
    const { minX, maxX } = validSegments[segIdx];

    // Find vertical bounds (minY, maxY) within this segment
    let minY = height;
    let maxY = 0;
    let segPixelCount = 0;

    for (let y = 0; y < height; y++) {
      for (let x = minX; x <= maxX; x++) {
        if (step1.binary[y * width + x] > 0) {
          segPixelCount++;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (segPixelCount < 15 || maxY < minY) continue;

    // Crop sub-image of this digit
    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;
    const segImageData = ctx.getImageData(minX, minY, cropW, cropH);

    const segStep1 = processStep1(segImageData, options.binarizationThreshold);
    const segStep2 = processStep2(segStep1);
    if (!segStep2) continue;

    const segStep3 = processStep3(segStep2, options.squarePaddingFactor);
    const segStep4 = processStep4(segStep3, options.targetResolution);
    const segStep5 = processStep5(segStep4, options.recenterEnabled);

    const features = extractInputFeatures(segStep5.data, segStep5.resolution);
    const pred = runForwardPass(features);

    digits.push({
      index: digits.length,
      digit: pred.predictedDigit,
      confidence: pred.confidence,
      probabilities: pred.probabilities,
      boundingBox: {
        minX,
        minY,
        maxX,
        maxY,
        width: cropW,
        height: cropH,
      },
      step5Result: segStep5,
      inputFeatures: features,
    });
  }

  const fullNumber = digits.map(d => d.digit.toString()).join('');

  return {
    hasContent: digits.length > 0,
    fullNumber,
    digitCount: digits.length,
    digits,
    primaryIndex: 0,
  };
}
