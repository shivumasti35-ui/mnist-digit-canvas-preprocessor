/**
 * Image processing algorithms implementing the MNIST pipeline:
 * Step 1: Grayscale & Binarization
 * Step 2: Bounding Box & Bounding Crop
 * Step 3: Square Padding (Aspect-preserving)
 * Step 4: Area-based downscaling (cv2.INTER_AREA equivalent)
 * Step 5: Center of Mass calculation and recentering (LeCun MNIST standard)
 */

import {
  BoundingBox,
  CenterOfMass,
  PipelineResult,
  PreprocessOptions,
  Step1Result,
  Step2Result,
  Step3Result,
  Step4Result,
  Step5Result,
} from '../types';

/**
 * Step 1: Convert RGBA canvas imageData to single-channel Grayscale and Binarize
 * Background is strictly 0, foreground stroke is > 0 (or 255 for binary).
 */
export function processStep1(
  imageData: ImageData,
  threshold: number = 25
): Step1Result {
  const { width, height, data } = imageData;
  const totalPixels = width * height;
  const grayscale = new Uint8Array(totalPixels);
  const binary = new Uint8Array(totalPixels);
  let nonZeroCount = 0;

  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const a = data[i * 4 + 3] / 255;

    // ITU-R BT.601 standard luminance calculation
    const gray = Math.round((0.299 * r + 0.587 * g + 0.114 * b) * a);
    grayscale[i] = gray;

    if (gray >= threshold) {
      binary[i] = 255;
      nonZeroCount++;
    } else {
      binary[i] = 0;
    }
  }

  return { width, height, grayscale, binary, nonZeroCount };
}

/**
 * Step 2: Find bounding box around drawn strokes and crop tight
 */
export function processStep2(step1: Step1Result): Step2Result | null {
  const { width, height, binary, grayscale } = step1;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (binary[idx] > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // Check for empty canvas
  if (maxX === -1 || maxY === -1) {
    return null;
  }

  const croppedWidth = maxX - minX + 1;
  const croppedHeight = maxY - minY + 1;
  const croppedData = new Uint8Array(croppedWidth * croppedHeight);

  for (let y = 0; y < croppedHeight; y++) {
    for (let x = 0; x < croppedWidth; x++) {
      const srcIdx = (minY + y) * width + (minX + x);
      croppedData[y * croppedWidth + x] = grayscale[srcIdx];
    }
  }

  const box: BoundingBox = {
    minX,
    minY,
    maxX,
    maxY,
    width: croppedWidth,
    height: croppedHeight,
  };

  return {
    box,
    croppedWidth,
    croppedHeight,
    data: croppedData,
  };
}

/**
 * Step 3: Square Padding
 * Fits the cropped digit into a square aspect ratio while keeping the digit centered,
 * preserving original shape without distortion, and applying proportional border margin.
 */
export function processStep3(
  step2: Step2Result,
  paddingFactor: number = 1.28
): Step3Result {
  const { croppedWidth, croppedHeight, data } = step2;
  const maxDim = Math.max(croppedWidth, croppedHeight);
  // Pad with margin so strokes don't touch the outer boundary
  const squareSize = Math.max(maxDim, Math.round(maxDim * paddingFactor));

  const squareData = new Uint8Array(squareSize * squareSize);
  // Center the cropped rectangle inside the square
  const offsetX = Math.floor((squareSize - croppedWidth) / 2);
  const offsetY = Math.floor((squareSize - croppedHeight) / 2);

  for (let y = 0; y < croppedHeight; y++) {
    for (let x = 0; x < croppedWidth; x++) {
      const srcVal = data[y * croppedWidth + x];
      const targetIdx = (offsetY + y) * squareSize + (offsetX + x);
      squareData[targetIdx] = srcVal;
    }
  }

  return {
    size: squareSize,
    data: squareData,
    offsetX,
    offsetY,
  };
}

/**
 * Step 4: Downscale square image into target resolution (e.g., 28 x 28)
 * Implements area-based downsampling equivalent to OpenCV's `cv2.INTER_AREA`.
 * For each destination pixel, calculates the exact area-weighted integral
 * of all intersecting source pixels.
 */
export function processStep4(
  step3: Step3Result,
  targetResolution: 14 | 28 | 56
): Step4Result {
  const srcSize = step3.size;
  const srcData = step3.data;
  const dstSize = targetResolution;

  const scale = srcSize / dstSize; // source pixels per destination pixel
  const floatData = new Float32Array(dstSize * dstSize);
  const byteData = new Uint8Array(dstSize * dstSize);

  for (let dy = 0; dy < dstSize; dy++) {
    const syStart = dy * scale;
    const syEnd = (dy + 1) * scale;
    const syStartFloor = Math.floor(syStart);
    const syEndCeil = Math.min(srcSize, Math.ceil(syEnd));

    for (let dx = 0; dx < dstSize; dx++) {
      const sxStart = dx * scale;
      const sxEnd = (dx + 1) * scale;
      const sxStartFloor = Math.floor(sxStart);
      const sxEndCeil = Math.min(srcSize, Math.ceil(sxEnd));

      let weightedSum = 0;
      let totalWeight = 0;

      for (let sy = syStartFloor; sy < syEndCeil; sy++) {
        // Vertical overlap
        const yOverlap =
          Math.min(sy + 1, syEnd) - Math.max(sy, syStart);
        if (yOverlap <= 0) continue;

        for (let sx = sxStartFloor; sx < sxEndCeil; sx++) {
          // Horizontal overlap
          const xOverlap =
            Math.min(sx + 1, sxEnd) - Math.max(sx, sxStart);
          if (xOverlap <= 0) continue;

          const weight = xOverlap * yOverlap;
          const pixelVal = srcData[sy * srcSize + sx];

          weightedSum += pixelVal * weight;
          totalWeight += weight;
        }
      }

      const avgIntensity = totalWeight > 0 ? weightedSum / totalWeight : 0;
      const clampedByte = Math.max(0, Math.min(255, Math.round(avgIntensity)));
      const dstIdx = dy * dstSize + dx;

      byteData[dstIdx] = clampedByte;
      floatData[dstIdx] = clampedByte / 255;
    }
  }

  return {
    resolution: targetResolution,
    data: floatData,
    rawBytes: byteData,
  };
}

/**
 * Calculate Center of Mass (Image Moments: M10/M00, M01/M00)
 */
export function calculateCenterOfMass(
  data: Float32Array | Uint8Array,
  size: number
): CenterOfMass {
  let m00 = 0;
  let m10 = 0;
  let m01 = 0;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const val = data[y * size + x];
      m00 += val;
      m10 += x * val;
      m01 += y * val;
    }
  }

  if (m00 === 0) {
    const center = (size - 1) / 2;
    return { cx: center, cy: center, totalMass: 0 };
  }

  return {
    cx: m10 / m00,
    cy: m01 / m00,
    totalMass: m00,
  };
}

/**
 * Step 5: Center of Mass Adjustment (Recenter)
 * Translates the image so its Center of Mass aligns with the center of the grid,
 * matching standard MNIST normalization (LeCun et al. / scipy.ndimage.measurements.center_of_mass).
 * Uses bilinear interpolation with zero border padding for smooth subpixel recentering.
 */
export function processStep5(
  step4: Step4Result,
  recenterEnabled: boolean = true
): Step5Result {
  const size = step4.resolution;
  const initialData = step4.data;
  const initialCom = calculateCenterOfMass(initialData, size);

  const targetCx = (size - 1) / 2;
  const targetCy = (size - 1) / 2;

  if (!recenterEnabled || initialCom.totalMass === 0) {
    return {
      resolution: size,
      data: new Float32Array(initialData),
      rawBytes: new Uint8Array(step4.rawBytes),
      initialCom,
      targetCom: { cx: targetCx, cy: targetCy },
      shift: { dx: 0, dy: 0 },
      finalCom: initialCom,
    };
  }

  // Calculate shift amount needed to move COM to target center
  const shiftX = targetCx - initialCom.cx;
  const shiftY = targetCy - initialCom.cy;

  const shiftedFloat = new Float32Array(size * size);
  const shiftedBytes = new Uint8Array(size * size);

  // Sample shifted image with bilinear interpolation (equivalent to cv2.warpAffine / scipy.ndimage.shift)
  for (let dy = 0; dy < size; dy++) {
    for (let dx = 0; dx < size; dx++) {
      // Source coordinate we are sampling from
      const sx = dx - shiftX;
      const sy = dy - shiftY;

      let val = 0;

      if (sx >= 0 && sx < size && sy >= 0 && sy < size) {
        const x0 = Math.floor(sx);
        const y0 = Math.floor(sy);
        const x1 = Math.min(size - 1, x0 + 1);
        const y1 = Math.min(size - 1, y0 + 1);

        const fx = sx - x0;
        const fy = sy - y0;

        const p00 = initialData[y0 * size + x0];
        const p10 = initialData[y0 * size + x1];
        const p01 = initialData[y1 * size + x0];
        const p11 = initialData[y1 * size + x1];

        // Bilinear interpolation
        val =
          p00 * (1 - fx) * (1 - fy) +
          p10 * fx * (1 - fy) +
          p01 * (1 - fx) * fy +
          p11 * fx * fy;
      }

      val = Math.max(0, Math.min(1, val));
      const idx = dy * size + dx;
      shiftedFloat[idx] = val;
      shiftedBytes[idx] = Math.round(val * 255);
    }
  }

  const finalCom = calculateCenterOfMass(shiftedFloat, size);

  return {
    resolution: size,
    data: shiftedFloat,
    rawBytes: shiftedBytes,
    initialCom,
    targetCom: { cx: targetCx, cy: targetCy },
    shift: { dx: shiftX, dy: shiftY },
    finalCom,
  };
}

/**
 * Execute full MNIST Preprocessing Pipeline
 */
export function runMnistPipeline(
  canvas: HTMLCanvasElement,
  options: PreprocessOptions
): PipelineResult {
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return { hasContent: false, message: 'Could not access canvas context' };
  }

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const step1 = processStep1(imageData, options.binarizationThreshold);

  // Check if anything drawn
  if (step1.nonZeroCount < 10) {
    return {
      hasContent: false,
      message: 'Canvas is empty! Draw a digit (0–9) on the canvas to begin preprocessing.',
      step1,
    };
  }

  const step2 = processStep2(step1);
  if (!step2) {
    return {
      hasContent: false,
      message: 'No distinct digit contours detected. Try drawing with thicker strokes.',
      step1,
    };
  }

  const step3 = processStep3(step2, options.squarePaddingFactor);
  const step4 = processStep4(step3, options.targetResolution);
  const step5 = processStep5(step4, options.recenterEnabled);

  return {
    hasContent: true,
    step1,
    step2,
    step3,
    step4,
    step5,
  };
}
