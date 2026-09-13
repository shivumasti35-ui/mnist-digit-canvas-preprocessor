import * as fs from 'fs';
import { PRESET_DIGITS } from './src/utils/sampleDigits';
import { processStep1, processStep2, processStep3, processStep4, processStep5 } from './src/utils/imageProcessing';

export function extractSquareFeatures(data: Float32Array, size: number = 28): number[] {
  let minX = size, maxX = 0, minY = size, maxY = 0;
  let total = 0;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const v = data[y * size + x];
      if (v > 0.08) {
        total += v;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (total < 0.1 || maxX < minX || maxY < minY) return new Array(16).fill(0);

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const maxSpan = Math.max(maxX - minX + 1, maxY - minY + 1);
  const half = (maxSpan * 1.1) / 2;

  const x0 = cx - half;
  const y0 = cy - half;
  const step = (half * 2) / 4;

  const feat = new Array(16).fill(0);
  for (let gy = 0; gy < 4; gy++) {
    for (let gx = 0; gx < 4; gx++) {
      const sx0 = Math.max(0, Math.floor(x0 + gx * step));
      const sx1 = Math.min(size, Math.ceil(x0 + (gx + 1) * step));
      const sy0 = Math.max(0, Math.floor(y0 + gy * step));
      const sy1 = Math.min(size, Math.ceil(y0 + (gy + 1) * step));

      let sum = 0, cnt = 0;
      for (let y = sy0; y < sy1; y++) {
        for (let x = sx0; x < sx1; x++) {
          sum += data[y * size + x];
          cnt++;
        }
      }
      feat[gy * 4 + gx] = cnt > 0 ? sum / cnt : 0;
    }
  }
  let max = 0;
  for (let i = 0; i < 16; i++) if (feat[i] > max) max = feat[i];
  if (max > 0.01) {
    for (let i = 0; i < 16; i++) feat[i] = Math.min(1.0, feat[i] / max);
  }
  return feat;
}

interface StrokeSample {
  digit: number;
  strokes: Array<Array<[number, number]>>;
}

const BASE_SAMPLES: StrokeSample[] = [
  // 0: Oval
  {
    digit: 0,
    strokes: [[
      [140, 65], [105, 95], [90, 145], [105, 205], [140, 235], [175, 205], [190, 145], [175, 95], [140, 65]
    ]]
  },
  // 0: Circular
  {
    digit: 0,
    strokes: [[
      [140, 75], [95, 105], [80, 150], [95, 195], [140, 225], [185, 195], [200, 150], [185, 105], [140, 75]
    ]]
  },
  // 0: Narrow
  {
    digit: 0,
    strokes: [[
      [140, 60], [115, 95], [110, 150], [115, 205], [140, 240], [165, 205], [170, 150], [165, 95], [140, 60]
    ]]
  },

  // 1: Straight line
  {
    digit: 1,
    strokes: [[[140, 60], [140, 240]]]
  },
  // 1: Slanted line
  {
    digit: 1,
    strokes: [[[130, 60], [145, 240]]]
  },
  // 1: With top serif/flag
  {
    digit: 1,
    strokes: [[[110, 95], [140, 65], [140, 240]]]
  },
  // 1: With bottom base bar
  {
    digit: 1,
    strokes: [
      [[115, 90], [140, 65], [140, 235]],
      [[105, 235], [175, 235]]
    ]
  },

  // 2: Standard rounded top, diagonal, flat base
  {
    digit: 2,
    strokes: [[[95, 95], [125, 65], [175, 65], [190, 105], [165, 155], [90, 235], [195, 235]]]
  },
  // 2: Loop 2
  {
    digit: 2,
    strokes: [[[100, 95], [130, 65], [175, 65], [185, 105], [160, 155], [90, 230], [95, 235], [195, 235]]]
  },

  // 3: Double curve
  {
    digit: 3,
    strokes: [
      [[95, 75], [135, 65], [180, 80], [175, 125], [140, 145]],
      [[140, 145], [185, 165], [180, 220], [135, 235], [95, 225]]
    ]
  },
  // 3: Continuous stroke
  {
    digit: 3,
    strokes: [[[95, 80], [140, 65], [180, 85], [170, 130], [135, 145], [175, 160], [180, 215], [135, 235], [90, 220]]]
  },

  // 4: Standard 2 strokes
  {
    digit: 4,
    strokes: [
      [[125, 65], [95, 165], [185, 165]],
      [[160, 60], [160, 240]]
    ]
  },
  // 4: Single stroke or open top
  {
    digit: 4,
    strokes: [
      [[115, 65], [85, 165], [195, 165]],
      [[155, 65], [155, 235]]
    ]
  },
  // 4: Triangle top
  {
    digit: 4,
    strokes: [
      [[150, 65], [90, 165], [185, 165]],
      [[150, 65], [150, 240]]
    ]
  },

  // 5: Standard
  {
    digit: 5,
    strokes: [
      [[175, 65], [115, 65], [110, 135]],
      [[110, 135], [150, 125], [185, 150], [185, 205], [145, 235], [95, 225]]
    ]
  },
  // 5: Continuous
  {
    digit: 5,
    strokes: [
      [[180, 70], [115, 70], [105, 135], [145, 125], [185, 150], [185, 200], [145, 235], [95, 225]]
    ]
  },

  // 6: Standard loop
  {
    digit: 6,
    strokes: [
      [[170, 75], [130, 95], [100, 145], [100, 195], [135, 235], [175, 215], [185, 175], [160, 140], [115, 145], [100, 185]]
    ]
  },
  // 6: Straight stem into loop
  {
    digit: 6,
    strokes: [
      [[155, 65], [110, 145], [105, 195], [135, 235], [175, 215], [185, 170], [155, 140], [110, 145]]
    ]
  },
  // 6: Quick curve loop
  {
    digit: 6,
    strokes: [
      [[160, 70], [115, 120], [105, 180], [135, 230], [175, 210], [180, 165], [145, 140], [110, 165]]
    ]
  },

  // 7: Standard flat top diagonal
  {
    digit: 7,
    strokes: [[[90, 70], [190, 70], [125, 240]]]
  },
  // 7: With crossbar
  {
    digit: 7,
    strokes: [
      [[90, 70], [190, 70], [125, 240]],
      [[130, 150], [170, 150]]
    ]
  },
  // 7: Slanted top
  {
    digit: 7,
    strokes: [[[90, 80], [185, 65], [130, 240]]]
  },

  // 8: Figure eight
  {
    digit: 8,
    strokes: [
      [[140, 65], [110, 90], [125, 135], [170, 180], [175, 220], [140, 240], [105, 220], [110, 180], [155, 135], [170, 90], [140, 65]]
    ]
  },
  // 8: Two stacked circles
  {
    digit: 8,
    strokes: [
      [[140, 65], [115, 85], [115, 125], [140, 145], [165, 125], [165, 85], [140, 65]],
      [[140, 145], [105, 170], [105, 215], [140, 240], [175, 215], [175, 170], [140, 145]]
    ]
  },
  // 8: Hand-drawn smooth two-loop
  {
    digit: 8,
    strokes: [
      [[140, 70], [115, 95], [140, 140], [105, 185], [140, 235], [175, 185], [140, 140], [165, 95], [140, 70]]
    ]
  },
  // 8: Stacked circles with slight waist
  {
    digit: 8,
    strokes: [
      [[140, 75], [118, 95], [118, 130], [140, 145], [162, 130], [162, 95], [140, 75]],
      [[140, 145], [110, 165], [110, 210], [140, 235], [170, 210], [170, 165], [140, 145]]
    ]
  },

  // 9: Top loop and right stem
  {
    digit: 9,
    strokes: [
      [[140, 65], [105, 90], [115, 140], [160, 145], [175, 110], [160, 65], [140, 65], [175, 110], [170, 240]]
    ]
  },
  // 9: Curved bottom stem
  {
    digit: 9,
    strokes: [
      [[140, 65], [105, 90], [115, 140], [160, 145], [175, 110], [160, 65], [140, 65], [175, 110], [170, 215], [140, 240], [115, 235]]
    ]
  }
];

// Data augmentation: stroke thickness, scaling, rotation, translation
function generateAugmentedFeatures() {
  const samples: Array<{ digit: number; feat: number[] }> = [];
  const canvasSize = 280;

  for (const base of BASE_SAMPLES) {
    const strokeRadii = [6, 8, 10, 13];
    const scales = [0.85, 1.0, 1.15];
    const dxs = [-15, 0, 15];
    const dys = [-15, 0, 15];
    const rotations = [-0.12, 0, 0.12];

    for (const r of strokeRadii) {
      for (const scale of scales) {
        for (const rot of rotations) {
          for (let i = 0; i < 4; i++) {
            const dx = dxs[Math.floor(Math.random() * dxs.length)];
            const dy = dys[Math.floor(Math.random() * dys.length)];

            const imgData = new Uint8ClampedArray(canvasSize * canvasSize * 4);
            const cos = Math.cos(rot);
            const sin = Math.sin(rot);

            for (const stroke of base.strokes) {
              for (let pt = 0; pt < stroke.length - 1; pt++) {
                const [ox0, oy0] = stroke[pt];
                const [ox1, oy1] = stroke[pt + 1];

                const rx0 = (ox0 - 140) * scale;
                const ry0 = (oy0 - 140) * scale;
                const x0 = 140 + rx0 * cos - ry0 * sin + dx;
                const y0 = 140 + rx0 * sin + ry0 * cos + dy;

                const rx1 = (ox1 - 140) * scale;
                const ry1 = (oy1 - 140) * scale;
                const x1 = 140 + rx1 * cos - ry1 * sin + dx;
                const y1 = 140 + rx1 * sin + ry1 * cos + dy;

                const dist = Math.hypot(x1 - x0, y1 - y0);
                const steps = Math.max(2, Math.ceil(dist * 2));
                for (let s = 0; s <= steps; s++) {
                  const cx = Math.round(x0 + (x1 - x0) * (s / steps));
                  const cy = Math.round(y0 + (y1 - y0) * (s / steps));
                  for (let dpy = -r; dpy <= r; dpy++) {
                    for (let dpx = -r; dpx <= r; dpx++) {
                      if (dpx * dpx + dpy * dpy <= r * r) {
                        const px = cx + dpx;
                        const py = cy + dpy;
                        if (px >= 0 && px < canvasSize && py >= 0 && py < canvasSize) {
                          const idx = (py * canvasSize + px) * 4;
                          imgData[idx] = 255;
                          imgData[idx + 1] = 255;
                          imgData[idx + 2] = 255;
                          imgData[idx + 3] = 255;
                        }
                      }
                    }
                  }
                }
              }
            }

            const s1 = processStep1({ width: canvasSize, height: canvasSize, data: imgData } as any);
            const s2 = processStep2(s1);
            if (!s2) continue;
            const s3 = processStep3(s2, 1.28);
            const s4 = processStep4(s3, 28);
            const s5 = processStep5(s4, true);

            const feat = extractSquareFeatures(s5.data, 28);
            samples.push({ digit: base.digit, feat });
          }
        }
      }
    }
  }

  return samples;
}

console.log("Generating training dataset...");
const dataset = generateAugmentedFeatures();
console.log(`Generated ${dataset.length} samples.`);

// Train 16 -> 14 -> 10 MLP
const D_IN = 16;
const D_HID = 14;
const D_OUT = 10;

// Initialize weights with He/Glorot initialization
const W1: number[][] = Array.from({ length: D_HID }, () =>
  Array.from({ length: D_IN }, () => (Math.random() - 0.5) * Math.sqrt(2 / D_IN))
);
const B1: number[] = new Array(D_HID).fill(0.1);

const W2: number[][] = Array.from({ length: D_OUT }, () =>
  Array.from({ length: D_HID }, () => (Math.random() - 0.5) * Math.sqrt(2 / D_HID))
);
const B2: number[] = new Array(D_OUT).fill(0.0);

// Adam optimizer states
const mW1 = Array.from({ length: D_HID }, () => new Array(D_IN).fill(0));
const vW1 = Array.from({ length: D_HID }, () => new Array(D_IN).fill(0));
const mB1 = new Array(D_HID).fill(0);
const vB1 = new Array(D_HID).fill(0);

const mW2 = Array.from({ length: D_OUT }, () => new Array(D_HID).fill(0));
const vW2 = Array.from({ length: D_OUT }, () => new Array(D_HID).fill(0));
const mB2 = new Array(D_OUT).fill(0);
const vB2 = new Array(D_OUT).fill(0);

const lr = 0.008;
const beta1 = 0.9;
const beta2 = 0.999;
const eps = 1e-8;
const epochs = 600;

for (let ep = 0; ep < epochs; ep++) {
  // Shuffle dataset
  for (let i = dataset.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [dataset[i], dataset[j]] = [dataset[j], dataset[i]];
  }

  let totalLoss = 0;
  let correct = 0;

  const gW1 = Array.from({ length: D_HID }, () => new Array(D_IN).fill(0));
  const gB1 = new Array(D_HID).fill(0);
  const gW2 = Array.from({ length: D_OUT }, () => new Array(D_HID).fill(0));
  const gB2 = new Array(D_OUT).fill(0);

  const batchSize = dataset.length;

  for (const { digit, feat } of dataset) {
    // Forward pass
    const z1 = new Array(D_HID);
    const a1 = new Array(D_HID);
    for (let h = 0; h < D_HID; h++) {
      let sum = B1[h];
      for (let i = 0; i < D_IN; i++) sum += W1[h][i] * feat[i];
      z1[h] = sum;
      a1[h] = Math.max(0, sum); // ReLU
    }

    const z2 = new Array(D_OUT);
    for (let o = 0; o < D_OUT; o++) {
      let sum = B2[o];
      for (let h = 0; h < D_HID; h++) sum += W2[o][h] * a1[h];
      z2[o] = sum;
    }

    // Softmax
    const maxZ = Math.max(...z2);
    const exps = z2.map(z => Math.exp(z - maxZ));
    const sumExp = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumExp);

    totalLoss += -Math.log(Math.max(1e-15, probs[digit]));

    let pred = 0;
    for (let o = 1; o < D_OUT; o++) {
      if (probs[o] > probs[pred]) pred = o;
    }
    if (pred === digit) correct++;

    // Backward pass
    const dZ2 = new Array(D_OUT);
    for (let o = 0; o < D_OUT; o++) {
      dZ2[o] = probs[o] - (o === digit ? 1 : 0);
      gB2[o] += dZ2[o];
      for (let h = 0; h < D_HID; h++) {
        gW2[o][h] += dZ2[o] * a1[h];
      }
    }

    const dZ1 = new Array(D_HID);
    for (let h = 0; h < D_HID; h++) {
      let sum = 0;
      for (let o = 0; o < D_OUT; o++) sum += dZ2[o] * W2[o][h];
      dZ1[h] = z1[h] > 0 ? sum : 0; // ReLU derivative
      gB1[h] += dZ1[h];
      for (let i = 0; i < D_IN; i++) {
        gW1[h][i] += dZ1[h] * feat[i];
      }
    }
  }

  // Weight updates with Adam + L2 regularization
  const t = ep + 1;
  const l2 = 0.0001;

  for (let o = 0; o < D_OUT; o++) {
    const gb = gB2[o] / batchSize;
    mB2[o] = beta1 * mB2[o] + (1 - beta1) * gb;
    vB2[o] = beta2 * vB2[o] + (1 - beta2) * gb * gb;
    const mHat = mB2[o] / (1 - Math.pow(beta1, t));
    const vHat = vB2[o] / (1 - Math.pow(beta2, t));
    B2[o] -= (lr * mHat) / (Math.sqrt(vHat) + eps);

    for (let h = 0; h < D_HID; h++) {
      const gw = gW2[o][h] / batchSize + l2 * W2[o][h];
      mW2[o][h] = beta1 * mW2[o][h] + (1 - beta1) * gw;
      vW2[o][h] = beta2 * vW2[o][h] + (1 - beta2) * gw * gw;
      const mHatW = mW2[o][h] / (1 - Math.pow(beta1, t));
      const vHatW = vW2[o][h] / (1 - Math.pow(beta2, t));
      W2[o][h] -= (lr * mHatW) / (Math.sqrt(vHatW) + eps);
    }
  }

  for (let h = 0; h < D_HID; h++) {
    const gb = gB1[h] / batchSize;
    mB1[h] = beta1 * mB1[h] + (1 - beta1) * gb;
    vB1[h] = beta2 * vB1[h] + (1 - beta2) * gb * gb;
    const mHat = mB1[h] / (1 - Math.pow(beta1, t));
    const vHat = vB1[h] / (1 - Math.pow(beta2, t));
    B1[h] -= (lr * mHat) / (Math.sqrt(vHat) + eps);

    for (let i = 0; i < D_IN; i++) {
      const gw = gW1[h][i] / batchSize + l2 * W1[h][i];
      mW1[h][i] = beta1 * mW1[h][i] + (1 - beta1) * gw;
      vW1[h][i] = beta2 * vW1[h][i] + (1 - beta2) * gw * gw;
      const mHatW = mW1[h][i] / (1 - Math.pow(beta1, t));
      const vHatW = vW1[h][i] / (1 - Math.pow(beta2, t));
      W1[h][i] -= (lr * mHatW) / (Math.sqrt(vHatW) + eps);
    }
  }

  if (ep % 100 === 0 || ep === epochs - 1) {
    const acc = ((correct / batchSize) * 100).toFixed(1);
    const loss = (totalLoss / batchSize).toFixed(4);
    console.log(`Epoch ${ep}: Loss ${loss}, Accuracy ${acc}%`);
  }
}

// Round weights to 4 decimals
const round4 = (x: number) => Math.round(x * 10000) / 10000;
const finalW1 = W1.map(row => row.map(round4));
const finalB1 = B1.map(round4);
const finalW2 = W2.map(row => row.map(round4));
const finalB2 = B2.map(round4);

fs.writeFileSync(
  './trained_square_weights.json',
  JSON.stringify({ W1: finalW1, B1: finalB1, W2: finalW2, B2: finalB2 }, null, 2)
);
console.log('Saved trained_square_weights.json');
