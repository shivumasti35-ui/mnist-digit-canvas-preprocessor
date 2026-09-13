/**
 * Preset digit stroke trajectories for quick testing and demonstration
 */

export interface PresetDigit {
  label: string;
  digit: number;
  description: string;
  strokes: Array<Array<[number, number]>>;
}

export const PRESET_DIGITS: PresetDigit[] = [
  {
    label: 'Digit 0',
    digit: 0,
    description: 'Smooth oval loop',
    strokes: [
      [
        [140, 65],
        [95, 100],
        [85, 155],
        [95, 210],
        [140, 245],
        [185, 210],
        [195, 155],
        [185, 100],
        [140, 65],
      ],
    ],
  },
  {
    label: 'Digit 1',
    digit: 1,
    description: 'Clean vertical stroke with top serif',
    strokes: [
      [
        [110, 95],
        [140, 65],
        [140, 240],
      ],
    ],
  },
  {
    label: 'Digit 2',
    digit: 2,
    description: 'Top arc, diagonal sweep, flat base',
    strokes: [
      [
        [95, 95],
        [125, 65],
        [175, 65],
        [190, 105],
        [165, 155],
        [90, 235],
        [195, 235],
      ],
    ],
  },
  {
    label: 'Digit 3',
    digit: 3,
    description: 'Classic curvy 3 with distinct center pinch',
    strokes: [
      [
        [80, 70],
        [140, 65],
        [190, 85],
        [195, 120],
        [170, 145],
        [140, 150],
        [175, 160],
        [195, 190],
        [185, 225],
        [135, 240],
        [80, 225],
      ],
    ],
  },
  {
    label: 'Digit 4',
    digit: 4,
    description: 'Angled descent, horizontal bar, vertical stem',
    strokes: [
      [
        [165, 65],
        [85, 175],
        [205, 175],
      ],
      [
        [165, 120],
        [165, 240],
      ],
    ],
  },
  {
    label: 'Digit 5',
    digit: 5,
    description: 'Top bar, vertical stem, rounded bottom bowl',
    strokes: [
      [
        [185, 68],
        [105, 68],
        [100, 140],
        [160, 135],
        [190, 165],
        [180, 215],
        [135, 240],
        [85, 225],
      ],
    ],
  },
  {
    label: 'Digit 6',
    digit: 6,
    description: 'Gentle curve into closed lower loop',
    strokes: [
      [
        [175, 70],
        [115, 115],
        [90, 175],
        [115, 235],
        [175, 235],
        [190, 185],
        [160, 150],
        [95, 165],
      ],
    ],
  },
  {
    label: 'Digit 7',
    digit: 7,
    description: 'Sharp horizontal bar with diagonal descent and crossbar',
    strokes: [
      [
        [75, 75],
        [205, 75],
        [180, 120],
        [145, 175],
        [120, 235],
      ],
      [
        [110, 145],
        [170, 145],
      ],
    ],
  },
  {
    label: 'Digit 8',
    digit: 8,
    description: 'Figure-eight continuous dual loop',
    strokes: [
      [
        [140, 65],
        [100, 85],
        [100, 125],
        [140, 150],
        [180, 180],
        [180, 225],
        [140, 245],
        [95, 225],
        [95, 180],
        [140, 150],
        [180, 125],
        [180, 85],
        [140, 65],
      ],
    ],
  },
  {
    label: 'Digit 9',
    digit: 9,
    description: 'Upper closed loop with sweeping tail',
    strokes: [
      [
        [185, 135],
        [150, 65],
        [100, 75],
        [85, 115],
        [115, 155],
        [175, 155],
        [185, 115],
        [180, 205],
        [135, 240],
      ],
    ],
  },
];

export function drawPresetOnCanvas(
  canvas: HTMLCanvasElement,
  preset: PresetDigit,
  strokeWidth: number = 18
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Clear with solid black background
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = strokeWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (const stroke of preset.strokes) {
    if (stroke.length < 2) continue;
    ctx.beginPath();
    ctx.moveTo(stroke[0][0], stroke[0][1]);

    for (let i = 1; i < stroke.length; i++) {
      const p0 = stroke[i - 1];
      const p1 = stroke[i];
      const midX = (p0[0] + p1[0]) / 2;
      const midY = (p0[1] + p1[1]) / 2;
      ctx.quadraticCurveTo(p0[0], p0[1], midX, midY);
    }
    const last = stroke[stroke.length - 1];
    ctx.lineTo(last[0], last[1]);
    ctx.stroke();
  }
}
