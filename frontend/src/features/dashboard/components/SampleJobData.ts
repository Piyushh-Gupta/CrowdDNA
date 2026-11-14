import type { JobResponse } from '../../../api/types';

/**
 * Realistic sample inference job result for demonstration and offline visual testing.
 */
export const SAMPLE_JOB_RESULT: JobResponse = {
  job_id: 'job-demo-crowd-8f3a',
  status: 'completed',
  created_at: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
  started_at: new Date(Date.now() - 1000 * 60 * 14).toISOString(),
  completed_at: new Date(Date.now() - 1000 * 60 * 13).toISOString(),
  error: null,
  result: {
    output_video_path: '/var/crowddna/outputs/annotated_corridor_inference.mp4',
    metadata: {
      fps: 30.0,
      width: 1920,
      height: 1080,
      frame_count: 45,
      duration_seconds: 1.5,
      sample_rate: 1,
    },
    timeline: Array.from({ length: 45 }, (_, frameIdx) => {
      // Simulate evolving crowd dynamics:
      // Frames 0-14: Safe
      // Frames 15-28: Congesting in bottleneck regions
      // Frames 29-37: Critical congestion
      // Frames 38-44: Dissipating back to Congesting/Safe
      let reg1Risk: 'Safe' | 'Congesting' | 'Critical' = 'Safe';
      let reg1Conf = 0.94;
      let reg2Risk: 'Safe' | 'Congesting' | 'Critical' = 'Safe';
      let reg2Conf = 0.91;
      let reg3Risk: 'Safe' | 'Congesting' | 'Critical' = 'Safe';
      let reg3Conf = 0.88;

      if (frameIdx >= 15 && frameIdx < 28) {
        reg1Risk = 'Congesting';
        reg1Conf = 0.82 + (frameIdx % 5) * 0.02;
        reg2Risk = frameIdx > 20 ? 'Congesting' : 'Safe';
        reg2Conf = 0.85;
      } else if (frameIdx >= 29 && frameIdx < 38) {
        reg1Risk = 'Critical';
        reg1Conf = 0.96;
        reg2Risk = 'Critical';
        reg2Conf = 0.89;
        reg3Risk = 'Congesting';
        reg3Conf = 0.78;
      } else if (frameIdx >= 38) {
        reg1Risk = 'Congesting';
        reg1Conf = 0.79;
        reg2Risk = 'Safe';
        reg2Conf = 0.84;
      }

      return {
        frame_index: frameIdx,
        predictions: [
          { region_id: 1, label: reg1Risk, confidence: reg1Conf },
          { region_id: 2, label: reg2Risk, confidence: reg2Conf },
          { region_id: 3, label: reg3Risk, confidence: reg3Conf },
          { region_id: 4, label: 'Safe', confidence: 0.95 },
        ],
      };
    }),
  },
};
