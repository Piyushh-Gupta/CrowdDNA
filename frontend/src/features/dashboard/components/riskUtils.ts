import type { TimelineEntryDTO } from '../../../api/types';

export type RiskLevel = 'Safe' | 'Congesting' | 'Critical';

/**
 * Determine the peak risk classification for a frame.
 * Priority: Critical (Red) > Congesting (Amber) > Safe (Green).
 */
export function getFrameRisk(entry?: TimelineEntryDTO): RiskLevel {
  if (!entry || !entry.predictions || entry.predictions.length === 0) {
    return 'Safe';
  }
  const labels = entry.predictions.map((p) => p.label.toLowerCase());
  if (labels.some((l) => l.includes('critical'))) return 'Critical';
  if (labels.some((l) => l.includes('congesting'))) return 'Congesting';
  return 'Safe';
}
