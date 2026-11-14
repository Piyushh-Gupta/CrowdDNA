/**
 * CrowdDNA API Data Transfer Objects and Schema Contracts.
 * Mirrors FastAPI backend schemas from crowdflow_dna/api/schemas/.
 */

export type JobStatusEnum = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface RiskPredictionDTO {
  region_id: number;
  label: 'Safe' | 'Congesting' | 'Critical' | string;
  confidence: number;
}

export interface TimelineEntryDTO {
  frame_index: number;
  predictions: RiskPredictionDTO[];
}

export interface VideoMetadataDTO {
  fps: number;
  width: number;
  height: number;
  frame_count: number;
  duration_seconds: number;
  sample_rate: number;
}

export interface InferenceResultDTO {
  output_video_path?: string | null;
  timeline: TimelineEntryDTO[];
  metadata?: VideoMetadataDTO | null;
}

export interface JobResponse {
  job_id: string;
  status: JobStatusEnum;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  result?: InferenceResultDTO | null;
}

export interface HealthStatus {
  status: string;
  timestamp: string;
  version: string;
}

export interface LiveStatus {
  status: string;
}

export interface ReadyStatus {
  status: string;
  ready?: boolean;
  model_loaded: boolean;
  device: string;
}

export interface VersionInfo {
  api_version: string;
  engine_version: string;
}

export interface ErrorDetail {
  loc?: (string | number)[];
  msg: string;
  type?: string;
}

export interface ErrorResponse {
  detail: string | ErrorDetail[];
  error: string;
  status_code: number;
}
