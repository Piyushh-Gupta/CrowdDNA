import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { apiClient } from '../../../api/client';
import type { JobResponse } from '../../../api/types';
import './InferencePage.css';

/** Maximum allowed video upload size: 200 Megabytes */
export const MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024;

/** Permitted video file extensions */
export const ALLOWED_EXTENSIONS = ['.mp4', '.avi'] as const;

/**
 * Validates selected file for extension and size constraints.
 * Returns null if valid, or an error string if invalid.
 */
export function validateVideoFile(file: File): string | null {
  if (!file || !file.name) {
    return 'No file selected. Please select a video file.';
  }

  const nameParts = file.name.split('.');
  const extension = '.' + (nameParts.length > 1 ? nameParts.pop()?.toLowerCase() : '');
  const isAllowed = ALLOWED_EXTENSIONS.some((ext) => ext.toLowerCase() === extension);
  if (!isAllowed) {
    return `Unsupported format "${extension}". Only ${ALLOWED_EXTENSIONS.join(' and ')} video files are supported.`;
  }

  if (file.size === 0) {
    return 'The selected file is empty (0 bytes). Please select a valid video.';
  }

  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    return `File size (${sizeMb} MB) exceeds maximum allowed limit of 200 MB.`;
  }

  return null;
}

/** Formats byte counts into human-readable strings */
export function formatFileSize(bytes: number): string {
  if (bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

/** Formats ISO timestamp into readable localized string */
export function formatTimestamp(isoString?: string | null): string {
  if (!isoString) return '—';
  try {
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return isoString;
    return (
      date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) +
      ' ' +
      date.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' })
    );
  } catch {
    return isoString;
  }
}

export default function InferencePage(): React.ReactElement {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Upload & Form State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [enableCalibration, setEnableCalibration] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  // Job Submission & Execution State
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [currentJob, setCurrentJob] = useState<JobResponse | null>(null);
  const [isCancelling, setIsCancelling] = useState<boolean>(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<boolean>(false);

  // File Selection and Validation Handler
  const handleFile = useCallback((file: File) => {
    setSubmitError(null);
    setCancelError(null);
    const error = validateVideoFile(file);
    if (error) {
      setValidationError(error);
      setSelectedFile(null);
    } else {
      setValidationError(null);
      setSelectedFile(file);
    }
  }, []);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleDragEnter = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);

    const file = event.dataTransfer.files?.[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleClearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    setValidationError(null);
    setSubmitError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Submit Inference Job
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!selectedFile) {
      setValidationError('Please select a video file to upload.');
      return;
    }

    const error = validateVideoFile(selectedFile);
    if (error) {
      setValidationError(error);
      return;
    }

    setIsSubmitting(true);
    setSubmitError(null);
    setCancelError(null);

    // Abort any ongoing polling before starting new submission
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await apiClient.submitInference(selectedFile, enableCalibration);
      setCurrentJob(response);

      // Start live polling using apiClient.pollJob
      apiClient
        .pollJob(
          response.job_id,
          1000,
          60000,
          (updatedJob) => {
            setCurrentJob(updatedJob);
          },
          controller.signal
        )
        .then((terminalJob) => {
          if (!controller.signal.aborted) {
            setCurrentJob(terminalJob);
          }
        })
        .catch((err) => {
          if (!controller.signal.aborted) {
            console.warn('Polling finished with error or timeout:', err);
          }
        });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Inference job submission failed.';
      setSubmitError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Cancel Inference Job
  const handleCancelJob = async () => {
    if (!currentJob || isCancelling) return;

    setIsCancelling(true);
    setCancelError(null);

    try {
      const cancelled = await apiClient.cancelJob(currentJob.job_id);
      setCurrentJob(cancelled);
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to cancel inference job.';
      setCancelError(message);
    } finally {
      setIsCancelling(false);
    }
  };

  // Clean up polling controller on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Copy Job ID to Clipboard
  const handleCopyJobId = async () => {
    if (!currentJob?.job_id) return;
    try {
      await navigator.clipboard.writeText(currentJob.job_id);
      setCopiedId(true);
      setTimeout(() => setCopiedId(false), 2000);
    } catch {
      setCopiedId(false);
    }
  };

  const handleResetWorkflow = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setCurrentJob(null);
    setSelectedFile(null);
    setValidationError(null);
    setSubmitError(null);
    setCancelError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const isJobActive = currentJob?.status === 'queued' || currentJob?.status === 'running';

  return (
    <div
      className="inference-page"
      role="region"
      aria-label="Crowd Inference Engine"
      data-testid="inference-page"
    >
      <header className="inference-header">
        <h1>Crowd Inference & Risk Prediction</h1>
        <p>
          Upload video footage (.mp4, .avi) for asynchronous crowd density tracking, velocity estimation,
          and critical risk zone prediction.
        </p>
      </header>

      {/* Global Submission or Execution Error Banner */}
      {(submitError || (currentJob?.status === 'failed' && currentJob.error)) && (
        <div className="alert-banner error" role="alert" aria-live="assertive">
          <svg className="alert-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <div className="alert-content">
            <strong>Error: </strong>
            <span>{submitError || currentJob?.error || 'Inference execution failed.'}</span>
          </div>
        </div>
      )}

      {/* Cancellation Error Banner */}
      {cancelError && (
        <div className="alert-banner error" role="alert" aria-live="polite">
          <svg className="alert-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="alert-content">
            <strong>Cancellation Notice: </strong>
            <span>{cancelError}</span>
          </div>
        </div>
      )}

      {/* Success Banner when Inference is Completed */}
      {currentJob?.status === 'completed' && (
        <div className="alert-banner success" role="status" aria-live="polite" data-testid="job-state-completed">
          <svg className="alert-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="alert-content">
            <strong>Inference Completed Successfully!</strong>
            {currentJob.result?.output_video_path && (
              <div style={{ marginTop: '4px', fontSize: '13px' }}>
                Annotated video output: <code>{currentJob.result.output_video_path}</code>
              </div>
            )}
            <div style={{ marginTop: '4px', fontSize: '12px' }}>
              Generated risk telemetry timeline with {currentJob.result?.timeline?.length ?? 0} frames analyzed.
            </div>
            <div className="alert-actions">
              <button
                type="button"
                className="banner-action-btn"
                onClick={() => navigate('/')}
                aria-label="View Results in Dashboard"
                data-testid="view-in-dashboard-button"
              >
                View Results in Dashboard
              </button>
              <button
                type="button"
                className="banner-secondary-btn"
                onClick={handleResetWorkflow}
                aria-label="Submit another video"
              >
                Submit Another Video
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Upload and Configuration Form */}
      <form onSubmit={handleSubmit} className="inference-section-card" noValidate>
        <h2 className="section-title">Video Upload & Configuration</h2>

        {/* Drag and Drop Zone */}
        <div
          className={`dropzone ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          tabIndex={0}
          role="button"
          aria-label="Drag and drop video file zone, or press Enter to browse files"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".mp4,.avi,video/mp4,video/x-msvideo"
            onChange={handleInputChange}
            style={{ display: 'none' }}
            data-testid="video-file-input"
          />

          <svg className="dropzone-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>

          <p className="dropzone-text">
            Drag and drop your video file here, or <span className="dropzone-browse">browse</span>
          </p>
          <span className="dropzone-hint">
            Accepted formats: <strong>.mp4, .avi</strong> &bull; Maximum file size: <strong>200 MB</strong>
          </span>
        </div>

        {/* Validation Error Message */}
        {validationError && (
          <div
            className="alert-banner error"
            role="alert"
            aria-live="assertive"
            data-testid="file-validation-error"
          >
            <svg className="alert-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div className="alert-content">{validationError}</div>
          </div>
        )}

        {/* Selected File Details Box */}
        {selectedFile && !validationError && (
          <div className="selected-file-box" data-testid="selected-file-box">
            <div className="file-info">
              <svg className="file-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                />
              </svg>
              <div className="file-details">
                <span className="file-name" title={selectedFile.name}>
                  {selectedFile.name}
                </span>
                <span className="file-size">{formatFileSize(selectedFile.size)}</span>
              </div>
            </div>
            <button
              type="button"
              className="file-remove-btn"
              onClick={handleClearFile}
              aria-label="Remove selected file"
            >
              Remove
            </button>
          </div>
        )}

        {/* Metric Calibration Toggle Switch */}
        <div className="calibration-container">
          <label className="calibration-toggle-row" htmlFor="metric-calibration-switch">
            <div className="calibration-label-group">
              <span className="calibration-title" id="calibration-label">
                Enable Ground-Plane Metric Calibration
              </span>
              <span className="calibration-desc">
                Applies homographic ground-plane calibration to map 2D image coordinates to physical metric measurements (meters, m/s).
              </span>
            </div>
            <div className="switch">
              <input
                id="metric-calibration-switch"
                type="checkbox"
                role="switch"
                aria-checked={enableCalibration}
                aria-labelledby="calibration-label"
                checked={enableCalibration}
                onChange={(e) => setEnableCalibration(e.target.checked)}
                data-testid="calibration-toggle"
              />
              <span className="slider"></span>
            </div>
          </label>
        </div>

        {/* Action Row */}
        <div className="submit-action-row">
          <button
            type="submit"
            className="submit-btn"
            disabled={!selectedFile || Boolean(validationError) || isSubmitting || isJobActive}
            aria-busy={isSubmitting}
            data-testid="submit-inference-btn"
          >
            {isSubmitting ? (
              <>
                <span className="spinner" aria-hidden="true"></span>
                <span>Submitting Video...</span>
              </>
            ) : (
              <span>Submit Inference Job</span>
            )}
          </button>
        </div>
      </form>

      {/* Live Asynchronous Job Tracking Card */}
      {currentJob && (
        <section className="tracking-card" aria-label="Inference Job Tracking" data-testid="job-tracking-card">
          <div className="tracking-header">
            <div className="tracking-header-left">
              <h2 className="tracking-title">Inference Job Tracking</h2>
              <div className="job-id-row">
                <span>Job ID:</span>
                <span className="job-id-code" data-testid="job-id-display">{currentJob.job_id}</span>
                <button
                  type="button"
                  className="copy-btn"
                  onClick={handleCopyJobId}
                  aria-label="Copy Job ID"
                >
                  {copiedId ? 'Copied!' : 'Copy'}
                </button>
              </div>
            </div>

            {/* Status Indicator Badge */}
            <div
              className={`status-badge ${currentJob.status}`}
              role="status"
              aria-label={`Job status: ${currentJob.status}`}
              data-testid="job-status-badge"
            >
              {currentJob.status === 'running' && <span className="spinner" aria-hidden="true"></span>}
              {currentJob.status === 'queued' && <span className="status-dot animate-pulse" aria-hidden="true"></span>}
              {currentJob.status === 'completed' && (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              )}
              {currentJob.status === 'failed' && (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
              {currentJob.status === 'cancelled' && (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" />
                </svg>
              )}
              <span>{currentJob.status}</span>
            </div>
          </div>

          {/* Timestamps Grid */}
          <div className="timestamps-grid">
            <div className="timestamp-item">
              <span className="timestamp-label">Created At</span>
              <span className="timestamp-value" data-testid="timestamp-created">
                {formatTimestamp(currentJob.created_at)}
              </span>
            </div>
            <div className="timestamp-item">
              <span className="timestamp-label">Started At</span>
              <span className="timestamp-value" data-testid="timestamp-started">
                {currentJob.started_at ? formatTimestamp(currentJob.started_at) : 'Pending worker pickup'}
              </span>
            </div>
            <div className="timestamp-item">
              <span className="timestamp-label">Finished At</span>
              <span className="timestamp-value" data-testid="timestamp-finished">
                {currentJob.completed_at
                  ? formatTimestamp(currentJob.completed_at)
                  : isJobActive
                  ? 'Processing in background...'
                  : '—'}
              </span>
            </div>
          </div>

          {/* Tracking Footer / Actions */}
          <div className="tracking-footer">
            <div className="polling-indicator">
              {isJobActive && (
                <>
                  <span className="spinner" aria-hidden="true"></span>
                  <span>Live telemetry polling active (1.0s interval)...</span>
                </>
              )}
            </div>

            {/* Cancel Button */}
            {isJobActive && (
              <button
                type="button"
                className="cancel-job-btn"
                onClick={handleCancelJob}
                disabled={isCancelling}
                aria-busy={isCancelling}
                data-testid="cancel-job-btn"
              >
                {isCancelling ? 'Cancelling Job...' : 'Cancel Job'}
              </button>
            )}

            {currentJob.status === 'completed' && (
              <Link
                to="/"
                className="banner-action-btn"
                aria-label="Go to Dashboard"
              >
                Go to Dashboard &rarr;
              </Link>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
