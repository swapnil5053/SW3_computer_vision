import { useState, useCallback } from 'react';
import { VideoFile, ProcessingResult, ProcessingStatus } from '../types';
import { APIService } from '../services/api';

export const useVideoProcessing = () => {
  const [status, setStatus] = useState<ProcessingStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ProcessingResult | null>(null);

  /**
   * Process the selected video using the chosen method/subMethod and optional arguments.
   * @param video - The uploaded video file
   * @param method - Main processing method (e.g., 'low-light', 'glare')
   * @param subMethod - Selected sub-method ID (e.g., 'clahe', 'glare-dim')
   * @param args - Optional extra arguments for backend script (e.g., ["--glare-thresh", "220"])
   */
  const processVideo = useCallback(
    async (video: VideoFile, method: string, subMethod?: string, args?: string[]) => {
      console.log('▶️ Starting video processing:', {
        method,
        subMethod,
        args,
        fileName: video.file.name,
      });

      setStatus('processing');
      setProgress(0);
      setError(null);

      try {
        // Simulate fake progress for UI feedback
        const progressInterval = setInterval(() => {
          setProgress((prev) => {
            if (prev >= 90) {
              clearInterval(progressInterval);
              return 90;
            }
            return prev + Math.random() * 10;
          });
        }, 1000);

        const startTime = Date.now();
        const result = await APIService.processVideo(video.file, method, subMethod, args);
        const endTime = Date.now();

        clearInterval(progressInterval);
        setProgress(100);

        // Set processing metadata
        result.processingTime = (endTime - startTime) / 1000;

        console.log('✅ Video processing completed:', result);
        setResult(result);
        setStatus('completed');
      } catch (err) {
        console.error('Video processing failed:', err);
        setStatus('error');
        setError(err instanceof Error ? err.message : 'Processing failed');
      }
    },
    []
  );

  const reset = useCallback(() => {
    setStatus('idle');
    setProgress(0);
    setError(null);
    setResult(null);
  }, []);

  return {
    status,
    progress,
    error,
    result,
    processVideo,
    reset,
  };
};
