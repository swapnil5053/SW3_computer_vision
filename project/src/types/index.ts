export interface ProcessingMethod {
  id: string;
  name: string;
  description: string;
  endpoint: string;
  color: string;
  subMethods?: SubMethod[];
}

export interface SubMethod {
  id: string;
  name: string;
  description: string;
  backendValue: string; // The value sent to backend
}

export interface VideoFile {
  file: File;
  url: string;
  name: string;
  size: number;
  duration?: number;
}

export interface ProcessingResult {
  outputUrl: string;
  processedVideoBlob: Blob;
  filename: string;
  processingTime: number;
  method: string;
  subMethod?: string;
}

export interface ProcessingLog {
  _id?: string;
  timestamp: string;
  process_category: string;
  model_used: string;
  video: {
    input_file: string;
    output_file: string;
    duration: number;
    resolution: string;
    fps: number;
    total_frames: number;
    processed_frames: number;
  };
  performance: {
    total_time: number;
    avg_delay_per_frame: number;
    cpu_usage_percent: number;
    gpu_usage_percent: number;
    device_used: string;
    device_specs: {
      cpu: string;
      gpu: string;
    };
  };
  status: string;
  error_message?: string;
}

export interface PerformanceData {
  frame: number;
  processingTime: number;
  timestamp: number;
}

export type ProcessingStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';