import { ProcessingResult, ProcessingLog } from '../types';

const API_BASE_URL = 'http://localhost:8001';
const RAINDROP_API = "http://localhost:8000"

export class APIService {
  static async processVideo(
    file: File,
    method: string,
    subMethod?: string,
    args?: string[]
  ): Promise<ProcessingResult> {
  
    if (method === 'deraining' && subMethod === 'raindrop') {
      return await this.processRaindropVideo(file);
    }
    const formData = new FormData();
    formData.append('file', file);

    // Use subMethod if provided, otherwise use main method
    const processingMethod = subMethod || method;
    formData.append('method', processingMethod);

    // Correct: Send args as a JSON string
    if (args && args.length > 0) {
      formData.append('args', JSON.stringify(args));  // <-- KEY CHANGE HERE
    }

    console.log(' Sending request with:', {
      method: processingMethod,
      args,
    });

    const response = await fetch(`${API_BASE_URL}/api/process_video/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Processing failed: ${response.statusText} - ${errorText}`);
    }

    // Ensure the response is a video file
    const contentType = response.headers.get('content-type');
    console.log('📦 Response content type:', contentType);

    if (!contentType || !contentType.includes('video')) {
      const text = await response.text();
      console.error('Expected video but got:', text);
      throw new Error('Server did not return a video file');
    }

    const blob = await response.blob();
    console.log('✅ Received video blob of size:', blob.size, 'bytes');

    if (blob.size === 0) {
      throw new Error('Received empty video file');
    }

    const outputUrl = URL.createObjectURL(blob);
    console.log('🎥 Created video URL:', outputUrl);

    return {
      outputUrl,
      processedVideoBlob: blob,
      filename: `processed_${processingMethod}_${file.name}`,
      processingTime: 0,
      method,
      subMethod
    };
  }
static async getFrameTimings(sessionId: string): Promise<any> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/frame_timings/${sessionId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch frame timings: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching frame timings:', error);
      return null;
    }
  }
static async getMultiProcessTimings(sessionIds: string[]): Promise<any[]> {
    try {
      const promises = sessionIds.map(sessionId => this.getFrameTimings(sessionId));
      const results = await Promise.all(promises);
      return results.filter(result => result && !result.error);
    } catch (error) {
      console.error('Error fetching multi-process timings:', error);
      return [];
    }
  }
  // MongoDB logging API
  static async getProcessingLogs(): Promise<ProcessingLog[]> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/logs`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch logs: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching processing logs:', error);
      return [];
    }
  }

  static async getLatestLog(): Promise<ProcessingLog | null> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/logs/latest`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        throw new Error(`Failed to fetch latest log: ${response.statusText}`);
      }
      const data = await response.json();
      return data.error ? null : data;
    } catch (error) {
      console.error('Error fetching latest log:', error);
      return null;
    }
}

  static async processRaindropVideo(
    file: File,
  ): Promise<ProcessingResult> {
    try {
      console.log('Starting ARDCNN pipeline...');
      
      // Generate a unique output filename
      const timestamp = Date.now();
      const outputVideoName = `cleaned_${timestamp}_${file.name}`;
      
      // Call the new pipeline endpoint that does ARDCNN + LaMa + Video stitching
      const pipelineFormData = new FormData();
      pipelineFormData.append('input_data', file);
      pipelineFormData.append('start_time', '0');
      pipelineFormData.append('end_time', '30');
      pipelineFormData.append('frame_rate', '1');
      pipelineFormData.append('lama_service_url', 'http://lama:8002');
      pipelineFormData.append('output_video_name', outputVideoName);

      const pipelineResponse = await fetch(`${RAINDROP_API}/detect_and_pipeline`, {
        method: 'POST',
        body: pipelineFormData,
      });

      if (!pipelineResponse.ok) {
        const errorText = await pipelineResponse.text();
        throw new Error(`Pipeline processing failed: ${pipelineResponse.statusText} - ${errorText}`);
      }

      const pipelineResult = await pipelineResponse.json();
      console.log('Pipeline completed:', pipelineResult);

      // Check if processing was successful
      if (pipelineResult.status !== 'success') {
        throw new Error(`Pipeline failed: ${pipelineResult.error || 'Unknown error'}`);
      }

      // Use the filename from the response or fallback to what we sent
      const videoFilename = pipelineResult.output_video || outputVideoName;
      const directVideoUrl = `${RAINDROP_API}/download/${videoFilename}`;
      
      console.log('Using direct video URL:', directVideoUrl);
      console.log('Video filename:', videoFilename);

      // Download the video file and create blob URL for playback
      // This solves the infinite buffering issue with direct URLs
      console.log('Downloading video for blob conversion...');
      const downloadResponse = await fetch(directVideoUrl);
      
      if (!downloadResponse.ok) {
        throw new Error(`Failed to download video: ${downloadResponse.statusText}`);
      }
      
      const videoBlob = await downloadResponse.blob();
      console.log('Downloaded video blob size:', videoBlob.size, 'bytes');
      
      if (videoBlob.size === 0) {
        throw new Error('Downloaded video is empty');
      }
      
      // Create blob URL for reliable playback
      const blobUrl = URL.createObjectURL(videoBlob);
      console.log('Created blob URL for playback:', blobUrl);

      // Add a small delay to ensure blob URL is fully ready
      await new Promise(resolve => setTimeout(resolve, 1000));
      console.log('Blob URL stabilization delay completed');

      return {
        outputUrl: blobUrl,
        processedVideoBlob: videoBlob,
        filename: `cleaned_${file.name}`,
        processingTime: 0,
        method: 'deraining',
        subMethod: 'raindrop'
      };
    } catch (error) {
      console.error('Pipeline processing error:', error);
      throw error;
    }
  }
}