import React, { useRef, useEffect, useState } from 'react';
import { RefreshCw, Layers } from 'lucide-react';
import { APIService } from '../services/api';

interface MultiProcessGraphProps {
  loading: boolean;
  latestVideoInfo?: any;
}

interface ProcessData {
  method: string;
  color: string;
  frameData: Array<{
    frame: number;
    processingTime: number;
    wasEnhanced: boolean;
  }>;
  avgTime: number;
  enhancementRate: number;
}

export const MultiProcessGraph: React.FC<MultiProcessGraphProps> = ({ loading, latestVideoInfo }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [processesData, setProcessesData] = useState<ProcessData[]>([]);
  const [dataLoading, setDataLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Method colors
  const methodColors: { [key: string]: string } = {
    'clahe_with_classifier': '#14B8A6',
    'unet_with_classifier': '#3B82F6',
    'glare_reduction': '#F59E0B',
    'deraining': '#8B5CF6',
    'dehazing': '#10B981',
    'tilt_correction': '#EF4444'
  };

  useEffect(() => {
    if (latestVideoInfo) {
      loadMultiProcessData();
    }
  }, [latestVideoInfo]);

  const loadMultiProcessData = async () => {
    setDataLoading(true);
    setError(null);
    
    try {
      // Get the base filename from the latest processed video
      const originalInputFile = latestVideoInfo.input_file?.split('/').pop() || 'video.mp4';
      const baseFilename = originalInputFile.substring(0, originalInputFile.lastIndexOf('.')) || originalInputFile;
      console.log('Base filename for multi-process data:', baseFilename);
      // Try to load timing data for different methods
      // These would be the actual JSON files from your processing runs
      const sessionIds = [
        `processed_unet_${baseFilename}`,
        `processed_clahe_${baseFilename}`,
        `processed_glare-dim_${baseFilename}`,
        `processed_tilt_${baseFilename}`,
        `processed_flare-reduction_${baseFilename}`,
        `processed_dehazing_${baseFilename}`,
        `processed_deraining_${baseFilename}`,
        `processed_combined_${baseFilename}`
      ];

      console.log('Loading timing data for session IDs:', sessionIds);

      const timingDataArray = await APIService.getMultiProcessTimings(sessionIds);
      
      if (timingDataArray.length > 0) {
        const combinedProcesses = createProcessDataFromJSON(timingDataArray);
        setProcessesData(combinedProcesses);
      } else {
        setError('No timing data found for multi-process analysis');
      }
    } catch (err) {
      console.error('Failed to load multi-process data:', err);
      setError('Failed to load timing data');
    } finally {
      setDataLoading(false);
    }
  };

  const createProcessDataFromJSON = (timingDataArray: any[]): ProcessData[] => {
    const processes: ProcessData[] = [];

    timingDataArray.forEach(timingData => {
      if (!timingData || !timingData.frame_by_frame_timings) return;

      const method = timingData.processing_method || 'unknown';
      const frameTimings = timingData.frame_by_frame_timings;

      // Convert JSON data to our format
      const frameData = frameTimings.map((timing: any) => ({
        frame: timing.frame_number,
        processingTime: timing.total_time,
        wasEnhanced: timing.was_enhanced || timing.is_low_light || false
      }));

      // Calculate statistics
      const avgTime = frameData.reduce((sum: number, frame: { frame: number; processingTime: number; wasEnhanced: boolean }) => sum + frame.processingTime, 0) / frameData.length;
      const enhancedCount = frameData.filter((frame: { frame: number; processingTime: number; wasEnhanced: boolean }) => frame.wasEnhanced).length;
      const enhancementRate = (enhancedCount / frameData.length) * 100;

      processes.push({
        method,
        color: methodColors[method] || '#6B7280',
        frameData,
        avgTime,
        enhancementRate
      });
    });

    return processes;
  };

  const maxTime = Math.max(
    ...processesData.flatMap(process => process.frameData.map(frame => frame.processingTime)),
    0.001
  );

  const maxFrames = Math.max(
    ...processesData.map(process => Math.max(...process.frameData.map(frame => frame.frame))),
    1
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || processesData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = 400;
    ctx.scale(2, 2);

    const width = canvas.width / 2;
    const height = canvas.height / 2;
    const padding = { top: 20, right: 20, bottom: 60, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = '#1F2937';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (chartWidth / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartHeight);
      ctx.stroke();
    }

    // Draw lines for each process
    processesData.forEach((process, processIndex) => {
      ctx.strokeStyle = process.color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      process.frameData.forEach((frame, frameIndex) => {
        const x = padding.left + (frame.frame / maxFrames) * chartWidth;
        const y = padding.top + chartHeight - (frame.processingTime / maxTime) * chartHeight;
        
        if (frameIndex === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Draw enhanced frame markers
      process.frameData.forEach(frame => {
        if (frame.wasEnhanced) {
          const x = padding.left + (frame.frame / maxFrames) * chartWidth;
          const y = padding.top + chartHeight - (frame.processingTime / maxTime) * chartHeight;
          
          ctx.fillStyle = process.color;
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    });

    // Draw axes labels
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Frame Number', width / 2, height - 10);
    
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Processing Time (s)', 0, 0);
    ctx.restore();

    // Draw axis values
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      const value = (maxTime * (5 - i) / 5).toFixed(3);
      ctx.fillText(value, padding.left - 10, y + 4);
    }

    ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (chartWidth / 10) * i;
      const frameNum = Math.round((maxFrames * i) / 10);
      ctx.fillText(frameNum.toString(), x, height - padding.bottom + 20);
    }
  }, [processesData, maxTime, maxFrames]);

  if (loading || dataLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-teal-400 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
          <p className="text-gray-400">Loading multi-process data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="text-red-400 mb-2">⚠️</div>
          <p className="text-red-400 mb-2">{error}</p>
          <p className="text-gray-500 text-xs">Make sure JSON timing files exist for different processing methods</p>
          <button
            onClick={loadMultiProcessData}
            className="mt-3 px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-md text-sm transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (processesData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Layers className="w-12 h-12 text-gray-500 mx-auto mb-2" />
          <p className="text-gray-400">No multi-process data available</p>
          <p className="text-gray-500 text-sm mt-1">Process the same video with different methods to see comparison</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-lg font-semibold text-white">Multi-Process Comparison</h4>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Layers className="w-4 h-4 text-teal-400" />
            <span className="text-sm text-gray-400">{processesData.length} Methods</span>
          </div>
          <button
            onClick={loadMultiProcessData}
            className="flex items-center space-x-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
          >
            <RefreshCw className="w-3 h-3" />
            <span>Refresh</span>
          </button>
        </div>
      </div>
      
      <div className="bg-gray-700 rounded-lg p-4">
        <canvas
          ref={canvasRef}
          className="w-full h-48 rounded"
          style={{ background: '#1F2937' }}
        />
        
        <div className="mt-2 text-xs text-gray-400 text-center">
          Combined processing times from different methods on the same video
          <br />
          Each colored line represents a different processing method • Dots show enhanced frames
        </div>
      </div>

      {/* Method Legend and Statistics */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h5 className="text-md font-semibold text-white mb-3">Method Comparison</h5>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {processesData.map((process) => (
            <div key={process.method} className="bg-gray-600 rounded-lg p-3">
              <div className="flex items-center space-x-2 mb-2">
                <div 
                  className="w-4 h-4 rounded" 
                  style={{ backgroundColor: process.color }}
                />
                <span className="text-white font-medium capitalize">{process.method}</span>
              </div>
              <div className="text-xs text-gray-300 space-y-1">
                <div>Avg Time: {process.avgTime.toFixed(3)}s</div>
                <div>Frames: {process.frameData.length}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};