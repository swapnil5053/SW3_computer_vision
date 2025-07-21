import React, { useState, useEffect, useRef } from 'react';
import { BarChart3, FileText, TrendingUp, Clock, Cpu, Monitor, RefreshCw } from 'lucide-react';
import { ProcessingLog } from '../types';
import { APIService } from '../services/api';
import { LineGraph } from './LineGraph';
import { MultiProcessGraph } from './MultiProcessGraph';

interface AnalyticsProps {
  isVisible: boolean;
  onClose: () => void;
}

export const Analytics: React.FC<AnalyticsProps> = ({ isVisible, onClose }) => {
  // ALL HOOKS MUST BE AT THE TOP LEVEL - NEVER CONDITIONAL
  const [activeTab, setActiveTab] = useState<'graph' | 'details' | 'multiprocess'>('graph');
  const [logs, setLogs] = useState<ProcessingLog[]>([]);
  const [latestLog, setLatestLog] = useState<ProcessingLog | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [frameTimings, setFrameTimings] = useState<any>(null);
  
  // Multi-process simulation data
  const [multiProcessData, setMultiProcessData] = useState<any>(null);

  // ✅ FIXED: State for the graph data is now at the top level.
  const [graphData, setGraphData] = useState<any[] | null>(null);

  // Load frame timings when latestLog changes
  useEffect(() => {
    const loadFrameTimings = async () => {
      if (latestLog?.video?.output_file) {
        try {
          const fullFilename = latestLog.video.output_file.split('/').pop() || '';
          const sessionId = fullFilename.substring(0, fullFilename.lastIndexOf('.')) || fullFilename;
          
          console.log('🔍 Loading frame timings for:', {
            output_file: latestLog.video.output_file,
            extracted_filename: fullFilename,
            session_id: sessionId,
            expected_timing_file: `${sessionId}_timings.json`
          });
          
          const timings = await APIService.getFrameTimings(sessionId);
          
          if (timings && !timings.error) {
            setFrameTimings(timings);
          } else {
            setFrameTimings(null); // Ensure fallback is triggered if timings are invalid
          }
        } catch (error) {
          console.error('Failed to load frame timings:', error);
          setFrameTimings(null);
        }
      }
    };
    loadFrameTimings();
  }, [latestLog]);

  // ✅ FIXED: Effect for preparing graph data is now at the top level.
  useEffect(() => {
    if (!latestLog) {
      setGraphData(null); // Clear data if there's no log
      return;
    }

    const { video, performance } = latestLog;

    // Happy path: Use real timing data if available
    if (frameTimings && frameTimings.frame_by_frame_timings) {
      const realFrameData = frameTimings.frame_by_frame_timings.map((timing: any) => ({
        frame: timing.frame_number,
        time: timing.total_time,
        classifyTime: timing.classify_time,
        processTime: timing.process_time || timing.enhance_time,
        isLowLight: timing.is_low_light,
        wasEnhanced: timing.was_enhanced !== undefined ? timing.was_enhanced : timing.is_low_light
      }));
      setGraphData(realFrameData);
    } else {
      // Fallback path: Generate dummy data asynchronously
      const timer = setTimeout(() => {
        const frameCount = video.total_frames;
        const avgTime = performance.avg_delay_per_frame;
        const dummyFrameData = Array.from({ length: frameCount }, (_, i) => ({
          frame: i + 1,
          time: avgTime * (0.8 + Math.random() * 0.4),
          classifyTime: avgTime * 0.3,
          processTime: avgTime * 0.7,
          isLowLight: Math.random() > 0.6,
          wasEnhanced: Math.random() > 0.6
        }));
        setGraphData(dummyFrameData);
      }, 5);

      // Cleanup function to prevent setting state on an unmounted component
      return () => clearTimeout(timer);
    }
  }, [latestLog, frameTimings]); // Re-run when log or timings data changes

  // Load logs when component becomes visible
  useEffect(() => {
    if (isVisible) {
      loadLogs();
    }
  }, [isVisible]);

  const loadLogs = async () => {
    setLoading(true);
    setError(null);
    try {
      const [allLogs, latest] = await Promise.all([
        APIService.getProcessingLogs(),
        APIService.getLatestLog()
      ]);
      setLogs(allLogs);
      setLatestLog(latest);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to connect to MongoDB';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // ✅ FIXED: This function is now a "dumb" renderer. It contains no hooks.
  const renderPerformanceGraph = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <RefreshCw className="w-8 h-8 text-teal-400 mx-auto mb-2 animate-spin" />
            <p className="text-gray-400">Loading performance data...</p>
          </div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="text-red-400 mb-2">⚠️</div>
            <p className="text-red-400 mb-2">MongoDB Connection Error</p>
            <p className="text-gray-400 text-sm">{error}</p>
            <p className="text-gray-500 text-xs mt-2">Backend API endpoints needed for MongoDB access</p>
            <button
              onClick={loadLogs}
              className="mt-3 px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-md text-sm transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    if (!graphData) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <BarChart3 className="w-12 h-12 text-gray-500 mx-auto mb-2" />
            <p className="text-gray-400">No processing data available</p>
            <p className="text-gray-500 text-sm mt-1">Process a video to see performance metrics</p>
          </div>
        </div>
      );
    }

    // Render graph using data from the top-level state
    return <LineGraph frameData={graphData} performance={latestLog.performance} onRefresh={loadLogs} />;
  };

  const renderProcessingDetails = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <RefreshCw className="w-8 h-8 text-teal-400 mx-auto mb-2 animate-spin" />
            <p className="text-gray-400">Loading processing details...</p>
          </div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="text-red-400 mb-2">⚠️</div>
            <p className="text-red-400 mb-2">MongoDB Connection Error</p>
            <p className="text-gray-400 text-sm">{error}</p>
            <p className="text-gray-500 text-xs mt-2">Backend API endpoints needed for MongoDB access</p>
            <button
              onClick={loadLogs}
              className="mt-3 px-4 py-2 bg-teal-600 hover:bg-teal-700 text-white rounded-md text-sm transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    if (!latestLog) {
      return (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <FileText className="w-12 h-12 text-gray-500 mx-auto mb-2" />
            <p className="text-gray-400">No processing details available</p>
            <p className="text-gray-500 text-sm mt-1">Process a video to see detailed metrics</p>
          </div>
        </div>
      );
    }

    const { video, performance, process_category, model_used, status, timestamp } = latestLog;

    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h4 className="text-lg font-semibold text-white">Latest Processing Session</h4>
          <div className="flex items-center space-x-2">
            <span className="text-xs text-gray-400">
              {new Date(timestamp).toLocaleString()}
            </span>
            <button
              onClick={loadLogs}
              className="flex items-center space-x-1 px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Refresh</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gray-700 rounded-lg p-4">
            <h5 className="text-md font-semibold text-white mb-3 flex items-center">
              <Monitor className="w-4 h-4 mr-2" />
              Video Information
            </h5>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Input File:</span>
                <span className="text-white text-xs truncate max-w-32" title={video.input_file}>
                  {video.input_file.split('/').pop()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Duration:</span>
                <span className="text-white">{video.duration.toFixed(1)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Resolution:</span>
                <span className="text-white">{video.resolution}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">FPS:</span>
                <span className="text-white">{video.fps.toFixed(1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Total Frames:</span>
                <span className="text-white">{video.total_frames.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Processed:</span>
                <span className="text-white">{video.processed_frames.toLocaleString()}</span>
              </div>
            </div>
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <h5 className="text-md font-semibold text-white mb-3 flex items-center">
              <TrendingUp className="w-4 h-4 mr-2" />
              Performance Metrics
            </h5>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Total Time:</span>
                <span className="text-white">{performance.total_time.toFixed(2)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Avg/Frame:</span>
                <span className="text-white">{performance.avg_delay_per_frame.toFixed(3)}s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Device:</span>
                <span className="text-white">{performance.device_used}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">CPU Usage:</span>
                <span className="text-white">{performance.cpu_usage_percent.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">GPU Usage:</span>
                <span className="text-white">
                  {performance.gpu_usage_percent ? `${performance.gpu_usage_percent.toFixed(1)}%` : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-700 rounded-lg p-4">
          <h5 className="text-md font-semibold text-white mb-3 flex items-center">
            <Cpu className="w-4 h-4 mr-2" />
            Processing Details
          </h5>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">Method:</span>
                <span className="text-white">{process_category}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Model:</span>
                <span className="text-white">{model_used}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Status:</span>
                <span className={`${status === 'success' ? 'text-green-400' : 'text-red-400'}`}>
                  {status}
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-400">CPU:</span>
                <span className="text-white text-xs" title={performance.device_specs.cpu}>
                  {performance.device_specs.cpu.length > 20 
                    ? `${performance.device_specs.cpu.substring(0, 20)}...`
                    : performance.device_specs.cpu
                  }
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">GPU:</span>
                <span className="text-white text-xs" title={performance.device_specs.gpu}>
                  {performance.device_specs.gpu.length > 20 
                    ? `${performance.device_specs.gpu.substring(0, 20)}...`
                    : performance.device_specs.gpu
                  }
                </span>
              </div>
            </div>
          </div>
        </div>

        {logs.length > 1 && (
          <div className="bg-gray-700 rounded-lg p-4">
            <h5 className="text-md font-semibold text-white mb-3">Recent Processing History</h5>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {logs.slice(0, 5).map((log, index) => (
                <div key={log._id || index} className="flex items-center justify-between text-sm py-1">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${log.status === 'success' ? 'bg-green-400' : 'bg-red-400'}`} />
                    <span className="text-gray-300">{log.model_used}</span>
                    <span className="text-gray-500">•</span>
                    <span className="text-gray-400">{log.video.resolution}</span>
                  </div>
                  <span className="text-gray-400 text-xs">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Early return AFTER all hooks have been called
  if (!isVisible) return null;

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 h-full flex flex-col">
      <div className="border-b border-gray-700 flex-shrink-0">
        <div className="flex items-center justify-between px-6 py-4">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('graph')}
              className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'graph'
                  ? 'border-teal-400 text-teal-400'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4" />
                <span>Line Graph</span>
              </div>
            </button>
            <button
              onClick={() => setActiveTab('details')}
              className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'details'
                  ? 'border-teal-400 text-teal-400'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              <div className="flex items-center space-x-2">
                <FileText className="w-4 h-4" />
                <span>Processing Details</span>
              </div>
            </button>
            <button
              onClick={() => setActiveTab('multiprocess')}
              className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'multiprocess'
                  ? 'border-teal-400 text-teal-400'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-4 h-4" />
                <span>Multi-Process</span>
              </div>
            </button>
          </nav>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>
      
      <div className="p-6 flex-1 overflow-y-auto">
        {activeTab === 'graph' && renderPerformanceGraph()}
        {activeTab === 'details' && renderProcessingDetails()}
        {activeTab === 'multiprocess' && <MultiProcessGraph loading={loading} />}
        {activeTab === 'multiprocess' && <MultiProcessGraph loading={loading} latestVideoInfo={latestLog?.video} />}
      </div>
    </div>
  );
};