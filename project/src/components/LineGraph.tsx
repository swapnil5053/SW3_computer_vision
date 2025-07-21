import React, { useRef, useEffect } from 'react';
import { RefreshCw, Clock } from 'lucide-react';

interface FrameData {
  frame: number;
  time: number;
  classifyTime: number;
  processTime: number;
  isLowLight: boolean;
  wasEnhanced: boolean;
}

interface LineGraphProps {
  frameData?: FrameData[];
  performance: any;
  onRefresh: () => void;
}

export const LineGraph: React.FC<LineGraphProps> = ({ frameData, performance, onRefresh }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const maxTime = Math.max(...(frameData?.map(d => d.time) ?? [0]));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    canvas.width = canvas.offsetWidth * 2;
    canvas.height = 320;
    ctx.scale(2, 2);

    const width = canvas.width / 2;
    const height = canvas.height / 2;
    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Clear canvas
    ctx.fillStyle = '#374151';
    ctx.fillRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = '#4B5563';
    ctx.lineWidth = 0.5;

    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();
    }

    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (chartWidth / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + chartHeight);
      ctx.stroke();
    }

    // Draw line graph
    ctx.strokeStyle = '#14B8A6';
    ctx.lineWidth = 2;
    ctx.beginPath();

    frameData?.forEach((data, index) => {
      const x = padding.left + (index / ((frameData.length ?? 1) - 1)) * chartWidth;
      const y = padding.top + chartHeight - (data.time / maxTime) * chartHeight;

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw enhanced frame markers
    frameData?.forEach((data, index) => {
      if (data.wasEnhanced) {
        const x = padding.left + (index / ((frameData.length ?? 1) - 1)) * chartWidth;
        const y = padding.top + chartHeight - (data.time / maxTime) * chartHeight;

        ctx.fillStyle = '#3B82F6';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
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

    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (chartHeight / 5) * i;
      const value = (maxTime * (5 - i) / 5).toFixed(3);
      ctx.fillText(value, padding.left - 10, y + 4);
    }

    ctx.textAlign = 'center';
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (chartWidth / 10) * i;
      const frameNum = Math.round((frameData?.length ?? 0 * i) / 10);
      ctx.fillText(frameNum.toString(), x, height - padding.bottom + 20);
    }
  }, [frameData, maxTime]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-lg font-semibold text-white">Frame Processing Times (Line Graph)</h4>
        <div className="flex items-center space-x-4 text-sm text-gray-400">
          <div className="flex items-center space-x-1">
            <div className="w-3 h-1 bg-teal-400"></div>
            <span>Processing Time</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
            <span>Enhanced Frames</span>
          </div>
          <div className="flex items-center space-x-1">
            <Clock className="w-4 h-4" />
            <span>Avg: {performance?.avg_delay_per_frame?.toFixed(3) ?? '0.000'}s</span>
          </div>
          <button
            onClick={onRefresh}
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
          className="w-full h-40 rounded"
          style={{ background: '#374151' }}
        />
        <div className="mt-2 text-xs text-gray-400 text-center">
          Frame-by-frame processing times (showing {frameData?.length ?? 0} frames)
          <br />
          <span className="text-blue-400">Blue dots</span> = Enhanced frames, <span className="text-teal-400">Teal line</span> = Processing time trend
        </div>
      </div>
    </div>
  );
};
