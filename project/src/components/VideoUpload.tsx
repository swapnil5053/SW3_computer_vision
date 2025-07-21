import React, { useRef, useState } from 'react';
import { Upload, X, Play } from 'lucide-react';
import { VideoFile } from '../types';

interface VideoUploadProps {
  onVideoSelect: (video: VideoFile) => void;
  selectedVideo: VideoFile | null;
  onClear: () => void;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({ onVideoSelect, selectedVideo, onClear }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (file: File) => {
    if (file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file);
      onVideoSelect({
        file,
        url,
        name: file.name,
        size: file.size
      });
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (selectedVideo) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-white">Selected Video</h3>
          <button
            onClick={onClear}
            className="text-gray-400 hover:text-red-400 transition-colors"
          >
            <X size={20} />
          </button>
        </div>
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-teal-600 rounded-lg flex items-center justify-center">
            <Play size={20} className="text-white" />
          </div>
          <div className="flex-1">
            <p className="text-white font-medium truncate">{selectedVideo.name}</p>
            <p className="text-gray-400 text-sm">{formatFileSize(selectedVideo.size)}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
        dragActive
          ? 'border-teal-400 bg-teal-400/5'
          : 'border-gray-600 hover:border-gray-500'
      }`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <Upload className="mx-auto mb-4 text-gray-400" size={48} />
      <h3 className="text-lg font-semibold text-white mb-2">Upload Video</h3>
      <p className="text-gray-400 mb-4">Drag and drop a video file here, or click to select</p>
      <button
        onClick={() => fileInputRef.current?.click()}
        className="bg-teal-600 hover:bg-teal-700 text-white px-6 py-2 rounded-md transition-colors"
      >
        Select Video
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        onChange={handleInputChange}
        className="hidden"
      />
    </div>
  );
};