import React, { useState, useEffect, useRef } from 'react';
import { VideoFile, ProcessingResult } from '../types';
import { Play, Pause, RotateCcw, Volume2, VolumeX, Settings, Maximize2 } from 'lucide-react';

interface VideoPlayerProps {
  originalVideo: VideoFile | null;
  processedVideo: ProcessingResult | null;
  title: string;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({ originalVideo, processedVideo, title }) => {
  const [originalVideoError, setOriginalVideoError] = useState(false);
  const [processedVideoError, setProcessedVideoError] = useState(false);
  const [originalVideoLoading, setOriginalVideoLoading] = useState(false);
  const [processedVideoLoading, setProcessedVideoLoading] = useState(false);
  
  // Synchronized playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [showIndividualControls, setShowIndividualControls] = useState(false);
  
  // Individual video states
  const [originalPlaying, setOriginalPlaying] = useState(false);
  const [processedPlaying, setProcessedPlaying] = useState(false);
  const [originalCurrentTime, setOriginalCurrentTime] = useState(0);
  const [processedCurrentTime, setProcessedCurrentTime] = useState(0);
  
  // Video refs for synchronization
  const originalVideoRef = useRef<HTMLVideoElement>(null);
  const processedVideoRef = useRef<HTMLVideoElement>(null);
  const isSeekingRef = useRef(false);
  const isSyncingRef = useRef(false);

  useEffect(() => {
    if (originalVideo?.url) {
      setOriginalVideoError(false);
      setOriginalVideoLoading(true);
    }
  }, [originalVideo?.url]);

  useEffect(() => {
    if (processedVideo?.outputUrl) {
      setProcessedVideoError(false);
      setProcessedVideoLoading(true);
    }
  }, [processedVideo?.outputUrl]);

  // Synchronize video playback
  const syncVideos = (sourceVideo: HTMLVideoElement, targetVideo: HTMLVideoElement | null) => {
    if (!targetVideo || isSeekingRef.current || isSyncingRef.current || showIndividualControls) return;
    
    const timeDiff = Math.abs(sourceVideo.currentTime - targetVideo.currentTime);
    if (timeDiff > 0.1) { // Only sync if difference is significant
      isSyncingRef.current = true;
      targetVideo.currentTime = sourceVideo.currentTime;
      setTimeout(() => {
        isSyncingRef.current = false;
      }, 100);
    }
  };

  const handlePlay = () => {
    const original = originalVideoRef.current;
    const processed = processedVideoRef.current;
    
    if (original) original.play();
    if (processed) processed.play();
    setIsPlaying(true);
  };

  const handlePause = () => {
    const original = originalVideoRef.current;
    const processed = processedVideoRef.current;
    
    if (original) original.pause();
    if (processed) processed.pause();
    setIsPlaying(false);
  };

  const handleSeek = (time: number) => {
    isSeekingRef.current = true;
    const original = originalVideoRef.current;
    const processed = processedVideoRef.current;
    
    if (original) original.currentTime = time;
    if (processed) processed.currentTime = time;
    
    setCurrentTime(time);
    
    setTimeout(() => {
      isSeekingRef.current = false;
    }, 100);
  };

  const handleReset = () => {
    handleSeek(0);
    handlePause();
  };

  const handleVolumeChange = (newVolume: number) => {
    const original = originalVideoRef.current;
    const processed = processedVideoRef.current;
    
    if (original) original.volume = newVolume;
    if (processed) processed.volume = newVolume;
    
    setVolume(newVolume);
    setIsMuted(newVolume === 0);
  };

  const handlePlaybackRateChange = (rate: number) => {
    const original = originalVideoRef.current;
    const processed = processedVideoRef.current;
    
    if (original) original.playbackRate = rate;
    if (processed) processed.playbackRate = rate;
    
    setPlaybackRate(rate);
  };

  const toggleMute = () => {
    if (isMuted) {
      handleVolumeChange(volume > 0 ? volume : 0.5);
    } else {
      handleVolumeChange(0);
    }
  };

  // Individual video controls
  const handleIndividualPlay = (videoType: 'original' | 'processed') => {
    const video = videoType === 'original' ? originalVideoRef.current : processedVideoRef.current;
    if (video) {
      video.play();
      if (videoType === 'original') setOriginalPlaying(true);
      else setProcessedPlaying(true);
    }
  };

  const handleIndividualPause = (videoType: 'original' | 'processed') => {
    const video = videoType === 'original' ? originalVideoRef.current : processedVideoRef.current;
    if (video) {
      video.pause();
      if (videoType === 'original') setOriginalPlaying(false);
      else setProcessedPlaying(false);
    }
  };

  const handleIndividualSeek = (videoType: 'original' | 'processed', time: number) => {
    const video = videoType === 'original' ? originalVideoRef.current : processedVideoRef.current;
    if (video) {
      video.currentTime = time;
      if (videoType === 'original') setOriginalCurrentTime(time);
      else setProcessedCurrentTime(time);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleOriginalVideoLoad = () => {
    setOriginalVideoLoading(false);
    setOriginalVideoError(false);
    const video = originalVideoRef.current;
    if (video) {
      setDuration(video.duration);
      video.volume = volume;
      video.playbackRate = playbackRate;
    }
  };

  const handleProcessedVideoLoad = () => {
    setProcessedVideoLoading(false);
    setProcessedVideoError(false);
    const video = processedVideoRef.current;
    if (video) {
      video.volume = volume;
      video.playbackRate = playbackRate;
    }
  };

  const handleOriginalVideoError = (e: React.SyntheticEvent<HTMLVideoElement, Event>) => {
    console.error('Original video loading error:', e);
    setOriginalVideoLoading(false);
    setOriginalVideoError(true);
  };

  const handleProcessedVideoError = (e: React.SyntheticEvent<HTMLVideoElement, Event>) => {
    console.error('Processed video loading error:', e);
    setProcessedVideoLoading(false);
    setProcessedVideoError(true);
  };

  const renderVideoSection = (
    video: { url: string; name: string } | null,
    title: string,
    isLoading: boolean,
    hasError: boolean,
    onLoad: () => void,
    onError: (e: React.SyntheticEvent<HTMLVideoElement, Event>) => void,
    videoRef: React.RefObject<HTMLVideoElement>,
    showProcessingComplete?: boolean,
    videoType?: 'original' | 'processed'
  ) => {
    const isOriginal = videoType === 'original';
    const individualPlaying = isOriginal ? originalPlaying : processedPlaying;
    const individualCurrentTime = isOriginal ? originalCurrentTime : processedCurrentTime;

    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden flex flex-col h-full">
        <div className="p-4 border-b border-gray-700 flex-shrink-0">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">{title}</h3>
            {showProcessingComplete && (
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-green-400 text-sm">Processing Complete</span>
              </div>
            )}
          </div>
          {video && (
            <p className="text-gray-400 text-sm mt-1 truncate">{video.name}</p>
          )}
        </div>
        
        <div className="relative flex-1 flex items-center justify-center bg-black min-h-0">
          {!video && (
            <div className="text-center">
              <div className="w-16 h-16 bg-gray-700 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-gray-400">
                {title === 'Original Video' ? 'Upload a video to get started' : 'Process video to see result'}
              </p>
            </div>
          )}

          {video && isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 z-10">
              <div className="text-center">
                <div className="w-8 h-8 border-2 border-teal-400 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                <p className="text-gray-400 text-sm">Loading video...</p>
              </div>
            </div>
          )}
          
          {video && hasError && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-75 z-10">
              <div className="text-center">
                <div className="text-red-400 mb-2">⚠️</div>
                <p className="text-red-400 mb-2">Video loading failed</p>
                <p className="text-gray-400 text-sm">Check console for details</p>
              </div>
            </div>
          )}
          
          {video && (
            <video
              ref={videoRef}
              src={video.url}
              className="w-full h-full object-contain max-h-full"
              onLoadStart={() => title === 'Original Video' ? setOriginalVideoLoading(true) : setProcessedVideoLoading(true)}
              onLoadedData={onLoad}
              onError={onError}
              onTimeUpdate={(e) => {
                const video = e.currentTarget;
                if (!showIndividualControls) {
                  setCurrentTime(video.currentTime);
                  // Sync the other video
                  if (title === 'Original Video') {
                    syncVideos(video, processedVideoRef.current);
                  } else {
                    syncVideos(video, originalVideoRef.current);
                  }
                } else {
                  // Individual controls mode
                  if (videoType === 'original') {
                    setOriginalCurrentTime(video.currentTime);
                  } else {
                    setProcessedCurrentTime(video.currentTime);
                  }
                }
              }}
              onPlay={() => {
                if (!showIndividualControls) {
                  setIsPlaying(true);
                } else {
                  if (videoType === 'original') setOriginalPlaying(true);
                  else setProcessedPlaying(true);
                }
              }}
              onPause={() => {
                if (!showIndividualControls) {
                  setIsPlaying(false);
                } else {
                  if (videoType === 'original') setOriginalPlaying(false);
                  else setProcessedPlaying(false);
                }
              }}
              muted={showIndividualControls ? false : isMuted}
              controls={showIndividualControls}
            >
              Your browser does not support the video tag.
            </video>
          )}
        </div>

      </div>
    );
  };

  const hasVideos = originalVideo || processedVideo;

  return (
    <div className="flex flex-col h-full">
      {/* Video Display */}
      <div className="grid grid-cols-2 gap-4 flex-1 min-h-0">
        {/* Original Video */}
        {renderVideoSection(
          originalVideo ? { url: originalVideo.url, name: originalVideo.name } : null,
          'Original Video',
          originalVideoLoading,
          originalVideoError,
          handleOriginalVideoLoad,
          handleOriginalVideoError,
          originalVideoRef,
          false,
          'original'
        )}

        {/* Processed Video */}
        {renderVideoSection(
          processedVideo ? { url: processedVideo.outputUrl, name: processedVideo.filename } : null,
          'Processed Video',
          processedVideoLoading,
          processedVideoError,
          handleProcessedVideoLoad,
          handleProcessedVideoError,
          processedVideoRef,
          !!processedVideo,
          'processed'
        )}
      </div>

      {/* Main Controls */}
      {hasVideos && (
        <div className="mt-4 bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="space-y-3">
            {/* Control Mode Toggle */}
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium text-white">Video Controls</h4>
              <button
                onClick={() => setShowIndividualControls(!showIndividualControls)}
                className="flex items-center space-x-2 px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-md text-sm text-gray-300 transition-colors"
              >
                <Settings size={14} />
                <span>{showIndividualControls ? 'Synchronized' : 'Individual'}</span>
              </button>
            </div>

            {/* Synchronized Controls */}
            {!showIndividualControls && (
              <>
                {/* Progress Bar */}
                <div className="flex items-center space-x-3">
                  <span className="text-sm text-gray-400 w-12">{formatTime(currentTime)}</span>
                  <div className="flex-1">
                    <div 
                      className="w-full h-2 bg-gray-700 rounded-lg cursor-pointer"
                      onClick={(e) => {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const clickX = e.clientX - rect.left;
                        const percentage = clickX / rect.width;
                        const newTime = percentage * (duration || 0);
                        handleSeek(newTime);
                      }}
                    >
                      <div 
                        className="h-full bg-teal-400 rounded-lg transition-all duration-200"
                        style={{ width: `${duration > 0 ? (currentTime / duration) * 100 : 0}%` }}
                      />
                    </div>
                  </div>
                  <span className="text-sm text-gray-400 w-12">{formatTime(duration)}</span>
                </div>

                {/* Control Buttons */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={isPlaying ? handlePause : handlePlay}
                      className="flex items-center justify-center w-10 h-10 bg-teal-600 hover:bg-teal-700 rounded-full text-white transition-colors"
                    >
                      {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                    </button>
                    
                    <button
                      onClick={handleReset}
                      className="flex items-center justify-center w-10 h-10 bg-gray-600 hover:bg-gray-700 rounded-full text-white transition-colors"
                    >
                      <RotateCcw size={18} />
                    </button>

                    {/* Playback Speed */}
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-400">Speed:</span>
                      <select
                        value={playbackRate}
                        onChange={(e) => handlePlaybackRateChange(parseFloat(e.target.value))}
                        className="bg-gray-700 text-white text-sm rounded px-2 py-1 border border-gray-600"
                      >
                        <option value={0.25}>0.25x</option>
                        <option value={0.5}>0.5x</option>
                        <option value={0.75}>0.75x</option>
                        <option value={1}>1x</option>
                        <option value={1.25}>1.25x</option>
                        <option value={1.5}>1.5x</option>
                        <option value={2}>2x</option>
                      </select>
                    </div>
                  </div>

                  {/* Volume Control */}
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={toggleMute}
                      className="text-gray-400 hover:text-white transition-colors"
                    >
                      {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
                    </button>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={isMuted ? 0 : volume}
                      onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
                      className="w-20 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <span className="text-xs text-gray-400 w-8">{Math.round((isMuted ? 0 : volume) * 100)}%</span>
                  </div>
                </div>
              </>
            )}

            {/* Individual Controls Info */}
            {showIndividualControls && (
              <div className="text-center text-sm text-gray-400">
                Individual controls are now enabled. Use the native video controls on each video to play them independently.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};