import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { VideoUpload } from './components/VideoUpload';
import { VideoPlayer } from './components/VideoPlayer';
import { ProcessingControls } from './components/ProcessingControls';
import { Analytics } from './components/Analytics';
import { processingMethods } from './config/methods';
import { useVideoProcessing } from './hooks/useVideoProcessing';
import { VideoFile } from './types';

function App() {
  const [activeMethod, setActiveMethod] = useState('low-light');
  const [selectedSubMethod, setSelectedSubMethod] = useState('flare-reduction');
  const [selectedVideo, setSelectedVideo] = useState<VideoFile | null>(null);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [thresholdValue, setThresholdValue] = useState<number>(128);
  const [thresholdMode, setThresholdMode] = useState<'auto' | 'manual'>('auto');

  const { status, progress, error, result, processVideo, reset } = useVideoProcessing();

  const currentMethod = processingMethods.find(m => m.id === activeMethod) || processingMethods[0];

  useEffect(() => {
    if (currentMethod.subMethods && currentMethod.subMethods.length > 0) {
      setSelectedSubMethod(currentMethod.subMethods[0].id);
    } else {
      setSelectedSubMethod('');
    }
  }, [activeMethod, currentMethod]);

  const handleMethodChange = (methodId: string) => {
    setActiveMethod(methodId);
    reset();
  };

  const handleSubMethodChange = (subMethodId: string) => {
    setSelectedSubMethod(subMethodId);
    reset();
  };

  const handleVideoSelect = (video: VideoFile) => {
    setSelectedVideo(video);
    reset();
  };

  const handleClearVideo = () => {
    setSelectedVideo(null);
    reset();
  };

  const handleProcess = async () => {
    if (!selectedVideo) return;

    const processingSubMethod = selectedSubMethod;
    const args: string[] = [];

    if (
      (processingSubMethod === 'glare-dim' || processingSubMethod === 'combined') &&
      thresholdMode === 'manual'
    ) {
      args.push('--threshold', thresholdValue.toString());
    }

    await processVideo(selectedVideo, activeMethod, processingSubMethod, args);
  };

  const canProcess = selectedVideo && status !== 'processing';

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col">
      <Header methods={processingMethods} activeMethod={activeMethod} onMethodChange={handleMethodChange} />

      <main className="flex-1 container mx-auto px-6 py-6 flex flex-col">
        {!selectedVideo && (
          <div className="mb-6">
            <VideoUpload onVideoSelect={handleVideoSelect} selectedVideo={selectedVideo} onClear={handleClearVideo} />
          </div>
        )}

        {selectedVideo && (
          <div className="flex-1 grid grid-cols-12 gap-6 min-h-0">
            <div className={`${showAnalytics ? 'col-span-6' : 'col-span-8'} min-h-0`}>
              <VideoPlayer originalVideo={selectedVideo} processedVideo={result} title="Video Player" />
            </div>

            <div className={`${showAnalytics ? 'col-span-6' : 'col-span-4'} flex flex-col space-y-4`}>
              <VideoUpload onVideoSelect={handleVideoSelect} selectedVideo={selectedVideo} onClear={handleClearVideo} />

              <ProcessingControls
                status={status}
                onProcess={handleProcess}
                canProcess={canProcess}
                activeMethod={currentMethod}
                selectedSubMethod={selectedSubMethod}
                onSubMethodChange={handleSubMethodChange}
                thresholdValue={thresholdValue}
                onThresholdChange={setThresholdValue}
                thresholdMode={thresholdMode}
                setThresholdMode={setThresholdMode}
                progress={progress}
              />

              {error && (
                <div className="bg-red-900/50 border border-red-800 rounded-lg p-4">
                  <p className="text-red-400 text-sm">{error}</p>
                </div>
              )}

              <button
                onClick={() => setShowAnalytics(!showAnalytics)}
                className="flex items-center justify-center space-x-2 px-4 py-3 bg-teal-600 hover:bg-teal-700 text-white rounded-md transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>{showAnalytics ? 'Hide Analytics' : 'Show Analytics'}</span>
              </button>
            </div>
          </div>
        )}

        {showAnalytics && (
          <div className="fixed right-6 top-20 bottom-6 w-2/5 z-50">
            <Analytics isVisible={showAnalytics} onClose={() => setShowAnalytics(false)} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
