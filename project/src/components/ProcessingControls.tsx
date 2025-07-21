import React from 'react';
import { Play, Loader2, CheckCircle, AlertCircle } from 'lucide-react';
import { ProcessingStatus, ProcessingMethod } from '../types';
import { SubMethodSelector } from './SubMethodSelector';

interface ProcessingControlsProps {
  status: ProcessingStatus;
  onProcess: () => void;
  canProcess: boolean;
  activeMethod: ProcessingMethod;
  selectedSubMethod?: string;
  onSubMethodChange?: (subMethodId: string) => void;
  progress?: number;
  thresholdValue: number;
  onThresholdChange: (value: number) => void;
  thresholdMode: 'auto' | 'manual';
  setThresholdMode: (mode: 'auto' | 'manual') => void;
}

export const ProcessingControls: React.FC<ProcessingControlsProps> = ({
  status,
  onProcess,
  canProcess,
  activeMethod,
  selectedSubMethod,
  onSubMethodChange,
  progress = 0,
  thresholdValue,
  onThresholdChange,
  thresholdMode,
  setThresholdMode
}) => {
  const [subMethodSelected, setSubMethodSelected] = React.useState(false);

  React.useEffect(() => {
    setSubMethodSelected(false);
  }, [activeMethod.id]);

  const handleSubMethodChange = (subMethodId: string) => {
    if (onSubMethodChange) {
      onSubMethodChange(subMethodId);
      setSubMethodSelected(true);
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'processing':
        return <Loader2 className="w-5 h-5 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Play className="w-5 h-5" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'uploading':
        return 'Uploading...';
      case 'processing':
        return 'Processing...';
      case 'completed':
        return 'Completed';
      case 'error':
        return 'Error';
      default:
        return 'Process Video';
    }
  };

  const getButtonColor = () => {
    switch (status) {
      case 'completed':
        return 'bg-green-600 hover:bg-green-700';
      case 'error':
        return 'bg-red-600 hover:bg-red-700';
      default:
        return 'bg-teal-600 hover:bg-teal-700';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-white mb-3">Processing Settings</h3>
        <div className="p-4 bg-gray-700 rounded-md">
          <div className="flex items-center justify-between mb-2">
            <span className="text-white font-medium">{activeMethod.name}</span>
            <span className="text-xs text-gray-400 bg-gray-600 px-2 py-1 rounded">
              {activeMethod.id}
            </span>
          </div>
          <p className="text-sm text-gray-400">{activeMethod.description}</p>
        </div>
      </div>

      {/* Sub-method selection UI */}
      {activeMethod.subMethods &&
        activeMethod.subMethods.length > 0 &&
        selectedSubMethod &&
        onSubMethodChange &&
        !subMethodSelected && (
          <SubMethodSelector
            subMethods={activeMethod.subMethods}
            selectedSubMethod={selectedSubMethod}
            thresholdValue={thresholdValue}
            onThresholdChange={onThresholdChange}
            onSubMethodChange={handleSubMethodChange}
            onNestedOptionSelect={(subMethodId, mode) => {
              handleSubMethodChange(subMethodId);
              setThresholdMode(mode);
            }}
          />
        )}

      {/* Display selected sub-method */}
      {subMethodSelected && activeMethod.subMethods && selectedSubMethod && (
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-white">Selected Algorithm</h4>
              <p className="text-sm text-teal-400 mt-1">
                {
                  activeMethod.subMethods.find((sm) =>
                    selectedSubMethod.startsWith(sm.id)
                  )?.name
                }
              </p>
            </div>
            <button
              onClick={() => setSubMethodSelected(false)}
              className="text-gray-400 hover:text-white text-sm underline"
            >
              Change
            </button>
          </div>
        </div>
      )}

      {/* Progress bar */}
      {status === 'processing' && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Progress</span>
            <span className="text-sm text-gray-400">{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-teal-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Main process button */}
      <button
        onClick={onProcess}
        disabled={!canProcess || status === 'processing'}
        className={`w-full flex items-center justify-center space-x-2 px-4 py-3 rounded-md text-white font-medium transition-all duration-200 ${
          !canProcess || status === 'processing'
            ? 'bg-gray-600 cursor-not-allowed'
            : getButtonColor()
        }`}
      >
        {getStatusIcon()}
        <span>{getStatusText()}</span>
      </button>
    </div>
  );
};
