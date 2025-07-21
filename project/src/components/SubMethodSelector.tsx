import React, { useState } from 'react';
import { ChevronDown } from 'lucide-react';

export interface SubMethod {
  id: string;
  name: string;
  description: string;
  backendValue: string;
  hasOptions?: boolean;
}

interface SubMethodSelectorProps {
  subMethods: SubMethod[];
  selectedSubMethod: string;
  thresholdValue: number;
  onSubMethodChange: (subMethodId: string) => void;
  onThresholdChange: (value: number) => void;
  onNestedOptionSelect: (subMethodId: string, mode: 'auto' | 'manual') => void;
}

export const SubMethodSelector: React.FC<SubMethodSelectorProps> = ({
  subMethods,
  selectedSubMethod,
  thresholdValue,
  onSubMethodChange,
  onThresholdChange,
  onNestedOptionSelect
}) => {
  const [hoveredSubMethod, setHoveredSubMethod] = useState<string | null>(null);
  const [localThreshold, setLocalThreshold] = useState<number>(thresholdValue ?? 128);

  return (
    <div className="relative space-y-3">
      <h4 className="text-sm font-medium text-gray-300">Processing Algorithm</h4>
      <div className="space-y-2">
        {subMethods.map((subMethod) => {
          const isHovered = hoveredSubMethod === subMethod.id;
          const isSelected = selectedSubMethod === subMethod.id;

          return (
            <div
              key={subMethod.id}
              className="relative group"
              onMouseEnter={() => subMethod.hasOptions && setHoveredSubMethod(subMethod.id)}
              onMouseLeave={() => subMethod.hasOptions && setHoveredSubMethod(null)}
            >
              <label className="flex items-start space-x-3 cursor-pointer group w-full">
                <input
                  type="radio"
                  name="subMethod"
                  value={subMethod.id}
                  checked={isSelected}
                  onChange={() => {
                    if (!subMethod.hasOptions) {
                      onSubMethodChange(subMethod.id);
                      onThresholdChange(0); // reset if no options
                      onNestedOptionSelect(subMethod.id, 'auto');
                    }
                  }}
                  className="mt-1 w-4 h-4 text-teal-600 bg-gray-700 border-gray-600 focus:ring-teal-500 focus:ring-2"
                />
                <div className="flex-1">
                  <div className="text-sm font-medium text-white group-hover:text-teal-400 transition-colors flex justify-between items-center">
                    {subMethod.name}
                    {subMethod.hasOptions && <ChevronDown className="w-4 h-4 ml-2" />}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">{subMethod.description}</div>
                </div>
              </label>

              {/* Nested options dropdown */}
              {subMethod.hasOptions && isHovered && (
                <div className="absolute left-0 top-full mt-2 w-60 bg-slate-800 border border-slate-600 rounded-md shadow-lg z-50 p-3">
                  <button
                    className="block w-full text-left text-sm text-slate-300 hover:bg-slate-700 px-2 py-1 rounded-md"
                    onClick={() => {
                      onThresholdChange(0); // clear threshold
                      onNestedOptionSelect(subMethod.id, 'auto');
                    }}
                  >
                    Auto
                  </button>

                  <div className="mt-3">
                    <label className="block text-xs text-slate-400 mb-1">Manual Threshold</label>
                    <input
                      type="range"
                      min={0}
                      max={255}
                      value={localThreshold}
                      onChange={(e) => setLocalThreshold(Number(e.target.value))}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-slate-300 mt-1">
                      <span>0</span>
                      <span>{localThreshold}</span>
                      <span>255</span>
                    </div>
                    <button
                      onClick={() => {
                        onThresholdChange(localThreshold);
                        onNestedOptionSelect(subMethod.id, 'manual');
                      }}
                      className="mt-2 w-full bg-teal-600 hover:bg-teal-700 text-white text-sm px-2 py-1 rounded-md"
                    >
                      Apply Threshold
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
