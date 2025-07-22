import React from 'react';
import { Monitor } from 'lucide-react';
import { ProcessingMethod } from '../types';

interface HeaderProps {
  methods: ProcessingMethod[];
  activeMethod: string;
  onMethodChange: (methodId: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ methods, activeMethod, onMethodChange }) => {
  return (
    <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Monitor className="w-8 h-8 text-teal-400" />
          <div>
            <h1 className="text-xl font-bold text-white">VisionEdge</h1>
            <p className="text-sm text-gray-400">Video Enhancement Platform by Team SW3</p>
          </div>
        </div>
        
        <nav className="flex space-x-2">
          {methods.map((method) => (
            <button
              key={method.id}
              onClick={() => onMethodChange(method.id)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                activeMethod === method.id
                  ? 'bg-teal-600 text-white shadow-lg'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600 hover:text-white'
              }`}
            >
              {method.name}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
};