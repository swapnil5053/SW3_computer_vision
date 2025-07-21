// methods.ts

export interface SubMethod {
  id: string;
  name: string;
  description: string;
  backendValue: string;
  hasOptions?: boolean;
}

export interface ProcessingMethod {
  id: string;
  name: string;
  description: string;
  endpoint: string;
  color: string;
  subMethods?: SubMethod[];
}

export const processingMethods: ProcessingMethod[] = [
  {
    id: 'low-light',
    name: 'Low Light',
    description: 'Enhance low-light videos using CLAHE or UNet algorithms',
    endpoint: '/api/process_video/',
    color: '#14B8A6',
    subMethods: [
      {
        id: 'clahe',
        name: 'CLAHE',
        description: 'Contrast Limited Adaptive Histogram Equalization',
        backendValue: 'clahe'
      },
      {
        id: 'unet',
        name: 'UNet',
        description: 'Deep learning based enhancement using UNet architecture',
        backendValue: 'unet'
      }
    ]
  },
  {
    id: 'glare',
    name: 'Glare Reduction',
    description: 'Reduce glare and reflections in video content',
    endpoint: '/api/process_video/',
    color: '#14B8A6',
    subMethods: [
      {
        id: 'flare-reduction',
        name: 'Flare Reduction',
        description: 'Reduce lens flare artifacts',
        backendValue: 'flare-reduction'
      },
      {
        id: 'glare-dim',
        name: 'Glare Dimming',
        description: 'Dimming glare effects in images',
        backendValue: 'glare-dim',
        hasOptions: true
      },
      {
        id: 'combined',
        name: 'Combined',
        description: 'Combines both of the above',
        backendValue: 'combined',
        hasOptions: true
      }
    ]
  },
  {
    id: 'deraining',
    name: 'Deraining',
    description: 'Remove rain effects from video footage',
    endpoint: '/api/process_video/',
    color: '#6B7280',
    subMethods: [
      {
        id: 'raindrop',
        name: 'Detection threshold',
        description: 'set the threshold for raindrop detection',
        backendValue: 'glare-dim',
        hasOptions: true
      }
    ]
  },
  {
    id: 'tilt',
    name: 'Tilt Detection',
    description: 'Detect and correct tilt in video frames',
    endpoint: '/api/process_video/',
    color: '#6B7280'
  },
  {
    id: 'dehazing',
    name: 'Dehazing',
    description: 'Remove haze and fog from video content',
    endpoint: '/api/process_video/',
    color: '#6B7280'
  }
];
