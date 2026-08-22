import { toggleProcess, setProcessMode } from './process.js';

export function toggleRobotViz() {
  toggleProcess();
  setProcessMode('dataset');
}
