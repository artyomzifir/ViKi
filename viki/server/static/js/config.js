// Server configuration panel + server-alive indicator.
import { api, log } from './core.js';

export function toggleConfig() {
  const panel = document.getElementById('config-panel');
  const isVisible = panel.style.display === 'block';
  panel.style.display = isVisible ? 'none' : 'block';
  if (!isVisible) loadConfig();
}

export async function loadConfig() {
  try {
    const config = await api('GET', '/api/config');
    document.getElementById('config-editor').value = JSON.stringify(config, null, 2);
  } catch (e) {
    log('Failed to load config: ' + e, 'error');
  }
}

export function toggleConfigHelp() {
  const help = document.getElementById('config-help');
  help.style.display = help.style.display === 'block' ? 'none' : 'block';
}

export async function saveConfig() {
  try {
    const content = document.getElementById('config-editor').value;
    const config = JSON.parse(content);
    await api('POST', '/api/config', config);
    log('Configuration saved successfully', 'ok');
  } catch (e) {
    log('Failed to save config: ' + e, 'error');
  }
}

export async function resetConfig() {
  if (!confirm('Reset all settings to defaults? This will overwrite your current configuration.')) return;
  try {
    await api('POST', '/api/config/reset');
    await loadConfig();
    log('Configuration reset to defaults', 'ok');
  } catch (e) {
    log('Failed to reset config: ' + e, 'error');
  }
}

export async function restartServer() {
  if (!confirm('Restart the server? You will be disconnected for a few seconds.')) return;
  try {
    await api('POST', '/api/restart');
    log('Restarting server... please wait.', 'ok');
  } catch (e) {
    log('Restart request failed: ' + e, 'error');
  }
}


