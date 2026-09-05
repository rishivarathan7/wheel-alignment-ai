/**
 * AI Wheel Alignment Monitoring System - Web Dashboard Controller
 * Asynchronous Communication with FastAPI Backend & Integrated Alert System
 */

// Global State
let pollingInterval = null;
let isMonitoring = false;
const seenAlertKeys = new Set();

/**
 * Add timestamped event to the event log console
 * @param {string} message - Event message text
 * @param {string} level - Log level ('info', 'warning', 'error', 'success')
 */
function addLog(message, level = 'info') {
    const consoleEl = document.getElementById('eventLogConsole');
    if (!consoleEl) return;

    const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
    const entry = document.createElement('div');
    entry.className = `log-entry ${level}`;
    entry.textContent = `[${ts}] ${message}`;
    consoleEl.appendChild(entry);
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

/**
 * Process escalated alert events from alert_system.py
 * Displays alerts in browser status card and adds timestamped events to event log.
 * Prevents duplicate alerts every frame by tracking seen alert keys.
 */
function processAlertEvents(alerts) {
    if (!alerts || !Array.isArray(alerts) || alerts.length === 0) return;

    alerts.forEach(alert => {
        if (!alert || !alert.timestamp) return;

        const alertKey = `${alert.timestamp}_${alert.track_id}_${alert.state}`;
        if (seenAlertKeys.has(alertKey)) {
            return; // Duplicate frame alert suppression
        }
        seenAlertKeys.add(alertKey);

        if (seenAlertKeys.size > 200) {
            const firstKey = seenAlertKeys.values().next().value;
            seenAlertKeys.delete(firstKey);
        }

        const tid = (alert.track_id !== undefined && alert.track_id !== null) ? alert.track_id : '--';
        const state = alert.state || 'POSSIBLE_MISALIGNMENT';
        const confPct = ((alert.confidence || 0.0) * 100).toFixed(0);
        const camberVal = (alert.camber_proxy >= 0 ? '+' : '') + (alert.camber_proxy || 0.0).toFixed(1);
        const toeVal = (alert.toe_proxy >= 0 ? '+' : '') + (alert.toe_proxy || 0.0).toFixed(1);
        const msgText = alert.message || 'Possible wheel alignment issue detected.';

        // 1. Display alert status in browser card
        updateStatus(state, `${msgText} (Wheel #${tid})`);

        // 2. Add timestamped event to event log console
        const logLevel = (state === 'SEVERE_MISALIGNMENT') ? 'error' : 'warn';
        const logMsg = `ALERT ESCALATED [Wheel #${tid}]: ${state} (${confPct}% Conf | Camber: ${camberVal}°, Toe: ${toeVal}°)`;
        addLog(logMsg, logLevel);
    });
}

/**
 * Update system status banner and display indicators
 * @param {string} statusText - Status title (IDLE, MONITORING, ALIGNED, MISALIGNED, ERROR, etc.)
 * @param {string} warningMsg - Optional descriptive text/warning message
 */
function updateStatus(statusText, warningMsg) {
    const statusBanner = document.getElementById('statusBanner');
    const statusTitle = document.getElementById('statusTitle');
    const warningText = document.getElementById('warningText');
    const streamStatusDot = document.getElementById('streamStatusDot');

    if (statusTitle && statusText) {
        statusTitle.textContent = statusText.toUpperCase();
    }

    if (warningText && warningMsg !== undefined) {
        warningText.textContent = warningMsg;
    }

    if (statusBanner) {
        // Reset status classes
        statusBanner.className = 'status-banner';
        const normStatus = (statusText || '').toUpperCase();

        if (normStatus === 'IDLE') {
            statusBanner.classList.add('status-idle');
            if (streamStatusDot) streamStatusDot.classList.remove('active');
        } else if (normStatus === 'MONITORING') {
            statusBanner.classList.add('status-monitoring');
            if (streamStatusDot) streamStatusDot.classList.add('active');
        } else if (normStatus === 'ALIGNED' || normStatus === 'NORMAL') {
            statusBanner.classList.add('status-normal');
            if (streamStatusDot) streamStatusDot.classList.add('active');
        } else if (normStatus === 'MISALIGNED' || normStatus.includes('MISALIGNMENT')) {
            statusBanner.classList.add('status-severe');
            if (streamStatusDot) streamStatusDot.classList.add('active');
        } else if (normStatus === 'ERROR') {
            statusBanner.classList.add('status-error');
            if (streamStatusDot) streamStatusDot.classList.remove('active');
        } else {
            statusBanner.classList.add('status-idle');
        }
    }
}

/**
 * Start Monitoring Session via POST /start
 */
async function startMonitor() {
    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');
    const videoSourceSelect = document.getElementById('videoSourceSelect');
    const customVideoPath = document.getElementById('customVideoPath');
    const videoFeed = document.getElementById('videoFeed');
    const videoOverlay = document.getElementById('videoOverlay');
    const overlayMessage = document.getElementById('overlayMessage');
    const streamResolution = document.getElementById('streamResolution');

    let source = 0;
    if (videoSourceSelect) {
        if (videoSourceSelect.value === 'custom') {
            source = customVideoPath ? customVideoPath.value.trim() : '';
            if (!source) {
                alert('Please enter a valid custom video file path.');
                return;
            }
        } else {
            source = parseInt(videoSourceSelect.value, 10) || 0;
        }
    }

    addLog(`Requesting monitoring start on source: ${source}...`, 'info');
    if (btnStart) btnStart.disabled = true;
    if (overlayMessage) overlayMessage.textContent = 'Initializing AI pipeline stream...';
    if (videoOverlay) videoOverlay.classList.remove('hidden');

    try {
        const response = await fetch('/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: source })
        });

        const data = await response.json();

        if (response.ok && data.status === 'success') {
            isMonitoring = true;
            seenAlertKeys.clear();
            if (btnStop) btnStop.disabled = false;

            // Start displaying /video_feed
            if (videoFeed) {
                videoFeed.src = `/video_feed?t=${Date.now()}`;
            }
            if (videoOverlay) videoOverlay.classList.add('hidden');
            if (streamResolution) streamResolution.textContent = `Source: ${source} (Active)`;

            // Update status to MONITORING
            updateStatus('MONITORING', 'System actively monitoring camera stream.');

            // Add event to event log
            addLog('Monitoring started successfully.', 'success');
        } else {
            const friendlyErr = data.message || data.detail || 'Camera unavailable';
            throw new Error(friendlyErr);
        }
    } catch (err) {
        const displayMsg = err.message || 'Camera unavailable';
        addLog(`Start Monitor Error: ${displayMsg}`, 'error');
        if (btnStart) btnStart.disabled = false;
        if (videoOverlay) videoOverlay.classList.add('hidden');
        updateStatus('ERROR', displayMsg);
        alert(`${displayMsg}`);
    }
}

/**
 * Stop Monitoring Session via POST /stop
 */
async function stopMonitor() {
    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');
    const videoFeed = document.getElementById('videoFeed');
    const streamResolution = document.getElementById('streamResolution');

    addLog('Stopping AI monitoring stream...', 'info');
    if (btnStop) btnStop.disabled = true;

    try {
        const response = await fetch('/stop', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();

        if (response.ok) {
            isMonitoring = false;
            if (btnStart) btnStart.disabled = false;

            // Stop video display
            if (videoFeed) {
                videoFeed.src = '/static/images/placeholder.svg';
            }
            if (streamResolution) streamResolution.textContent = 'Source: Stopped';

            // Update status to IDLE
            updateStatus('IDLE', 'System ready for real-time monitoring stream.');

            // Add event to event log
            addLog('Monitoring stopped.', 'info');
            resetTelemetryUI();
        } else {
            throw new Error(data.message || data.detail || 'Failed to stop stream.');
        }
    } catch (err) {
        addLog(`Stop Monitor Error: ${err.message}`, 'error');
        if (btnStop) btnStop.disabled = false;
    }
}

/**
 * Reset HUD and telemetry table to idle state
 */
function resetTelemetryUI() {
    const hudFps = document.getElementById('hudFps');
    const hudLatency = document.getElementById('hudLatency');
    const hudDetLat = document.getElementById('hudDetLat');
    const hudMlLat = document.getElementById('hudMlLat');
    const activeTrackCount = document.getElementById('activeTrackCount');
    const telemetryTableBody = document.getElementById('telemetryTableBody');

    seenAlertKeys.clear();

    if (hudFps) hudFps.textContent = '0.0';
    if (hudLatency) hudLatency.textContent = '0.0 ms';
    if (hudDetLat) hudDetLat.textContent = '0.0 ms';
    if (hudMlLat) hudMlLat.textContent = '0.0 ms';
    if (activeTrackCount) activeTrackCount.textContent = '0 Wheels Detected';

    if (telemetryTableBody) {
        telemetryTableBody.innerHTML = `
            <tr class="empty-row">
                <td colspan="6">No active wheels currently tracked</td>
            </tr>
        `;
    }
}

/**
 * Poll system status from GET /status without blocking the browser
 */
async function pollStatus() {
    try {
        const response = await fetch('/status');
        if (!response.ok) return;

        const data = await response.json();
        if (data.status !== 'success') return;

        isMonitoring = data.is_streaming || data.monitoring;

        // Process any escalated alert events from alert_system.py
        if (data.alerts && Array.isArray(data.alerts)) {
            processAlertEvents(data.alerts);
        }

        // Synchronize UI button states if changed externally
        const btnStart = document.getElementById('btnStart');
        const btnStop = document.getElementById('btnStop');
        if (btnStart && btnStop) {
            btnStart.disabled = isMonitoring;
            btnStop.disabled = !isMonitoring;
        }

        // Update performance HUD
        const perf = data.performance || {};
        const hudFps = document.getElementById('hudFps');
        const hudLatency = document.getElementById('hudLatency');
        const hudDetLat = document.getElementById('hudDetLat');
        const hudMlLat = document.getElementById('hudMlLat');
        const hudModelStatus = document.getElementById('hudModelStatus');

        if (hudFps) hudFps.textContent = (perf.fps || 0.0).toFixed(1);
        if (hudLatency) hudLatency.textContent = `${(perf.total_latency_ms || 0.0).toFixed(1)} ms`;
        if (hudDetLat) hudDetLat.textContent = `${(perf.det_latency_ms || 0.0).toFixed(1)} ms`;
        if (hudMlLat) hudMlLat.textContent = `${(perf.ml_latency_ms || 0.0).toFixed(1)} ms`;

        const modelStatusText = data.model_status || perf.model_status || 'Active';
        if (hudModelStatus) hudModelStatus.textContent = modelStatusText;

        // Update wheel telemetry table
        const telemetry = data.telemetry || data.wheels || [];
        const activeTrackCount = document.getElementById('activeTrackCount');
        const telemetryTableBody = document.getElementById('telemetryTableBody');

        if (activeTrackCount) {
            activeTrackCount.textContent = `${telemetry.length} Wheels Detected`;
        }

        if (telemetryTableBody) {
            if (telemetry.length === 0) {
                const emptyMsg = data.status_note || 'No wheel detected';
                telemetryTableBody.innerHTML = `
                    <tr class="empty-row">
                        <td colspan="6">${emptyMsg}</td>
                    </tr>
                `;
                if (isMonitoring && seenAlertKeys.size === 0) {
                    updateStatus('MONITORING', emptyMsg);
                }
            } else {

                let html = '';
                let highestSeverity = 'ALIGNED';

                telemetry.forEach(rec => {
                    const tid = (rec.track_id !== undefined && rec.track_id !== null) ? rec.track_id : (rec.wheel_id !== undefined ? rec.wheel_id : '--');
                    const status = rec.status || 'NORMAL';
                    const conf = ((rec.confidence || 0.0) * 100).toFixed(0);
                    const camber = (rec.camber_proxy >= 0 ? '+' : (rec.camber >= 0 ? '+' : '')) + (rec.camber_proxy !== undefined ? rec.camber_proxy : rec.camber || 0.0).toFixed(1);
                    const toe = (rec.toe_proxy >= 0 ? '+' : (rec.toe >= 0 ? '+' : '')) + (rec.toe_proxy !== undefined ? rec.toe_proxy : rec.toe || 0.0).toFixed(1);

                    let spdVal = 0.0;
                    if (rec.speed !== undefined) spdVal = rec.speed;
                    else if (rec.motion && rec.motion.velocity_magnitude !== undefined) spdVal = rec.motion.velocity_magnitude;
                    const spd = spdVal.toFixed(0);

                    let pillClass = 'normal';
                    if (status === 'POSSIBLE_MISALIGNMENT') pillClass = 'possible';
                    if (status === 'SEVERE_MISALIGNMENT') pillClass = 'severe';
                    if (status === 'INSUFFICIENT_DATA') pillClass = 'data';

                    if (status === 'SEVERE_MISALIGNMENT') highestSeverity = 'MISALIGNMENT';
                    else if (status === 'POSSIBLE_MISALIGNMENT' && highestSeverity !== 'MISALIGNMENT') highestSeverity = 'MISALIGNMENT';

                    html += `
                        <tr>
                            <td>#${tid}</td>
                            <td><span class="status-pill ${pillClass}">${status.replace('_MISALIGNMENT', '')}</span></td>
                            <td>${conf}%</td>
                            <td>${camber}°</td>
                            <td>${toe}°</td>
                            <td>${spd}px/s</td>
                        </tr>
                    `;
                });

                telemetryTableBody.innerHTML = html;

                if (isMonitoring && seenAlertKeys.size === 0) {
                    if (highestSeverity === 'MISALIGNMENT') {
                        updateStatus('MISALIGNED', 'ALERT: Wheel misalignment anomaly detected.');
                    } else {
                        updateStatus('ALIGNED', 'Wheel alignment operating within nominal parameters.');
                    }
                }
            }
        }
    } catch (err) {
        // Silently ignore transient network poll errors
    }
}

// Global initialization & Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    const btnStart = document.getElementById('btnStart');
    const btnStop = document.getElementById('btnStop');
    const btnClearLog = document.getElementById('btnClearLog');
    const videoSourceSelect = document.getElementById('videoSourceSelect');
    const customSourceInputGroup = document.getElementById('customSourceInputGroup');
    const customVideoPath = document.getElementById('customVideoPath');
    const btnSettings = document.getElementById('btnSettings');
    const settingsModal = document.getElementById('settingsModal');
    const btnCloseSettings = document.getElementById('btnCloseSettings');
    const btnCancelSettings = document.getElementById('btnCancelSettings');
    const btnSaveSettings = document.getElementById('btnSaveSettings');

    // Settings Modal Form Controls
    const cfgCameraSelect = document.getElementById('cfgCameraSelect');
    const cfgCustomSourceGroup = document.getElementById('cfgCustomSourceGroup');
    const cfgCustomSource = document.getElementById('cfgCustomSource');
    const cfgCamberLimit = document.getElementById('cfgCamberLimit');
    const cfgToeLimit = document.getElementById('cfgToeLimit');
    const cfgDetConf = document.getElementById('cfgDetConf');
    const valDetConf = document.getElementById('valDetConf');
    const cfgPredConf = document.getElementById('cfgPredConf');
    const valPredConf = document.getElementById('valPredConf');
    const cfgConsecutive = document.getElementById('cfgConsecutive');
    const cfgCooldown = document.getElementById('cfgCooldown');
    const cfgEnableAudio = document.getElementById('cfgEnableAudio');
    const cfgDebugMode = document.getElementById('cfgDebugMode');

    if (btnStart) {
        btnStart.addEventListener('click', startMonitor);
    }

    if (btnStop) {
        btnStop.addEventListener('click', stopMonitor);
    }

    if (btnClearLog) {
        btnClearLog.addEventListener('click', () => {
            const consoleEl = document.getElementById('eventLogConsole');
            if (consoleEl) {
                consoleEl.innerHTML = '';
                addLog('Log cleared.', 'info');
            }
        });
    }

    if (videoSourceSelect) {
        videoSourceSelect.addEventListener('change', () => {
            if (customSourceInputGroup) {
                if (videoSourceSelect.value === 'custom') {
                    customSourceInputGroup.classList.remove('hidden');
                } else {
                    customSourceInputGroup.classList.add('hidden');
                }
            }
        });
    }

    if (cfgCameraSelect) {
        cfgCameraSelect.addEventListener('change', () => {
            if (cfgCustomSourceGroup) {
                if (cfgCameraSelect.value === 'custom') {
                    cfgCustomSourceGroup.classList.remove('hidden');
                } else {
                    cfgCustomSourceGroup.classList.add('hidden');
                }
            }
        });
    }

    // Modal Load Handler
    if (btnSettings) {
        btnSettings.addEventListener('click', async () => {
            try {
                const resp = await fetch('/api/config');
                const cfg = await resp.json();

                const srcVal = cfg['video.source'];
                if (cfgCameraSelect) {
                    if (srcVal === 0 || srcVal === 1 || srcVal === 2 || srcVal === '0' || srcVal === '1' || srcVal === '2') {
                        cfgCameraSelect.value = String(srcVal);
                        if (cfgCustomSourceGroup) cfgCustomSourceGroup.classList.add('hidden');
                    } else if (srcVal !== undefined && srcVal !== null) {
                        cfgCameraSelect.value = 'custom';
                        if (cfgCustomSource) cfgCustomSource.value = srcVal;
                        if (cfgCustomSourceGroup) cfgCustomSourceGroup.classList.remove('hidden');
                    }
                }

                if (cfgCamberLimit) cfgCamberLimit.value = cfg['prediction.camber_angle_limit'] || 3.0;
                if (cfgToeLimit) cfgToeLimit.value = cfg['prediction.toe_angle_limit'] || 2.0;

                if (cfgDetConf) {
                    cfgDetConf.value = cfg['detection.confidence_threshold'] || 0.50;
                    if (valDetConf) valDetConf.textContent = parseFloat(cfgDetConf.value).toFixed(2);
                }
                if (cfgPredConf) {
                    cfgPredConf.value = cfg['alert.min_confidence'] || 0.60;
                    if (valPredConf) valPredConf.textContent = parseFloat(cfgPredConf.value).toFixed(2);
                }
                if (cfgConsecutive) cfgConsecutive.value = cfg['alert.consecutive_threshold'] || 3;
                if (cfgCooldown) cfgCooldown.value = cfg['alert.cooldown_seconds'] || 3.0;
                if (cfgEnableAudio) cfgEnableAudio.checked = !!cfg['alert.enable_audio'];
                if (cfgDebugMode) cfgDebugMode.checked = !!cfg['pipeline.debug_mode'];

                if (settingsModal) settingsModal.classList.remove('hidden');
            } catch (e) {
                addLog(`Failed to load config: ${e.message}`, 'error');
            }
        });
    }

    if (cfgDetConf && valDetConf) {
        cfgDetConf.addEventListener('input', () => {
            valDetConf.textContent = parseFloat(cfgDetConf.value).toFixed(2);
        });
    }

    if (cfgPredConf && valPredConf) {
        cfgPredConf.addEventListener('input', () => {
            valPredConf.textContent = parseFloat(cfgPredConf.value).toFixed(2);
        });
    }

    const closeModal = () => {
        if (settingsModal) settingsModal.classList.add('hidden');
    };
    if (btnCloseSettings) btnCloseSettings.addEventListener('click', closeModal);
    if (btnCancelSettings) btnCancelSettings.addEventListener('click', closeModal);

    // Modal Save Handler
    if (btnSaveSettings) {
        btnSaveSettings.addEventListener('click', async () => {
            const camberVal = parseFloat(cfgCamberLimit ? cfgCamberLimit.value : 3.0);
            const toeVal = parseFloat(cfgToeLimit ? cfgToeLimit.value : 2.0);
            const detConfVal = parseFloat(cfgDetConf ? cfgDetConf.value : 0.50);
            const predConfVal = parseFloat(cfgPredConf ? cfgPredConf.value : 0.60);
            const consecutiveVal = parseInt(cfgConsecutive ? cfgConsecutive.value : 3, 10);
            const cooldownVal = parseFloat(cfgCooldown ? cfgCooldown.value : 3.0);

            // Client-side numeric validation
            if (isNaN(camberVal) || camberVal < 0.1 || camberVal > 30.0) {
                alert('Please enter a valid Camber Threshold between 0.1° and 30.0°.');
                return;
            }
            if (isNaN(toeVal) || toeVal < 0.1 || toeVal > 20.0) {
                alert('Please enter a valid Toe Threshold between 0.1° and 20.0°.');
                return;
            }
            if (isNaN(detConfVal) || detConfVal < 0.05 || detConfVal > 0.95) {
                alert('Please enter a valid Detection Confidence between 0.05 and 0.95.');
                return;
            }
            if (isNaN(predConfVal) || predConfVal < 0.05 || predConfVal > 1.0) {
                alert('Please enter a valid Minimum Prediction Confidence between 0.05 and 1.0.');
                return;
            }
            if (isNaN(consecutiveVal) || consecutiveVal < 1 || consecutiveVal > 50) {
                alert('Please enter a valid Consecutive Detection Threshold between 1 and 50.');
                return;
            }
            if (isNaN(cooldownVal) || cooldownVal < 0.1 || cooldownVal > 120.0) {
                alert('Please enter a valid Alert Cooldown between 0.1 and 120.0 seconds.');
                return;
            }

            let cameraSource = 0;
            if (cfgCameraSelect) {
                if (cfgCameraSelect.value === 'custom') {
                    cameraSource = cfgCustomSource ? cfgCustomSource.value.trim() : '';
                    if (!cameraSource) {
                        alert('Please enter a valid custom video file path.');
                        return;
                    }
                } else {
                    cameraSource = parseInt(cfgCameraSelect.value, 10);
                }
            }

            const payload = {
                'video.source': cameraSource,
                'prediction.camber_angle_limit': camberVal,
                'prediction.toe_angle_limit': toeVal,
                'detection.confidence_threshold': detConfVal,
                'alert.min_confidence': predConfVal,
                'alert.consecutive_threshold': consecutiveVal,
                'alert.cooldown_seconds': cooldownVal,
                'alert.enable_audio': cfgEnableAudio ? cfgEnableAudio.checked : false,
                'pipeline.debug_mode': cfgDebugMode ? cfgDebugMode.checked : false
            };

            try {
                const resp = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const res = await resp.json();

                if (resp.ok && res.status === 'success') {
                    addLog('Pipeline settings saved successfully.', 'success');
                    if (videoSourceSelect) {
                        if (typeof cameraSource === 'number') {
                            videoSourceSelect.value = String(cameraSource);
                            if (customSourceInputGroup) customSourceInputGroup.classList.add('hidden');
                        } else {
                            videoSourceSelect.value = 'custom';
                            if (customVideoPath) customVideoPath.value = cameraSource;
                            if (customSourceInputGroup) customSourceInputGroup.classList.remove('hidden');
                        }
                    }
                    closeModal();
                } else {
                    throw new Error(res.detail || res.message || 'Save failed.');
                }
            } catch (e) {
                alert(`Error saving settings: ${e.message}`);
            }
        });
    }

    addLog('Web Monitoring Dashboard initialized.', 'info');

    // Start non-blocking periodic polling of GET /status (every 1000ms)
    pollingInterval = setInterval(pollStatus, 1000);
});

// Export functions to global scope (window) for external accessibility
window.startMonitor = startMonitor;
window.stopMonitor = stopMonitor;
window.updateStatus = updateStatus;
window.addLog = addLog;
window.pollStatus = pollStatus;
window.processAlertEvents = processAlertEvents;
