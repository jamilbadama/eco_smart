let markersChart = null;
let riskGauge = null;
let monitorSocket = null;
let currentSessionSummary = null; // Store current summary for PHQ updates
let videoStream = null;
let audioContext = null;
let audioWorklet = null;
let monitorInterval = null;

// Initialize charts
function initCharts() {
    const ctxMarkers = document.getElementById('markers-chart').getContext('2d');
    markersChart = new Chart(ctxMarkers, {
        type: 'bar',
        data: {
            labels: ['Movement', 'Speech Sentiment', 'Audio Energy'],
            datasets: [{
                label: 'Normalized Markers',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(0, 210, 255, 0.5)',
                    'rgba(0, 255, 136, 0.5)',
                    'rgba(255, 180, 0, 0.5)'
                ],
                borderColor: [
                    '#00d2ff',
                    '#00ff88',
                    '#ffb400'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    const ctxGauge = document.getElementById('risk-gauge').getContext('2d');
    riskGauge = new Chart(ctxGauge, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [0, 1],
                backgroundColor: ['#00d2ff', 'rgba(255, 255, 255, 0.05)'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270,
                cutout: '80%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            }
        }
    });
}

// Update UI with clinical data
function updateUI(data) {
    console.log("DEBUG: updateUI called with payload:", data);

    try {
        // Support both old and new schema
        let prediction = data.clinical_prediction;
        let groundTruth = data.ground_truth;
        let markers = data.markers;

        if (data.analysis) {
            console.log("DEBUG: Detected new Eco-SMART schema");
            const analysis = data.analysis;
            prediction = {
                depression_risk_probability: analysis.evidence.primary_signals.binary_prob,
                risk_level: analysis.interpretation.risk_level,
                phq8_score_predicted: analysis.evidence.primary_signals.score
            };
            groundTruth = {
                phq8_score: analysis.evidence.clinical_grounding.phq8 || "Hidden (Test Set)"
            };
            markers = {
                speech: { sentiment_positivity: analysis.markers.text.sentiment },
                movement: { mean_intensity: analysis.markers.video.movement_mean },
                audio: { energy: analysis.markers.audio.energy_mean || analysis.markers.audio.activity_mean || 0 }
            };
        }

        console.log("DEBUG: Normalized Data - Prediction:", prediction);
        console.log("DEBUG: Normalized Data - GroundTruth:", groundTruth);
        console.log("DEBUG: Normalized Data - Markers:", markers);

        if (!prediction) {
            throw new Error("Missing prediction data in payload");
        }

        // Update risk probability
        if (riskGauge) {
            const prob = prediction.depression_risk_probability;
            riskGauge.data.datasets[0].data = [prob, 1 - prob];

            const riskColor = prob > 0.6 ? '#ff4757' : (prob > 0.4 ? '#ffb400' : '#00d2ff');
            riskGauge.data.datasets[0].backgroundColor[0] = riskColor;
            riskGauge.update();

            document.getElementById('risk-value').innerText = `${Math.round(prob * 100)}%`;
            document.getElementById('risk-value').style.color = riskColor;
            document.getElementById('risk-label').innerText = `Risk Level: ${prediction.risk_level}`;
        }

        // Update PHQ Score
        const phq = prediction.phq8_score_predicted;
        const phqActual = groundTruth.phq8_score;

        if (document.getElementById('phq-score')) {
            document.getElementById('phq-score').innerText = phq.toFixed(1);
            const progress = (phq / 24) * 100;
            const progressEl = document.getElementById('phq-progress');
            if (progressEl) progressEl.style.width = `${progress}%`;
            const labelEl = document.getElementById('phq-label');
            if (labelEl) labelEl.innerText = `Clinical Context: score ${phqActual}`;
        }

        // Update Markers Chart
        if (markersChart) {
            let sentiment = 0.5;
            if (markers && markers.speech) {
                sentiment = markers.speech.sentiment_positivity;
                if (sentiment < 0) sentiment = (sentiment + 1) / 2;
            }

            const movement = markers && markers.movement ? Math.min(markers.movement.mean_intensity * 50, 1) : 0;
            const energy = markers && markers.audio ? markers.audio.energy : 0;

            console.log("DEBUG: Filling Chart with:", [movement, sentiment, energy]);
            markersChart.data.datasets[0].data = [movement, sentiment, energy];
            markersChart.update();
        }

        // Update Emotion Badge & Dashboard Card if present (Real-time flow)
        const emotionData = data.emotion || (data.analysis ? data.analysis.emotion : null);
        console.log("DEBUG: Processing Emotion Data:", emotionData);

        if (emotionData) {
            const labelStr = emotionData.label.charAt(0).toUpperCase() + emotionData.label.slice(1);

            // 1. Monitor Badge
            const badge = document.getElementById('emotion-badge');
            const label = document.getElementById('emotion-label');
            if (badge && label) {
                badge.classList.remove('hidden');
                label.innerText = labelStr;

                const emotion = emotionData.label.toLowerCase();
                if (emotion === 'happy') badge.style.background = 'rgba(0, 255, 136, 0.8)';
                else if (emotion === 'surprise') badge.style.background = 'rgba(255, 206, 86, 0.8)';
                else if (emotion === 'sad' || emotion === 'angry' || emotion === 'fear' || emotion === 'disgust') badge.style.background = 'rgba(255, 71, 87, 0.8)';
                else badge.style.background = 'rgba(211, 211, 211, 0.8)'; // neutral
            }

            // 2. Dashboard Card
            const dashLabel = document.getElementById('dash-emotion-label');
            const dashIcon = document.getElementById('emotion-icon');
            const dashConfFill = document.getElementById('emotion-confidence-fill');
            const dashConfLabel = document.getElementById('emotion-confidence-label');

            if (dashLabel) dashLabel.innerText = labelStr;
            if (dashConfFill) dashConfFill.style.width = `${Math.round(emotionData.score * 100)}%`;
            if (dashConfLabel) dashConfLabel.innerText = `Confidence: ${Math.round(emotionData.score * 100)}%`;

            if (dashIcon) {
                const emotion = emotionData.label.toLowerCase();
                dashIcon.className = 'ph-fill';
                if (emotion === 'happy') dashIcon.classList.add('ph-smiley');
                else if (emotion === 'sad') dashIcon.classList.add('ph-smiley-sad');
                else if (emotion === 'angry') dashIcon.classList.add('ph-smiley-angry');
                else if (emotion === 'surprise') dashIcon.classList.add('ph-smiley-wink');
                else if (emotion === 'fear') dashIcon.classList.add('ph-smiley-nervous');
                else if (emotion === 'disgust') dashIcon.classList.add('ph-smiley-meh');
                else dashIcon.classList.add('ph-smiley-blank');
            }
        }

        console.log("DEBUG: UI Update successful");

    } catch (err) {
        console.error("CRITICAL: updateUI failed:", err);
        addMessage('system', `_Warning: Failed to update clinical dashboard (${err.message}). Check console for details._`);
    }
}

// Handle Analysis
async function runAnalysis(query) {
    const loader = document.getElementById('loader');
    loader.classList.remove('hidden');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        const result = await response.json();
        console.log("Analysis result:", result);

        if (result.status === 'success') {
            addMessage('system', result.report);
            if (result.data) {
                updateUI(result.data);
            } else {
                addMessage('system', "_Note: Behavioral benchmarks could not be parsed for this analysis._");
            }
        } else {
            addMessage('system', `Error: ${result.detail}`);
        }
    } catch (err) {
        addMessage('system', `Network Error: ${err.message}`);
    } finally {
        loader.classList.add('hidden');
    }
}

// UI Helpers
function addMessage(role, content) {
    const chat = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    // Basic Markdown to HTML conversion for the agent's report
    let htmlContent = content
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/### (.*?)(?:<br>|$)/g, '<h4>$1</h4>')
        .replace(/^- (.*?)(?:<br>|$)/gm, '<li>$1</li>');

    msgDiv.innerHTML = `<p>${htmlContent}</p>`;
    chat.appendChild(msgDiv);
    chat.scrollTop = chat.scrollHeight;
}

// Event Listeners
document.getElementById('analyze-btn').addEventListener('click', () => {
    const id = document.getElementById('patient-id-input').value.trim();
    if (id) {
        const query = `Analyze patient ${id} and explain the risk level.`;
        addMessage('user', query);
        runAnalysis(query);
    }
});

document.getElementById('send-query-btn').addEventListener('click', () => {
    const query = document.getElementById('ai-query').value.trim();
    if (query) {
        addMessage('user', query);
        document.getElementById('ai-query').value = '';
        runAnalysis(query);
    }
});

document.getElementById('ai-query').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('send-query-btn').click();
    }
});

// Real-time Monitoring Logic
async function startRealtimeMonitoring() {
    console.log("DEBUG: Starting real-time monitoring...");
    const monitorOverlay = document.getElementById('monitor-overlay');
    const startBtn = document.getElementById('start-monitor-btn');
    const stopBtn = document.getElementById('stop-monitor-btn');
    const statusDot = document.getElementById('connection-status');
    const video = document.getElementById('webcam-preview');

    try {
        // 1. Request Media Permissions
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 480, height: 270, frameRate: 15 },
            audio: true
        });
        video.srcObject = videoStream;

        // 2. Setup WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        monitorSocket = new WebSocket(`${protocol}//${window.location.host}/ws/monitor`);

        monitorSocket.onopen = () => {
            console.log("DEBUG: WebSocket connected");
            statusDot.classList.remove('inactive');
            statusDot.classList.add('active');
            monitorOverlay.classList.remove('hidden');
            startBtn.classList.add('hidden');
            stopBtn.classList.remove('hidden');

            // Start Capture Loop
            startCaptureLoop();
        };

        monitorSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.analysis) {
                updateUI(data);
            } else if (data.status === 'saved') {
                console.log("DEBUG: Session saved successfully:", data.session_id);
                addMessage('system', `_Success: Session ${data.session_id} has been saved to the database. Displaying findings..._`);

                if (data.summary) {
                    currentSessionSummary = data.summary;
                    displaySessionFindings(data.summary);
                }

                // Final cleanup after save confirmation
                performStopCleanup();
            } else if (data.status === 'error') {
                addMessage('system', `_Warning: Session might not have saved correctly (${data.message})._`);
                performStopCleanup();
            }
        };

        monitorSocket.onclose = () => {
            console.log("DEBUG: WebSocket closed");
            stopRealtimeMonitoring();
        };

        monitorSocket.onerror = (err) => {
            console.error("DEBUG: WebSocket error:", err);
            addMessage('system', "_Error: WebSocket connection failed._");
            stopRealtimeMonitoring();
        };

    } catch (err) {
        console.error("DEBUG: Media access failed:", err);
        addMessage('system', `_Error: Could not access webcam or microphone (${err.message})_`);
    }
}

function stopRealtimeMonitoring() {
    console.log("DEBUG: Triggering session save and stop...");

    if (monitorSocket && monitorSocket.readyState === WebSocket.OPEN) {
        // Send save command before closing
        const patientId = document.getElementById('session-id-input')?.value || "LIVE_SESSION";
        monitorSocket.send(JSON.stringify({ command: "save", session_id: patientId }));
        // Cleanup will happen after receiving 'saved' status or after a short delay
        setTimeout(() => {
            if (monitorSocket) performStopCleanup();
        }, 1500);
    } else {
        performStopCleanup();
    }
}

function performStopCleanup() {
    console.log("DEBUG: Performing final monitor cleanup...");
    if (monitorInterval) clearInterval(monitorInterval);
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    if (monitorSocket) {
        monitorSocket.onclose = null; // Prevent recursion
        monitorSocket.close();
        monitorSocket = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    document.getElementById('monitor-overlay').classList.add('hidden');
    document.getElementById('emotion-badge').classList.add('hidden');
    document.getElementById('start-monitor-btn').classList.remove('hidden');
    document.getElementById('stop-monitor-btn').classList.add('hidden');
    document.getElementById('connection-status').classList.remove('active');
    document.getElementById('connection-status').classList.add('inactive');
    document.getElementById('webcam-preview').srcObject = null;
}

function displaySessionFindings(summary) {
    console.log("DEBUG: Displaying findings for session:", summary.session_id);

    const analysis = summary.analysis;
    const findingsModal = document.getElementById('findings-modal');

    // 1. Risk Overview
    const riskLevel = analysis.interpretation.risk_level;
    const riskProb = Math.round(analysis.evidence.primary_signals.binary_prob * 100);
    const riskEl = document.getElementById('findings-risk-level');

    riskEl.innerText = riskLevel;
    document.getElementById('findings-risk-prob').innerText = `${riskProb}%`;
    document.getElementById('findings-risk-summary').innerText = analysis.interpretation.summary;

    // Set risk color
    if (riskLevel.toLowerCase().includes('severe')) riskEl.style.color = 'var(--accent-danger)';
    else if (riskLevel.toLowerCase().includes('moderate')) riskEl.style.color = 'var(--accent-warning)';
    else riskEl.style.color = 'var(--accent-emerald)';

    // 2. Patterns (MMFformer)
    const patterns = (analysis.evidence.fusion_assessment && analysis.evidence.fusion_assessment.detected_patterns) || [];
    const patternsList = document.getElementById('findings-patterns-list');
    patternsList.innerHTML = '';

    if (patterns.length > 0) {
        patterns.forEach(p => {
            const li = document.createElement('li');
            li.innerText = p;
            patternsList.appendChild(li);
        });
    } else {
        patternsList.innerHTML = '<li style="background:transparent; border:none; opacity:0.5;">No specific high-level patterns detected</li>';
    }

    // 3. Affect
    const dominantEmotion = analysis.markers.dominant_emotion || "neutral";
    document.getElementById('findings-emotion-label').innerText = dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1);

    const emoIcon = document.getElementById('findings-emotion-icon');
    emoIcon.className = 'ph-fill';
    const emotionLower = dominantEmotion.toLowerCase();
    if (emotionLower === 'happy') emoIcon.classList.add('ph-smiley');
    else if (emotionLower === 'sad') emoIcon.classList.add('ph-smiley-sad');
    else if (emotionLower === 'angry') emoIcon.classList.add('ph-smiley-angry');
    else emoIcon.classList.add('ph-smiley-blank');

    // 4. Agreement Points
    const agreements = analysis.evidence.agreement_points || [];
    const agreementList = document.getElementById('findings-agreement-list');
    agreementList.innerHTML = '';

    agreements.forEach(a => {
        const li = document.createElement('li');
        li.innerText = a;
        agreementList.appendChild(li);
    });

    // Show Modal
    findingsModal.classList.remove('hidden');
}

function startCaptureLoop() {
    const video = document.getElementById('webcam-preview');
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 320; // Reduced for performance
    canvas.height = 180;

    // Video Loop (every 200ms approx 5fps)
    monitorInterval = setInterval(() => {
        if (monitorSocket && monitorSocket.readyState === WebSocket.OPEN) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const frame = canvas.toDataURL('image/jpeg', 0.5);
            monitorSocket.send(JSON.stringify({ video: frame }));
        }
    }, 200);

    // Audio Capture
    setupAudioCapture();
}

async function setupAudioCapture() {
    try {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(videoStream);
        const processor = audioContext.createScriptProcessor(4096, 1, 1);

        source.connect(processor);
        processor.connect(audioContext.destination);

        processor.onaudioprocess = (e) => {
            if (monitorSocket && monitorSocket.readyState === WebSocket.OPEN) {
                const inputData = e.inputBuffer.getChannelData(0);
                // Convert Float32Array to plain array for JSON transport
                // Truncate to save bandwidth/processing
                const chunk = Array.from(inputData).filter((_, i) => i % 4 === 0);
                monitorSocket.send(JSON.stringify({ audio: chunk }));
            }
        };
    } catch (err) {
        console.warn("DEBUG: Audio context failed:", err);
    }
}

// PHQ-8 Clinical Questions
const PHQ8_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual"
];

const PHQ_OPTIONS = [
    { label: "Not at all", value: 0 },
    { label: "Several days", value: 1 },
    { label: "More than half the days", value: 2 },
    { label: "Nearly every day", value: 3 }
];

let phqAnswers = new Array(8).fill(null);

function showPHQQuestionnaire() {
    const container = document.getElementById('phq-questions-container');
    container.innerHTML = '';
    phqAnswers = new Array(8).fill(null);
    document.getElementById('phq-total-display').innerText = '0';

    PHQ8_QUESTIONS.forEach((q, idx) => {
        const item = document.createElement('div');
        item.className = 'phq-question-item';
        item.innerHTML = `
            <p>${idx + 1}. ${q}</p>
            <div class="phq-options" data-qid="${idx}">
                ${PHQ_OPTIONS.map(opt => `
                    <div class="phq-option" data-value="${opt.value}">${opt.label}</div>
                `).join('')}
            </div>
        `;
        container.appendChild(item);
    });

    // Add click listeners for options
    container.querySelectorAll('.phq-option').forEach(opt => {
        opt.addEventListener('click', (e) => {
            const parent = e.target.parentElement;
            const qidx = parseInt(parent.dataset.qid);
            const val = parseInt(e.target.dataset.value);

            // UI Toggle
            parent.querySelectorAll('.phq-option').forEach(o => o.classList.remove('selected'));
            e.target.classList.add('selected');

            // Update answer
            phqAnswers[qidx] = val;
            calculatePHQTotal();
        });
    });

    document.getElementById('findings-modal').classList.add('hidden');
    document.getElementById('phq-modal').classList.remove('hidden');
}

function calculatePHQTotal() {
    const total = phqAnswers.reduce((sum, val) => sum + (val === null ? 0 : val), 0);
    document.getElementById('phq-total-display').innerText = total;
}

async function submitPHQGroundTruth() {
    if (phqAnswers.includes(null)) {
        alert("Please answer all questions before submitting.");
        return;
    }

    if (!currentSessionSummary) return;

    const totalScore = phqAnswers.reduce((a, b) => a + b, 0);
    const sessionId = currentSessionSummary.session_id;

    try {
        const response = await fetch('/api/session/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                phq8_score: totalScore
            })
        });

        const result = await response.json();
        if (result.status === 'success') {
            addMessage('system', `_Success: Ground truth clinical score (${totalScore}) updated for session ${sessionId}._`);
            document.getElementById('phq-modal').classList.add('hidden');
        } else {
            addMessage('system', `_Error: Failed to update ground truth (${result.message})._`);
        }
    } catch (err) {
        addMessage('system', `_Network Error: ${err.message}_`);
    }
}

// Start
window.onload = () => {
    initCharts();

    // Bind Real-time controls
    document.getElementById('start-monitor-btn').addEventListener('click', startRealtimeMonitoring);
    document.getElementById('stop-monitor-btn').addEventListener('click', stopRealtimeMonitoring);

    // Modal controls
    document.getElementById('close-findings-btn').addEventListener('click', () => {
        document.getElementById('findings-modal').classList.add('hidden');
    });
    document.getElementById('findings-ok-btn').addEventListener('click', () => {
        document.getElementById('findings-modal').classList.add('hidden');
    });
    document.getElementById('findings-phq-btn').addEventListener('click', showPHQQuestionnaire);

    document.getElementById('close-phq-btn').addEventListener('click', () => {
        document.getElementById('phq-modal').classList.add('hidden');
    });
    document.getElementById('submit-phq-btn').addEventListener('click', submitPHQGroundTruth);
};
