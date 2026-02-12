/**
 * Transport Policing Dashboard - Core Logic
 */

const App = {
    state: {
        token: localStorage.getItem('token') || null,
        user: null,
        tempToken: null, // For 2FA step
        view: 'dashboard',
        refreshInterval: null
    },

    // --- API Client ---
    async api(endpoint, method = 'GET', body = null, token = true) {
        const headers = { 'Content-Type': 'application/json' };
        if (token && this.state.token) headers['Authorization'] = `Bearer ${this.state.token}`;

        console.log(`[API] ${method} ${endpoint}`, body || '');
        try {
            const res = await fetch(endpoint, {
                method,
                headers,
                body: body ? JSON.stringify(body) : null
            });
            const text = await res.text();
            let data;
            try { data = JSON.parse(text); } catch { data = text; }

            if (!res.ok) {
                console.error(`[API] ${method} ${endpoint} → ${res.status}`, data);
                if (res.status === 401 && token) this.logout();
                throw new Error(data.detail || data || 'API Error');
            }
            console.log(`[API] ${method} ${endpoint} → 200`, data);
            return data;
        } catch (e) {
            this.toast(e.message, 'error');
            throw e;
        }
    },

    // --- Authentication ---
    async login(username, password) {
        try {
            const data = await this.api('/auth/login', 'POST', { username, password }, false);
            // 2FA disabled for testing — go straight in
            this.state.token = data.access_token;
            localStorage.setItem('token', this.state.token);
            this.init();
        } catch (e) { console.error(e); }
    },

    async verify2FA(code) {
        try {
            // Temporarily set token for this request if needed, or send as body. 
            // The protocol expects 'token' in the body for /verify-2fa based on original code.
            // Wait, original routes.py /verify-2fa takes {token: "jwt", code: "otp"}.

            const payload = { token: this.state.tempToken, code };
            await this.api('/auth/verify-2fa', 'POST', payload, false);

            // Success
            this.state.token = this.state.tempToken;
            localStorage.setItem('token', this.state.token);
            this.state.tempToken = null;
            this.init();
        } catch (e) { console.error(e); }
    },

    async register(username, password, role) {
        try {
            const data = await this.api('/auth/register', 'POST', { username, password, role }, false);
            document.getElementById('reg-out').innerHTML =
                `<div class="alert badge-safe">Registered! Secret: ${data.twofa_secret}</div>`;
        } catch (e) {
            document.getElementById('reg-out').innerHTML = `<div class="badge badge-danger">${e.message}</div>`;
        }
    },

    logout() {
        this.state.token = null;
        this.state.user = null;
        localStorage.removeItem('token');
        this.renderAuth();
    },

    // --- Views ---
    init() {
        if (!this.state.token) {
            this.renderAuth();
        } else {
            document.getElementById('auth-view').classList.add('hidden');
            document.getElementById('app-view').classList.remove('hidden');
            this.setupRouter();
            this.startPolling();
            this.setupROI();
        }
    },

    renderAuth() {
        document.getElementById('auth-view').classList.remove('hidden');
        document.getElementById('app-view').classList.add('hidden');
        this.showAuthStep('login');
    },

    showAuthStep(step) {
        ['login-step', '2fa-step', 'register-step'].forEach(id =>
            document.getElementById(id).classList.add('hidden'));
        document.getElementById(`${step}-step`).classList.remove('hidden');
    },

    setupRouter() {
        window.onhashchange = () => this.handleRoute();
        this.handleRoute(); // Initial
    },

    handleRoute() {
        const hash = window.location.hash.slice(1) || 'dashboard';
        this.state.view = hash;

        // Update Sidebar
        document.querySelectorAll('.nav-item').forEach(el => {
            el.classList.toggle('active', el.getAttribute('href') === `#${hash}`);
        });

        // Show View
        document.querySelectorAll('.view-section').forEach(el => el.classList.add('hidden'));
        const activeView = document.getElementById(`view-${hash}`);
        if (activeView) activeView.classList.remove('hidden');

        // Refresh Data
        if (hash === 'dashboard') { this.loadAlerts(); this.loadStats(); }
        if (hash === 'trust') this.loadTrust();
        if (hash === 'ingest') this.loadModelStatus();
        if (hash !== 'live') this.stopStream();
    },

    // --- Data Loading ---
    async loadAlerts() {
        const container = document.getElementById('alerts-feed');
        if (!container) return;
        try {
            const alerts = await this.api('/alerts');
            container.innerHTML = alerts.map(a => `
                <div class="card flex justify-between items-center" style="margin-bottom: 0.5rem; padding: 1rem;">
                    <div>
                        <div class="flex items-center gap-2">
                            <span class="badge badge-${this.getSeverityClass(a.severity)}">${a.severity}</span>
                            <span style="font-weight: 500;">Bus ${a.bus_id}</span>
                            <small>${new Date(a.created_at).toLocaleTimeString()}</small>
                        </div>
                        <div style="margin-top:0.25rem; font-size: 0.9rem; color: var(--text-muted)">Route: ${a.route_id} | Status: ${a.status}</div>
                    </div>
                    ${a.status !== 'acknowledged' ? `<button onclick="App.ackAlert(${a.id})" class="btn btn-sm btn-outline">Ack</button>` : '<i style="color:var(--safe)">✓</i>'}
                </div>
            `).join('');

            // Update Stats
            document.getElementById('stat-alerts').innerText = alerts.filter(a => a.status === 'open').length;
        } catch (e) {
            container.innerHTML = `<div class="text-center text-muted">Failed to load alerts</div>`;
        }
    },

    async loadTrust() {
        const container = document.getElementById('trust-table');
        if (!container) return;
        try {
            const data = await this.api('/trust/overview');
            container.innerHTML = `
                <table>
                    <tr><th>Scope</th><th>ID</th><th>Trust Score</th><th>Status</th></tr>
                    ${data.map(t => `
                        <tr>
                            <td>${t.scope_type}</td>
                            <td>${t.scope_id}</td>
                            <td>
                                <div class="flex items-center gap-2">
                                    <div style="width: 50px; height: 6px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden;">
                                        <div style="width: ${Math.min(t.score, 100)}%; height: 100%; background: ${t.score > 80 ? 'var(--safe)' : t.score > 50 ? 'var(--warn)' : 'var(--danger)'}"></div>
                                    </div>
                                    ${t.score.toFixed(1)}
                                </div>
                            </td>
                            <td>${t.score > 80 ? 'High' : t.score > 50 ? 'Medium' : 'Low'}</td>
                        </tr>
                    `).join('')}
                </table>
            `;
        } catch (e) { }
    },

    async loadStats() {
        try {
            // Mock stats if endpoints don't exist for summary, or derive from alerts
            // We already update alert count in loadAlerts
        } catch (e) { }
    },

    async ackAlert(id) {
        try {
            await this.api(`/alerts/${id}/ack`, 'POST');
            this.loadAlerts();
            this.toast('Alert acknowledged', 'success');
        } catch (e) { }
    },

    // --- Unified Video Scan ---
    _initScanFileInput() {
        const fileInput = document.getElementById('scan-file');
        if (!fileInput) return;
        fileInput.addEventListener('change', () => {
            const nameEl = document.getElementById('scan-filename');
            if (fileInput.files.length) {
                nameEl.textContent = `📎 ${fileInput.files[0].name}`;
                nameEl.style.display = 'block';
            } else {
                nameEl.style.display = 'none';
            }
        });
    },

    async scanVideo() {
        const fileInput = document.getElementById('scan-file');
        const btn = document.getElementById('scan-btn');
        const resultsDiv = document.getElementById('scan-results');

        if (!fileInput || !fileInput.files.length) {
            return this.toast('Select a video file first', 'error');
        }

        btn.disabled = true;
        btn.textContent = '⏳ Running all models...';
        resultsDiv.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            this.toast('Scanning video with all models...', 'safe');
            const res = await fetch('/scan/video', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${this.state.token}` },
                body: formData,
            });
            const data = await res.json();
            console.log('[Scan] result:', data);
            if (!res.ok) throw new Error(data.detail || 'Scan failed');
            if (data.error) throw new Error(data.error);

            this._renderScanResults(data);
            this.toast('Scan complete!', 'success');
        } catch (e) {
            console.error('[Scan] error:', e);
            this.toast('Scan failed: ' + e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = '🔬 Run Full Scan';
        }
    },

    _renderScanResults(data) {
        const resultsDiv = document.getElementById('scan-results');
        resultsDiv.classList.remove('hidden');

        // Severity banner
        const banner = document.getElementById('scan-severity-banner');
        const sevColors = {
            critical: { bg: 'rgba(239,68,68,0.2)', border: '#ef4444', text: '🚨 CRITICAL — Immediate action required' },
            high: { bg: 'rgba(245,158,11,0.2)', border: '#f59e0b', text: '⚠️ HIGH — Violations detected' },
            medium: { bg: 'rgba(234,179,8,0.15)', border: '#eab308', text: '⚡ MEDIUM — Seat violation found' },
            low: { bg: 'rgba(34,197,94,0.15)', border: '#22c55e', text: '✅ LOW — No violations detected' },
        };
        const sev = sevColors[data.severity] || sevColors.low;
        banner.style.background = sev.bg;
        banner.style.border = `2px solid ${sev.border}`;
        banner.style.color = sev.border;
        banner.textContent = sev.text;

        // Violence card
        const vScore = (data.violence_score * 100).toFixed(1);
        const vColor = data.violence ? '#ef4444' : '#22c55e';
        document.getElementById('scan-violence-score').textContent = `${vScore}%`;
        document.getElementById('scan-violence-score').style.color = vColor;
        document.getElementById('scan-violence-bar').style.width = `${vScore}%`;
        document.getElementById('scan-violence-bar').style.background = vColor;
        document.getElementById('scan-violence-icon').textContent = data.violence ? '🔴' : '🟢';
        document.getElementById('scan-violence-label').textContent = data.violence ? 'Violence Detected' : 'No Violence';

        // Seat violations
        const seatCount = data.seat_violations || 0;
        document.getElementById('scan-seat-count').textContent = seatCount;
        document.getElementById('scan-seat-count').style.color = seatCount > 0 ? '#ef4444' : '#22c55e';

        // People count
        document.getElementById('scan-people-max').textContent = data.people_count_max || 0;
        document.getElementById('scan-people-avg').textContent = data.people_count_avg || 0;

        // Gender breakdown
        const genderDiv = document.getElementById('scan-gender');
        const g = data.gender_breakdown || {};
        genderDiv.innerHTML = [
            g.Male ? `<span style="color:#60a5fa;">♂ ${g.Male} Male</span>` : '',
            g.Female ? `<span style="color:#f472b6;">♀ ${g.Female} Female</span>` : '',
            g.Unknown ? `<span style="color:#9ca3af;">? ${g.Unknown} Unknown</span>` : '',
        ].filter(Boolean).join('') || '<span style="color:var(--text-muted);">No people detected</span>';

        // Summary
        document.getElementById('scan-summary').textContent = data.summary;
        document.getElementById('scan-meta').textContent =
            `Duration: ${data.video_duration_sec}s • Frames sampled: ${data.frames_sampled} • Frames analyzed: ${data.frames_analyzed}`;
    },

    // --- Model Selection ---
    async loadModelStatus() {
        try {
            const data = await this.api('/model/status');
            const statusEl = document.getElementById('model-status-text');
            const selectEl = document.getElementById('model-select');
            if (statusEl) {
                statusEl.textContent = data.yolo_available
                    ? `Active: ${data.current_model} ✅`
                    : `${data.current_model} (not loaded) ⚠️`;
            }
            if (selectEl) selectEl.value = data.current_model;
        } catch (e) { }
        this.loadROI();
    },

    // --- ROI Management ---
    setupROI() {
        this._initROICanvas();
        this.loadROI();
    },

    captureROIFrame() {
        // Loads live feed image for ROI calibration
        const img = document.getElementById('preview-img');
        if (!img) return;
        // Point to live stream MJPEG (browser handles stream)
        // Add random query param to bust cache if re-clicked
        img.src = `/stream/live?t=${Date.now()}`;
        this.toast('Live feed active on calibration area', 'success');
    },

    // --- ROI Canvas Drag & Drop ---
    _roiRect: null,   // {x1, y1, x2, y2} in fractions 0-1
    _roiDragging: false,
    _roiStart: null,

    _initROICanvas() {
        const canvas = document.getElementById('roi-canvas');
        const img = document.getElementById('preview-img');
        if (!canvas || !img) return;

        // Match canvas internal resolution to the displayed image size
        const syncSize = () => {
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            this._drawROIRect();
        };
        img.onload = syncSize;
        new ResizeObserver(syncSize).observe(img);
        if (img.complete) syncSize();

        canvas.onmousedown = (e) => {
            const rect = canvas.getBoundingClientRect();
            this._roiDragging = true;
            this._roiStart = {
                x: (e.clientX - rect.left) / rect.width,
                y: (e.clientY - rect.top) / rect.height
            };
        };
        canvas.onmousemove = (e) => {
            if (!this._roiDragging) return;
            const rect = canvas.getBoundingClientRect();
            const curX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            const curY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
            this._roiRect = {
                x1: Math.min(this._roiStart.x, curX),
                y1: Math.min(this._roiStart.y, curY),
                x2: Math.max(this._roiStart.x, curX),
                y2: Math.max(this._roiStart.y, curY),
            };
            this._drawROIRect();
        };
        canvas.onmouseup = () => { this._roiDragging = false; };
        canvas.onmouseleave = () => { this._roiDragging = false; };
    },

    _drawROIRect() {
        const canvas = document.getElementById('roi-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!this._roiRect) return;

        const { x1, y1, x2, y2 } = this._roiRect;
        const px1 = x1 * canvas.width, py1 = y1 * canvas.height;
        const pw = (x2 - x1) * canvas.width, ph = (y2 - y1) * canvas.height;

        // Semi-transparent fill
        ctx.fillStyle = 'rgba(0, 255, 255, 0.15)';
        ctx.fillRect(px1, py1, pw, ph);

        // Border
        ctx.strokeStyle = '#00ffff';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(px1, py1, pw, ph);
        ctx.setLineDash([]);

        // Label
        ctx.font = '13px sans-serif';
        ctx.fillStyle = '#00ffff';
        ctx.fillText('RESERVED SEAT (Female)', px1 + 6, py1 + 16);
    },

    async loadROI() {
        console.log('[ROI] loadROI called');
        try {
            const data = await this.api('/roi/config');
            console.log('[ROI] loaded from server:', data);
            this._roiRect = { x1: data.roi_x1, y1: data.roi_y1, x2: data.roi_x2, y2: data.roi_y2 };
            this._drawROIRect();
        } catch (e) { console.error('[ROI] loadROI error:', e); }
    },

    async saveROI() {
        console.log('[ROI] saveROI called, _roiRect =', this._roiRect);
        if (!this._roiRect) {
            this.toast('Draw a rectangle on the preview first', 'error');
            return;
        }
        const { x1, y1, x2, y2 } = this._roiRect;
        const payload = { roi_x1: x1, roi_y1: y1, roi_x2: x2, roi_y2: y2 };
        console.log('[ROI] saving payload:', payload);
        try {
            const result = await this.api('/roi/config', 'POST', payload);
            console.log('[ROI] save response:', result);
            this.toast('ROI saved! Preview again to see updated detection.', 'success');
            const status = document.getElementById('roi-status');
            if (status) status.textContent = `Active ROI: (${(x1 * 100).toFixed(0)}%, ${(y1 * 100).toFixed(0)}%) → (${(x2 * 100).toFixed(0)}%, ${(y2 * 100).toFixed(0)}%)`;
        } catch (e) {
            console.error('[ROI] save FAILED:', e);
            this.toast('Failed to save ROI: ' + (e.message || e), 'error');
        }
    },

    clearROI() {
        console.log('[ROI] clearROI');
        this._roiRect = null;
        this._drawROIRect();
        this.toast('ROI cleared — draw a new one or save to reset', 'success');
    },

    async switchModel() {
        const model = document.getElementById('model-select').value;
        const statusEl = document.getElementById('model-status-text');
        if (statusEl) statusEl.textContent = 'Switching model (downloading if needed)...';
        try {
            const data = await this.api('/model/switch', 'POST', { model });
            this.toast(`Switched to ${data.current_model}`, 'success');
            this.loadModelStatus();
        } catch (e) {
            this.loadModelStatus();
        }
    },

    // --- Live Stream ---
    startStream() {
        const custom = document.getElementById('stream-custom').value.trim();
        const source = custom || document.getElementById('stream-source').value;
        const feedImg = document.getElementById('live-feed');
        const placeholder = document.getElementById('feed-placeholder');
        const status = document.getElementById('feed-status');

        feedImg.src = `/stream/live?source=${encodeURIComponent(source)}`;
        feedImg.style.display = 'block';
        placeholder.style.display = 'none';
        status.textContent = 'Streaming';
        status.className = 'badge badge-danger';
        this.toast(`Stream started: ${source}`, 'success');
    },

    stopStream() {
        const feedImg = document.getElementById('live-feed');
        const placeholder = document.getElementById('feed-placeholder');
        const status = document.getElementById('feed-status');

        if (feedImg) {
            feedImg.src = '';
            feedImg.style.display = 'none';
        }
        if (placeholder) placeholder.style.display = '';
        if (status) {
            status.textContent = 'Stopped';
            status.className = 'badge badge-safe';
        }
    },

    // --- Utilities ---
    getSeverityClass(sev) {
        if (!sev) return 'safe';
        sev = sev.toLowerCase();
        if (sev.includes('critic')) return 'critical';
        if (sev.includes('high')) return 'danger';
        if (sev.includes('med')) return 'warn';
        return 'safe';
    },

    toast(msg, type = 'safe') {
        const t = document.createElement('div');
        t.className = `btn btn-primary badge-${type}`;
        t.style.position = 'fixed';
        t.style.bottom = '20px';
        t.style.right = '20px';
        t.style.zIndex = 1000;
        t.innerText = msg;
        document.body.appendChild(t);
        setTimeout(() => t.remove(), 3000);
    },



    startPolling() {
        if (this.state.refreshInterval) clearInterval(this.state.refreshInterval);
        this.state.refreshInterval = setInterval(() => {
            if (this.state.view === 'dashboard') this.loadAlerts();
            if (this.state.view === 'trust') this.loadTrust();
        }, 5000);
    }
};

window.App = App;
window.onload = () => App.init();
