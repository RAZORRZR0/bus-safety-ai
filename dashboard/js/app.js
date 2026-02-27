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
                if (res.status === 401 && token) {
                    this.toast('Session expired — please log in again.', 'error');
                    this.logout();
                    return;
                }
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
            this.initMdSelects();
            this.loadCameras();
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
        if (hash === 'dashboard') { this.loadAlerts(); }
        if (hash === 'trust') this.loadTrust();
        if (hash === 'ingest') this.loadModelStatus();
        if (hash === 'fleet') this.loadFleet();
        if (hash !== 'live') this.stopStream();
    },

    // --- Data Loading ---
    async loadAlerts() {
        const container = document.getElementById('alerts-feed');
        if (!container) return;
        try {
            const alerts = await this.api('/alerts');
            container.innerHTML = alerts.map(a => `
                <div class="card card-outlined flex justify-between items-center" style="margin-bottom: 0.5rem; padding: 0.875rem 1rem;">
                    <div>
                        <div class="flex items-center gap-2">
                            <span class="badge badge-${this.getSeverityClass(a.severity)}">${a.severity}</span>
                            <span style="font-weight: 500; font-size:0.875rem;">Bus ${a.bus_id}</span>
                            <small>${new Date(a.created_at).toLocaleTimeString()}</small>
                        </div>
                        <div style="margin-top:0.25rem; font-size: 0.8125rem; color: var(--md-on-surface-variant)">Route: ${a.route_id} | Status: ${a.status}</div>
                    </div>
                    <div class="flex gap-2">
                        <button class="btn btn-sm btn-outline" style="padding: 0 0.5rem;" title="Mark True Positive" onclick="App.submitFeedback('${a.violation_event_id}', true, 'alert_violence_true')">
                            <span class="material-symbols-outlined" style="font-size:16px; color:#22c55e">thumb_up</span>
                        </button>
                        <button class="btn btn-sm btn-outline" style="padding: 0 0.5rem;" title="Mark False Positive" onclick="App.submitFeedback('${a.violation_event_id}', false, 'alert_violence_false')">
                            <span class="material-symbols-outlined" style="font-size:16px; color:#ef4444">thumb_down</span>
                        </button>
                        ${a.status !== 'acknowledged' ? `<button onclick="App.ackAlert(${a.id})" class="btn btn-sm btn-outline">Ack</button>` : '<span class="material-symbols-outlined" style="color:var(--md-safe); font-size:20px">check_circle</span>'}
                    </div>
                </div>
            `).join('');

            // Update Stats
            document.getElementById('stat-alerts').innerText = alerts.filter(a => a.status === 'open').length;
        } catch (e) {
            container.innerHTML = `<div class="text-center text-muted"><span class="material-symbols-outlined" style="font-size:32px; opacity:0.3; display:block; margin-bottom:0.5rem">error_outline</span>Failed to load alerts</div>`;
        }
    },

    async loadTrust() {
        const container = document.getElementById('trust-table');
        if (!container) return;
        try {
            const data = await this.api('/trust/overview');

            if (!data.length) {
                container.innerHTML = `<div class="text-center" style="padding: 3rem;">
                    <span class="material-symbols-outlined" style="font-size:48px; opacity:0.25; display:block; margin-bottom:0.75rem">shield</span>
                    <p style="margin:0">No trust scores yet. Scan a video tagged to a bus to generate scores.</p>
                </div>`;
                return;
            }

            // Group by scope_type
            const groups = { bus: [], route: [], timeslot: [] };
            data.forEach(t => { (groups[t.scope_type] || (groups[t.scope_type] = [])).push(t); });

            const scoreColor = s => s > 80 ? '#22c55e' : s > 50 ? '#f59e0b' : '#ef4444';
            const scoreLabel = s => s > 80 ? 'Trusted' : s > 50 ? 'At Risk' : 'Flagged';
            const scoreIcon = s => s > 80 ? 'verified_user' : s > 50 ? 'warning' : 'dangerous';

            const renderCard = t => {
                const color = scoreColor(t.score);
                const pct = Math.min(t.score, 100);
                const circumference = 2 * Math.PI * 22; // r=22
                const dash = (pct / 100) * circumference;
                return `
                <div style="display:flex; flex-direction:column; align-items:center; gap:0.4rem; padding:0.75rem 1rem;
                            background:var(--md-surface-container-low); border-radius:var(--md-shape-sm);
                            border:1px solid var(--md-outline-variant); min-width:100px;">
                    <!-- Score Ring -->
                    <svg width="56" height="56" viewBox="0 0 56 56">
                        <circle cx="28" cy="28" r="22" fill="none" stroke="var(--md-outline-variant)" stroke-width="5"/>
                        <circle cx="28" cy="28" r="22" fill="none" stroke="${color}" stroke-width="5"
                            stroke-dasharray="${dash.toFixed(1)} ${circumference.toFixed(1)}"
                            stroke-linecap="round" transform="rotate(-90 28 28)"/>
                        <text x="28" y="33" text-anchor="middle" font-size="12" font-weight="600" fill="${color}">${t.score.toFixed(0)}</text>
                    </svg>
                    <div style="font-weight:500; font-size:0.8125rem; text-align:center; word-break:break-all;">${t.scope_id}</div>
                    <div style="display:flex; align-items:center; gap:0.25rem; font-size:0.7rem; color:${color};">
                        <span class="material-symbols-outlined" style="font-size:14px">${scoreIcon(t.score)}</span>
                        ${scoreLabel(t.score)}
                    </div>
                </div>`;
            };

            const sections = [
                { key: 'bus', label: 'Buses', icon: 'directions_bus' },
                { key: 'route', label: 'Routes', icon: 'route' },
                { key: 'timeslot', label: 'Timeslots', icon: 'schedule' },
            ];

            container.innerHTML = sections
                .filter(s => groups[s.key]?.length)
                .map(s => `
                    <div style="margin-bottom:1.25rem;">
                        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.75rem;">
                            <span class="material-symbols-outlined" style="font-size:18px; color:var(--md-on-surface-variant)">${s.icon}</span>
                            <span style="font-size:0.75rem; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; color:var(--md-on-surface-variant);">${s.label}</span>
                        </div>
                        <div style="display:flex; flex-wrap:wrap; gap:0.75rem;">
                            ${groups[s.key].sort((a, b) => a.score - b.score).map(renderCard).join('')}
                        </div>
                    </div>
                `).join('');

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

        // Fleet context (optional)
        const busId = (document.getElementById('scan-bus-select')?.value || '').trim();
        const routeId = (document.getElementById('scan-route-display')?.value || '').trim();
        const timeslot = (document.getElementById('scan-timeslot-input')?.value || '').trim();

        btn.disabled = true;
        btn.textContent = '⏳ Running all models...';
        resultsDiv.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        if (busId) formData.append('bus_id', busId);
        if (routeId) formData.append('route_id', routeId);
        if (timeslot) formData.append('timeslot', timeslot);

        try {
            const fleetMsg = busId ? `Scanning for bus ${busId}...` : 'Scanning video with all models...';
            this.toast(fleetMsg, 'safe');
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
            if (data.event_id) {
                this.toast(`Scan complete! Event #${data.event_id} logged — trust updated.`, 'success');
            } else {
                this.toast('Scan complete!', 'success');
            }
        } catch (e) {
            console.error('[Scan] error:', e);
            this.toast('Scan failed: ' + e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = '🔬 Run Full Scan';
        }
    },

    currentScanRefId: null,

    _renderScanResults(data) {
        const resultsDiv = document.getElementById('scan-results');
        resultsDiv.classList.remove('hidden');
        this.currentScanRefId = data.reference_id;

        // Fleet event badge (shown if scan was tagged to a bus)
        const existingBadge = document.getElementById('scan-event-badge');
        if (existingBadge) existingBadge.remove();
        if (data.event_id) {
            const badge = document.createElement('div');
            badge.id = 'scan-event-badge';
            badge.style.cssText = 'margin-bottom:0.75rem; padding:0.5rem 1rem; border-radius:8px; background:rgba(26,115,232,0.1); border:1px solid var(--md-primary); font-size:0.8125rem; display:flex; align-items:center; gap:0.5rem;';
            badge.innerHTML = `<span class="material-symbols-outlined" style="font-size:18px;color:var(--md-primary)">link</span>
                <span>Linked to <strong>${data.bus_id}</strong> / Route <strong>${data.route_id}</strong> — Event <strong>#${data.event_id}</strong> logged${data.alert_id ? ` · Alert <strong>#${data.alert_id}</strong> raised` : ''}</span>`;
            resultsDiv.prepend(badge);
        }

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
        const iconEl = document.getElementById('scan-violence-icon');
        iconEl.textContent = data.violence ? 'dangerous' : 'verified_user';
        iconEl.style.color = vColor;
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
            if (selectEl) {
                selectEl.value = data.current_model;
                this.syncMdSelect(selectEl);
            }

            // Violence Model
            const vStatusEl = document.getElementById('violence-model-status-text');
            const vSelectEl = document.getElementById('violence-model-select');
            const vMenuEl = document.getElementById('violence-md-select-menu');
            const vValueEl = document.getElementById('violence-md-select-value');

            if (vStatusEl) {
                vStatusEl.textContent = data.current_violence_model
                    ? `Active: ${data.current_violence_model} ✅`
                    : `No violence model active ⚠️`;
            }

            if (vSelectEl && vMenuEl && vValueEl && data.available_violence_models) {
                const current = data.current_violence_model;

                vSelectEl.innerHTML = data.available_violence_models.map(m =>
                    `<option value="${m}" ${m === current ? 'selected' : ''}>${m}</option>`
                ).join('');

                vMenuEl.innerHTML = data.available_violence_models.map(m =>
                    `<li class="md-select-option ${m === current ? 'active' : ''}" data-value="${m}" role="option">${m}</li>`
                ).join('');

                vValueEl.textContent = current || 'Select Model';

                // Re-initialize select event listeners for the newly injected menu items
                this.syncMdSelect(vSelectEl);
            }
        } catch (e) { }
        this.loadROI();
        this._loadScanFleetDropdown();
    },

    // Populate the bus dropdown in the Scanner's Fleet Context card
    async _loadScanFleetDropdown() {
        const sel = document.getElementById('scan-bus-select');
        if (!sel) return;
        try {
            const buses = await this.api('/fleet/buses');
            // Store route_id lookup by bus_id for auto-fill
            this._scanBusRouteMap = {};
            buses.forEach(b => { this._scanBusRouteMap[b.bus_id] = b.route_id; });
            // Re-build options
            const current = sel.value;
            sel.innerHTML = '<option value="">— No bus selected —</option>' +
                buses.map(b => `<option value="${b.bus_id}">${b.bus_id}</option>`).join('');
            if (current) sel.value = current;
            this.onScanBusChange(sel);
        } catch (e) { }
    },

    _scanBusRouteMap: {},

    onScanBusChange(sel) {
        const routeInput = document.getElementById('scan-route-display');
        if (!routeInput) return;
        const busId = sel.value;
        routeInput.value = busId ? (this._scanBusRouteMap[busId] || '') : '';
    },

    // --- Camera Enumeration (uses browser MediaDevices API — same as Discord/Zoom) ---
    async loadCameras() {
        try {
            // Step 1: enumerate devices (labels may be empty without permission)
            let devices = await navigator.mediaDevices.enumerateDevices();
            let videoDevices = devices.filter(d => d.kind === 'videoinput');

            // Step 2: if labels are empty, briefly request camera to unlock labels
            if (videoDevices.length > 0 && !videoDevices[0].label) {
                try {
                    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    tempStream.getTracks().forEach(t => t.stop()); // release immediately
                    devices = await navigator.mediaDevices.enumerateDevices();
                    videoDevices = devices.filter(d => d.kind === 'videoinput');
                } catch (permErr) {
                    console.warn('[Cameras] Permission denied, using generic names');
                }
            }

            if (!videoDevices.length) return;

            // Build camera list: index = OpenCV camera index
            const cameras = videoDevices.map((dev, i) => ({
                id: String(i),
                name: dev.label || `Camera (${i})`
            }));

            const sourceSelect = document.getElementById('stream-source');
            const wrapper = sourceSelect?.closest('.md-select');
            if (!sourceSelect || !wrapper) return;

            const menu = wrapper.querySelector('.md-select-menu');
            const valueSpan = wrapper.querySelector('.md-select-value');

            sourceSelect.innerHTML = cameras.map((cam, i) =>
                `<option value="${cam.id}" ${i === 0 ? 'selected' : ''}>${cam.name}</option>`
            ).join('');

            menu.innerHTML = cameras.map((cam, i) =>
                `<li class="md-select-option ${i === 0 ? 'active' : ''}" data-value="${cam.id}" role="option">${cam.name}</li>`
            ).join('');

            if (valueSpan && cameras.length > 0) {
                valueSpan.textContent = cameras[0].name;
            }

            this.syncMdSelect(sourceSelect);
        } catch (e) {
            console.warn('[Cameras] Failed to load cameras:', e);
        }
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

    async switchViolenceModel() {
        const model = document.getElementById('violence-model-select').value;
        const statusEl = document.getElementById('violence-model-status-text');
        if (statusEl) statusEl.textContent = 'Switching violence model...';
        try {
            const data = await this.api('/model/switch', 'POST', { violence_model: model });
            this.toast(`Switched violence model to ${data.current_violence_model}`, 'success');
            this.loadModelStatus();
        } catch (e) {
            this.loadModelStatus();
        }
    },

    // --- Live Stream ---
    async uploadStreamFile() {
        const fileInput = document.getElementById('stream-upload-input');
        const customInput = document.getElementById('stream-custom');
        if (!fileInput || !fileInput.files.length) return;

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            this.toast('Uploading file for stream...', 'safe');
            const res = await fetch('/stream/upload', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${this.state.token}` },
                body: formData,
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Upload failed');

            customInput.value = data.path;
            this.toast('File uploaded! Press Start to stream.', 'success');
        } catch (e) {
            console.error(e);
            this.toast('Upload failed: ' + e.message, 'error');
        }
    },

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

    async stopStream() {
        const feedImg = document.getElementById('live-feed');
        const placeholder = document.getElementById('feed-placeholder');
        const status = document.getElementById('feed-status');

        // 1. Disconnect browser from MJPEG stream immediately
        if (feedImg) {
            feedImg.src = '';
            feedImg.style.display = 'none';
        }
        if (placeholder) placeholder.style.display = '';
        if (status) {
            status.textContent = 'Stopping...';
            status.className = 'badge badge-warn';
        }

        // 2. Tell backend to stop generator and release camera
        try {
            const headers = {};
            if (this.state.token) headers['Authorization'] = `Bearer ${this.state.token}`;
            const res = await fetch('/stream/stop', { method: 'POST', headers });
            console.log('[Stream] stop response:', res.status);
        } catch (e) {
            console.warn('[Stream] stop request failed:', e);
        }

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
        // Remove any existing snackbar
        const existing = document.querySelector('.snackbar');
        if (existing) existing.remove();

        const snackbar = document.createElement('div');
        let typeClass = '';
        if (type === 'error' || type === 'danger') typeClass = 'snackbar-error';
        else if (type === 'success') typeClass = 'snackbar-success';
        snackbar.className = `snackbar ${typeClass}`;

        // Icon
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined';
        icon.style.fontSize = '20px';
        if (type === 'error' || type === 'danger') icon.textContent = 'error';
        else if (type === 'success') icon.textContent = 'check_circle';
        else icon.textContent = 'info';
        snackbar.appendChild(icon);

        const text = document.createElement('span');
        text.textContent = msg;
        snackbar.appendChild(text);

        document.body.appendChild(snackbar);
        // Trigger reflow then animate in
        requestAnimationFrame(() => {
            requestAnimationFrame(() => snackbar.classList.add('visible'));
        });
        setTimeout(() => {
            snackbar.classList.remove('visible');
            setTimeout(() => snackbar.remove(), 300);
        }, 3000);
    },



    async submitFeedback(refIdOrAction, isCorrect = null, type = null) {
        try {
            // Support both manual scan feedback and alert feedback
            let refId = refIdOrAction;
            let finalType = type;
            let finalIsCorrect = isCorrect;

            if (isCorrect === null) {
                if (!this.currentScanRefId) {
                    return this.toast('No active scan to provide feedback for.', 'error');
                }
                refId = this.currentScanRefId;
                finalIsCorrect = refIdOrAction === 'correct_violence';
                finalType = refIdOrAction;
            }

            const payload = {
                event_reference_id: String(refId),
                feedback_type: finalType,
                is_correct: finalIsCorrect
            };

            await this.api('/feedback', 'POST', payload);
            this.toast('Feedback submitted. Thanks for improving the model!', 'success');
        } catch (e) {
            console.error(e);
        }
    },

    startPolling() {
        if (this.state.refreshInterval) clearInterval(this.state.refreshInterval);
        this.state.refreshInterval = setInterval(() => {
            if (this.state.view === 'dashboard') this.loadAlerts();
            if (this.state.view === 'trust') this.loadTrust();
        }, 5000);
    }
};

// ── Fleet Management ──────────────────────────────────────────────
App.loadFleet = async function () {
    await Promise.all([App._loadBuses(), App._loadRoutes()]);
};

App._loadBuses = async function () {
    const wrap = document.getElementById('bus-table-wrap');
    if (!wrap) return;
    try {
        const buses = await App.api('/fleet/buses');
        if (!buses.length) {
            wrap.innerHTML = `<div class="text-center text-muted" style="padding:2rem">
                <span class="material-symbols-outlined" style="font-size:32px;opacity:0.3;display:block;margin-bottom:0.5rem">directions_bus</span>
                No buses registered yet</div>`;
            return;
        }
        wrap.innerHTML = `<table>
            <tr><th>Bus ID</th><th>Route</th><th>Model</th><th>Year</th><th>Capacity</th><th></th></tr>
            ${buses.map(b => `<tr>
                <td><strong>${b.bus_id}</strong></td>
                <td>${b.route_id}</td>
                <td>${b.model}</td>
                <td>${b.year}</td>
                <td>${b.capacity}</td>
                <td><button class="btn btn-sm btn-outline" style="padding:0 0.4rem" title="Delete" onclick="App.deleteBus('${b.bus_id}')">
                    <span class="material-symbols-outlined" style="font-size:16px;color:#ef4444">delete</span>
                </button></td>
            </tr>`).join('')}
        </table>`;
    } catch (e) { }
};

App._loadRoutes = async function () {
    const wrap = document.getElementById('route-table-wrap');
    if (!wrap) return;
    try {
        const routes = await App.api('/fleet/routes');
        if (!routes.length) {
            wrap.innerHTML = `<div class="text-center text-muted" style="padding:2rem">
                <span class="material-symbols-outlined" style="font-size:32px;opacity:0.3;display:block;margin-bottom:0.5rem">route</span>
                No routes registered yet</div>`;
            return;
        }
        wrap.innerHTML = `<table>
            <tr><th>Route ID</th><th>Name</th><th>From</th><th>To</th><th>Dist (km)</th><th></th></tr>
            ${routes.map(r => `<tr>
                <td><strong>${r.route_id}</strong></td>
                <td>${r.name}</td>
                <td>${r.start_location}</td>
                <td>${r.end_location}</td>
                <td>${r.distance_km}</td>
                <td><button class="btn btn-sm btn-outline" style="padding:0 0.4rem" title="Delete" onclick="App.deleteRoute('${r.route_id}')">
                    <span class="material-symbols-outlined" style="font-size:16px;color:#ef4444">delete</span>
                </button></td>
            </tr>`).join('')}
        </table>`;
    } catch (e) { }
};

App.addBus = async function () {
    const busId = document.getElementById('bus-id-input').value.trim();
    const routeId = document.getElementById('bus-route-input').value.trim();
    const model = document.getElementById('bus-model-input').value.trim();
    const year = parseInt(document.getElementById('bus-year-input').value);
    const capacity = parseInt(document.getElementById('bus-capacity-input').value);
    if (!busId || !routeId || !model || !year || !capacity)
        return App.toast('Fill in all bus fields', 'error');
    try {
        await App.api('/fleet/buses', 'POST', { bus_id: busId, route_id: routeId, model, year, capacity });
        App.toast(`Bus ${busId} registered!`, 'success');
        ['bus-id-input', 'bus-route-input', 'bus-model-input', 'bus-year-input', 'bus-capacity-input']
            .forEach(id => document.getElementById(id).value = '');
        App._loadBuses();
    } catch (e) { }
};

App.deleteBus = async function (busId) {
    if (!confirm(`Delete bus ${busId}?`)) return;
    try {
        await App.api(`/fleet/buses/${encodeURIComponent(busId)}`, 'DELETE');
        App.toast(`Bus ${busId} removed`, 'success');
        App._loadBuses();
    } catch (e) { }
};

App.addRoute = async function () {
    const routeId = document.getElementById('route-id-input').value.trim();
    const name = document.getElementById('route-name-input').value.trim();
    const start = document.getElementById('route-start-input').value.trim();
    const end = document.getElementById('route-end-input').value.trim();
    const dist = parseFloat(document.getElementById('route-dist-input').value);
    if (!routeId || !name || !start || !end || isNaN(dist))
        return App.toast('Fill in all route fields', 'error');
    try {
        await App.api('/fleet/routes', 'POST', { route_id: routeId, name, start_location: start, end_location: end, distance_km: dist });
        App.toast(`Route ${routeId} registered!`, 'success');
        ['route-id-input', 'route-name-input', 'route-start-input', 'route-end-input', 'route-dist-input']
            .forEach(id => document.getElementById(id).value = '');
        App._loadRoutes();
    } catch (e) { }
};

App.deleteRoute = async function (routeId) {
    if (!confirm(`Delete route ${routeId}?`)) return;
    try {
        await App.api(`/fleet/routes/${encodeURIComponent(routeId)}`, 'DELETE');
        App.toast(`Route ${routeId} removed`, 'success');
        App._loadRoutes();
    } catch (e) { }
};

window.App = App;
window.onload = () => {
    App.initMdSelects(); // Init custom dropdowns for auth view too
    App.init();
};

/* ── Custom Material Select Dropdown Logic ──────────────── */
App.initMdSelects = function () {
    document.querySelectorAll('.md-select:not(.md-initialized)').forEach(wrapper => {
        wrapper.classList.add('md-initialized');
        const trigger = wrapper.querySelector('.md-select-trigger');
        const menu = wrapper.querySelector('.md-select-menu');
        const valueSpan = wrapper.querySelector('.md-select-value');
        const hiddenSelect = wrapper.querySelector('select');
        if (!trigger || !menu || !hiddenSelect) return;

        // Toggle menu on trigger click
        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            // Close all other open dropdowns first
            document.querySelectorAll('.md-select.open').forEach(other => {
                if (other !== wrapper) other.classList.remove('open');
            });
            wrapper.classList.toggle('open');
        });

        // Option click
        menu.querySelectorAll('.md-select-option').forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                const val = opt.dataset.value;
                // Update hidden select
                hiddenSelect.value = val;
                // Update display
                valueSpan.textContent = opt.textContent;
                // Update active state
                menu.querySelectorAll('.md-select-option').forEach(o => o.classList.remove('active'));
                opt.classList.add('active');
                // Close
                wrapper.classList.remove('open');
            });
        });
    });

    // Close dropdowns when clicking outside
    document.addEventListener('click', () => {
        document.querySelectorAll('.md-select.open').forEach(w => w.classList.remove('open'));
    });
};

// Sync a custom dropdown UI when native select value changes programmatically
// Also re-binds click handlers for dynamically injected menu items
App.syncMdSelect = function (selectEl) {
    const wrapper = selectEl.closest('.md-select');
    if (!wrapper) return;
    const val = selectEl.value;
    const valueSpan = wrapper.querySelector('.md-select-value');
    const menu = wrapper.querySelector('.md-select-menu');
    if (!menu) return;
    menu.querySelectorAll('.md-select-option').forEach(opt => {
        const isActive = opt.dataset.value === val;
        opt.classList.toggle('active', isActive);
        if (isActive && valueSpan) valueSpan.textContent = opt.textContent;

        // Re-bind click handler (remove old by cloning)
        const newOpt = opt.cloneNode(true);
        newOpt.addEventListener('click', (e) => {
            e.stopPropagation();
            const v = newOpt.dataset.value;
            selectEl.value = v;
            if (valueSpan) valueSpan.textContent = newOpt.textContent;
            menu.querySelectorAll('.md-select-option').forEach(o => o.classList.remove('active'));
            newOpt.classList.add('active');
            wrapper.classList.remove('open');
        });
        opt.parentNode.replaceChild(newOpt, opt);
    });
};
