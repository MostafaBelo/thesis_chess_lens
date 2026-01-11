// Global State
let selectedPi = null;
let selectedTournament = null;
let piStates = {};
let tournamentData = {};
let currentFenCount = 0;

// Chess board state
let chessBoard = null;
let fenList = [];
let currentMoveIndex = -1;
let autoplayInterval = null;
let isAutoPlaying = false;

// WebSocket Setup
const ws = new WebSocket(`ws://${location.host}/ws`);

ws.onopen = () => {
    console.log("✓ WebSocket connected");
    initializeApp();
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log("WS Message:", data.type);
    
    if (data.type === "INITIAL_STATE") {
        handleInitialState(data.nodes);
    } else if (data.type === "STATUS_UPDATE") {
        handleStatusUpdate(data);
    } else if (data.type === "FEN_UPDATE") {
        handleFenUpdate(data);
    } else if (data.type === "TOURNAMENT_UPDATE") {
        handleTournamentUpdate();
    }
};

ws.onclose = () => console.log("✗ WebSocket disconnected");
ws.onerror = (error) => console.error("WebSocket error:", error);

// Initialize App
async function initializeApp() {
    await loadPiDevices();
    await loadTournaments();
    initChessBoard();
}

// Initialize Chess Board
function initChessBoard() {
    const config = {
        draggable: false,
        position: 'start',
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
    };
    chessBoard = Chessboard('chess-board', config);
}

// Handle Initial State
function handleInitialState(nodes) {
    for (const piId in nodes) {
        piStates[piId] = {
            state: nodes[piId].state || 'unknown',
            last_seen: nodes[piId].last_seen || 0
        };
    }
    updateDeviceList();
    updateHeaderStats();
}

// Handle Status Update
function handleStatusUpdate(data) {
    const { pi_id, state, last_seen } = data;
    
    // Check if this is a new Pi device
    const isNewDevice = !(pi_id in piStates);
    
    piStates[pi_id] = { state, last_seen };
    
    if (isNewDevice) {
        console.log(`🆕 New device connected: ${pi_id}`);
        loadPiDevices();
    } else {
        updateDeviceCard(pi_id);
    }
    
    updateHeaderStats();
    
    // Update control panel if this device is selected
    if (selectedPi === pi_id) {
        updateControlPanel(pi_id);
    }
}

// Handle Tournament Update (file changed)
async function handleTournamentUpdate() {
    console.log("📁 Tournament file changed - refreshing data");
    await refreshTournaments();
    
    // If currently viewing a game, refresh it
    if (selectedTournament && selectedPi) {
        await loadGameForViewing(selectedTournament, selectedPi);
    }
}

// Handle FEN Update
function handleFenUpdate(data) {
    console.log("📊 FEN update received:", data);
    
    // If viewing this device's game, refresh
    if (selectedPi === data.pi_id) {
        for (const tournament in tournamentData) {
            if (tournamentData[tournament][data.pi_id]) {
                loadGameForViewing(tournament, data.pi_id);
                break;
            }
        }
    }
}

// Load Pi Devices
async function loadPiDevices() {
    try {
        const piIds = new Set();
        
        // Get devices from tournaments
        try {
            const response = await fetch('/api/tournaments');
            const data = await response.json();
            
            for (const tournament in data.tournaments) {
                for (const piId in data.tournaments[tournament]) {
                    piIds.add(piId);
                }
            }
        } catch (e) {
            console.warn("Could not load tournament data:", e);
        }
        
        // Get ALL connected Pi devices from status endpoint
        try {
            const statusRes = await fetch('/api/pi-status');
            const statusData = await statusRes.json();
            for (const piId in statusData.nodes) {
                piIds.add(piId);
                piStates[piId] = {
                    state: statusData.nodes[piId].state || 'unknown',
                    last_seen: statusData.nodes[piId].last_seen || 0
                };
            }
        } catch (e) {
            console.warn("Could not load Pi status:", e);
        }
        
        renderDeviceList(Array.from(piIds).sort());
        updateHeaderStats();
    } catch (error) {
        console.error("Error loading Pi devices:", error);
    }
}

// Render Device List
function renderDeviceList(piIds) {
    const container = document.getElementById('device-list');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (piIds.length === 0) {
        container.innerHTML = `
            <div class="empty-state" style="padding: 2rem 1rem;">
                <i class="ri-device-line"></i>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">No Pi devices connected</p>
            </div>
        `;
        return;
    }
    
    piIds.forEach(piId => {
        const state = piStates[piId]?.state || 'unknown';
        const statusClass = getStatusClass(state);
        
        const card = document.createElement('div');
        card.className = 'device-card';
        card.id = `device-${piId}`;
        card.innerHTML = `
            <div class="device-header">
                <span class="device-name">${piId}</span>
                <span class="device-status ${statusClass}">
                    <span class="status-dot"></span>
                    ${state}
                </span>
            </div>
            <div class="device-actions-quick">
                <button class="quick-btn quick-btn-view" onclick="viewPiGame('${piId}'); event.stopPropagation();" title="View Game">
                    <i class="ri-eye-line"></i>
                </button>
                <button class="quick-btn quick-btn-control" onclick="selectDevice('${piId}'); event.stopPropagation();" title="Control">
                    <i class="ri-settings-3-line"></i>
                </button>
            </div>
        `;
        
        card.onclick = () => selectDevice(piId);
        container.appendChild(card);
    });
}

// Get status class based on state
function getStatusClass(state) {
    if (state === 'STREAMING') return 'streaming';
    if (state === 'IDLE') return 'idle';
    if (state === 'DISCONNECTED') return 'disconnected';
    if (state === 'OFFLINE') return 'offline';
    return 'offline';
}

// Update Device Card
function updateDeviceCard(piId) {
    const card = document.getElementById(`device-${piId}`);
    if (!card) {
        loadPiDevices();
        return;
    }
    
    const state = piStates[piId]?.state || 'unknown';
    const statusClass = getStatusClass(state);
    
    const statusElement = card.querySelector('.device-status');
    if (statusElement) {
        statusElement.className = `device-status ${statusClass}`;
        statusElement.innerHTML = `
            <span class="status-dot"></span>
            ${state}
        `;
    }
}

// Update Device List
function updateDeviceList() {
    Object.keys(piStates).forEach(piId => updateDeviceCard(piId));
}

// View Pi Game
async function viewPiGame(piId) {
    console.log(`🎮 Viewing game for ${piId}`);
    
    // Find which tournament this Pi belongs to
    let foundTournament = null;
    for (const tournament in tournamentData) {
        if (tournamentData[tournament][piId]) {
            foundTournament = tournament;
            break;
        }
    }
    
    if (!foundTournament) {
        showEmptyBoard(piId);
        return;
    }
    
    selectedPi = piId;
    selectedTournament = foundTournament;
    await loadGameForViewing(foundTournament, piId);
}

// Load Game for Viewing
async function loadGameForViewing(tournament, piId) {
    try {
        const response = await fetch('/api/tournaments');
        const data = await response.json();
        const fens = data.tournaments[tournament]?.[piId] || [];
        
        if (fens.length === 0) {
            showEmptyBoard(piId);
            return;
        }
        
        fenList = fens;
        currentMoveIndex = fens.length - 1; // Start at last move
        
        // Update viewer title
        document.getElementById('viewer-title').textContent = `${piId} • ${tournament}`;
        
        // Show board, hide info
        document.getElementById('chess-board').style.display = 'block';
        document.getElementById('board-info').classList.remove('show');
        
        // Display current position
        updateBoardPosition();
        
    } catch (error) {
        console.error("Error loading game:", error);
    }
}

// Show Empty Board
function showEmptyBoard(piId) {
    selectedPi = piId;
    fenList = [];
    currentMoveIndex = -1;
    
    document.getElementById('viewer-title').textContent = `${piId} • No Game Data`;
    document.getElementById('chess-board').style.display = 'none';
    const boardInfo = document.getElementById('board-info');
    boardInfo.classList.add('show');
    boardInfo.innerHTML = `
        <div class="empty-state">
            <i class="ri-chess-line"></i>
            <h3>No Game Data</h3>
            <p>${piId} hasn't published any FENs yet.<br>Start streaming to see the game.</p>
        </div>
    `;
    
    updateNavigationButtons();
}

// Update Board Position
function updateBoardPosition() {
    if (currentMoveIndex >= 0 && currentMoveIndex < fenList.length) {
        const fen = fenList[currentMoveIndex];
        chessBoard.position(fen);
        document.getElementById('move-indicator').textContent = `${currentMoveIndex + 1}/${fenList.length}`;
    }
    updateNavigationButtons();
}

// Navigation Functions
function goToFirstMove() {
    if (fenList.length > 0) {
        currentMoveIndex = 0;
        updateBoardPosition();
    }
}

function goToPrevMove() {
    if (currentMoveIndex > 0) {
        currentMoveIndex--;
        updateBoardPosition();
    }
}

function goToNextMove() {
    if (currentMoveIndex < fenList.length - 1) {
        currentMoveIndex++;
        updateBoardPosition();
    }
}

function goToLastMove() {
    if (fenList.length > 0) {
        currentMoveIndex = fenList.length - 1;
        updateBoardPosition();
    }
}

function toggleAutoplay() {
    if (isAutoPlaying) {
        stopAutoplay();
    } else {
        startAutoplay();
    }
}

function startAutoplay() {
    if (fenList.length === 0) return;
    
    isAutoPlaying = true;
    document.getElementById('btn-play').classList.add('playing');
    document.getElementById('btn-play').innerHTML = '<i class="ri-pause-fill"></i>';
    
    autoplayInterval = setInterval(() => {
        if (currentMoveIndex < fenList.length - 1) {
            goToNextMove();
        } else {
            stopAutoplay();
        }
    }, 1500); // 1.5 seconds per move
}

function stopAutoplay() {
    isAutoPlaying = false;
    if (autoplayInterval) {
        clearInterval(autoplayInterval);
        autoplayInterval = null;
    }
    document.getElementById('btn-play').classList.remove('playing');
    document.getElementById('btn-play').innerHTML = '<i class="ri-play-fill"></i>';
}

function updateNavigationButtons() {
    document.getElementById('btn-first').disabled = currentMoveIndex <= 0 || fenList.length === 0;
    document.getElementById('btn-prev').disabled = currentMoveIndex <= 0 || fenList.length === 0;
    document.getElementById('btn-next').disabled = currentMoveIndex >= fenList.length - 1 || fenList.length === 0;
    document.getElementById('btn-last').disabled = currentMoveIndex >= fenList.length - 1 || fenList.length === 0;
    document.getElementById('btn-play').disabled = fenList.length === 0;
}

// Select Device (for control panel)
function selectDevice(piId) {
    selectedPi = piId;
    
    // Update active state
    document.querySelectorAll('.device-card').forEach(card => {
        card.classList.remove('active');
    });
    document.getElementById(`device-${piId}`)?.classList.add('active');
    
    showControlPanel(piId);
}

// Show Control Panel
function showControlPanel(piId) {
    const section = document.getElementById('control-section-content');
    if (!section) return;
    
    const state = piStates[piId]?.state || 'unknown';
    const isStreaming = state === 'STREAMING';
    const statusClass = getStatusClass(state);
    
    section.innerHTML = `
        <div class="control-info">
            <div class="info-card">
                <div class="info-label">Device ID</div>
                <div class="info-value">${piId}</div>
            </div>
            <div class="info-card">
                <div class="info-label">Status</div>
                <div class="info-value">
                    <span class="device-status ${statusClass}">
                        <span class="status-dot"></span>
                        ${state}
                    </span>
                </div>
            </div>
            <div class="info-card">
                <div class="info-label">Last Seen</div>
                <div class="info-value">${formatLastSeen(piStates[piId]?.last_seen || 0)}</div>
            </div>
        </div>
        <div class="control-actions" style="margin-top: 1.5rem;">
            <button class="btn btn-start" onclick="startStreaming('${piId}')" ${isStreaming ? 'disabled' : ''} style="width: 100%; margin-bottom: 0.75rem;">
                <i class="ri-play-fill"></i> Start Streaming
            </button>
            <button class="btn btn-stop" onclick="stopStreaming('${piId}')" ${!isStreaming ? 'disabled' : ''} style="width: 100%;">
                <i class="ri-stop-fill"></i> Stop
            </button>
        </div>
    `;
}

// Update Control Panel
function updateControlPanel(piId) {
    if (selectedPi === piId) {
        showControlPanel(piId);
    }
}

// Start Streaming
async function startStreaming(piId) {
    try {
        const response = await fetch(`/api/command/${piId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                action: 'start_recording',
                game_id: `game_${piId}_${Date.now()}`
            })
        });
        const data = await response.json();
        console.log(`✓ Start streaming: ${piId}`, data);
    } catch (error) {
        console.error(`✗ Error starting streaming for ${piId}:`, error);
    }
}

// Stop Streaming
async function stopStreaming(piId) {
    try {
        const response = await fetch(`/api/command/${piId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'stop' })
        });
        const data = await response.json();
        console.log(`✓ Stop streaming: ${piId}`, data);
    } catch (error) {
        console.error(`✗ Error stopping streaming for ${piId}:`, error);
    }
}

// Load Tournaments
async function loadTournaments() {
    try {
        const response = await fetch('/api/tournaments');
        const data = await response.json();
        tournamentData = data.tournaments;
        renderTournaments(data.tournament_list, data.tournaments);
        updateHeaderStats();
    } catch (error) {
        console.error("Error loading tournaments:", error);
    }
}

// Render Tournaments
function renderTournaments(tournamentList, tournaments) {
    const container = document.getElementById('tournament-grid');
    if (!container) return;
    
    container.innerHTML = '';
    
    if (tournamentList.length === 0) {
        container.innerHTML = '<div class="empty-state"><i class="ri-folder-line"></i><p>No tournaments found</p></div>';
        return;
    }
    
    tournamentList.forEach(tournament => {
        const piCount = Object.keys(tournaments[tournament] || {}).length;
        let totalFens = 0;
        for (const piId in tournaments[tournament]) {
            totalFens += tournaments[tournament][piId].length;
        }
        
        const card = document.createElement('div');
        card.className = 'tournament-card';
        card.innerHTML = `
            <div class="tournament-icon">
                <i class="ri-trophy-line"></i>
            </div>
            <div class="tournament-name">${tournament}</div>
            <div class="tournament-meta">
                <span><i class="ri-device-line"></i> ${piCount} devices</span>
                <span><i class="ri-file-list-line"></i> ${totalFens} moves</span>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Refresh Tournaments
async function refreshTournaments() {
    await loadTournaments();
}

// Update Header Stats
function updateHeaderStats() {
    const activeDevices = Object.values(piStates).filter(state => 
        state.state === 'STREAMING' || state.state === 'IDLE'
    ).length;
    
    const tournamentCount = Object.keys(tournamentData).length;
    
    const activeEl = document.getElementById('active-devices');
    const tournamentEl = document.getElementById('tournament-count');
    const deviceBadge = document.getElementById('device-badge');
    
    if (activeEl) activeEl.textContent = activeDevices;
    if (tournamentEl) tournamentEl.textContent = tournamentCount;
    if (deviceBadge) deviceBadge.textContent = Object.keys(piStates).length;
}

// Format Last Seen
function formatLastSeen(timestamp) {
    if (!timestamp) return 'Never';
    const now = Math.floor(Date.now() / 1000);
    const diff = now - timestamp;
    
    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log("🚀 App initialized");
});