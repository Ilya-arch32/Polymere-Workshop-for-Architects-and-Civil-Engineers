/**
 * Polymère — Space Syntax Visualizer (Dual-Mode: Placement + VGA Heatmap)
 *
 * Interactive Three.js visualizer for VGA (Visibility Graph Analysis).
 * Operates in two sequential modes:
 *
 *   MODE 1 — PLACEMENT:
 *     User clicks on the IFC mesh to place observer points.
 *     A glassmorphic floating panel shows observer count, Run Analysis, and Clear buttons.
 *     Eye-level offset (+1.5m Y) is applied automatically.
 *
 *   MODE 2 — VISUALIZATION:
 *     After analysis completes, renders an instanced heatmap of observer nodes
 *     colored by connectivity. On hover, draws a "starburst" of visibility edges
 *     while ghosting the IFC building model.
 *
 * Performance notes:
 *   • Nodes use THREE.InstancedMesh — O(1) draw calls regardless of N.
 *   • Edges are rebuilt per-hover as a single THREE.LineSegments geometry.
 *   • Raycaster is throttled to one cast per animation frame.
 *
 * Usage:
 *   const viz = new SpaceSyntaxVisualizer(scene, camera, renderer, ifcModel, {
 *       solver,       // Initialized SpaceSyntaxSolver instance
 *       voxelBuffer,  // GPU voxel buffer
 *       gridConfig,   // { bounds, resolution, dimensions }
 *       onAnalysisComplete(results) { ... }  // Callback after analysis
 *   });
 *
 *   viz.enterPlacementMode();   // Start observer placement
 *   viz.dispose();              // Full teardown
 */
import * as THREE from 'three';
import { BehaviorSimulator } from './BehaviorSimulator.js';

export class SpaceSyntaxVisualizer {

    // ═══════════════════════════════════════════════════════════════════════
    //  Construction
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @param {THREE.Scene}    scene      — The active Three.js scene.
     * @param {THREE.Camera}   camera     — Perspective camera for raycasting.
     * @param {THREE.Renderer} renderer   — WebGL renderer (for domElement events).
     * @param {THREE.Object3D} ifcModel   — The IFC building mesh for raycasting/ghosting.
     * @param {Object}         options    — Configuration object.
     * @param {Object}         options.solver      — Initialized SpaceSyntaxSolver instance.
     * @param {GPUBuffer}      options.voxelBuffer — GPU voxel buffer for analysis.
     * @param {Object}         options.gridConfig  — Voxel grid configuration.
     * @param {Function}       [options.onAnalysisComplete] — Callback fired with results after analysis.
     */
    constructor(scene, camera, renderer, ifcModel, options = {}) {
        this.scene    = scene;
        this.camera   = camera;
        this.renderer = renderer;
        this.ifcModel = ifcModel;

        // ── Config from options ──
        this.solver      = options.solver      || null;
        this.voxelBuffer = options.voxelBuffer  || null;
        this.gridConfig  = options.gridConfig   || null;
        this.onAnalysisComplete = options.onAnalysisComplete || null;

        // ── Existing VGA data (for direct visualization without placement) ──
        this.data = options.data || null;

        // ── Eye-level offset (meters) ──
        this.eyeLevelOffset = 1.5;

        // ── Placement state ──
        this.mode              = 'idle';        // 'idle', 'placement', 'visualization'
        this.placedObservers   = [];            // Array<{x, y, z}> in WORLD coords (eye-level adjusted)
        this.observerMarkers   = [];            // Array<THREE.Mesh> for placed observer spheres
        this.previewSphere     = null;          // Semi-transparent preview sphere
        this.overlayPanel      = null;          // DOM element for the floating UI panel

        // ── Visualization state (Mode 2) ──
        this.nodesMesh         = null;          // THREE.InstancedMesh for heatmap nodes
        this.edgeLines         = null;          // THREE.LineSegments (recreated per hover)
        this.visionHelper      = null;          // Wireframe frustum for Partial Isovist
        this.observerPoints    = null;          // Voxel-coord observer points (from solver result)
        this.adjacencyMatrix   = null;          // Uint32Array (N×N flat)
        this.integrationArr    = null;          // Float64Array (N)
        this.N                 = 0;             // Number of observer nodes
        this.worldPositions    = null;          // Float32Array (3 per point)

        // ── Interaction state ──
        this.raycaster         = new THREE.Raycaster();
        this.mouse             = new THREE.Vector2();
        this.hoveredIndex      = -1;
        this._enabled          = false;
        this._rafPending       = false;

        // ── IFC ghosting cache ──
        this._originalMats     = new Map();
        this._isGhosted        = false;

        // ── Placement state ──
        this.intersectDepthIndex = 0;
        this._lastMousePos     = new THREE.Vector2();

        // ── Bound event handlers (for clean removal) ──
        this._onPlacementClickBound     = this._onPlacementClick.bind(this);
        this._onPlacementMouseMoveBound = this._onPlacementMouseMove.bind(this);
        this._onPlacementKeyDownBound   = this._onPlacementKeyDown.bind(this);
        this._onHoverMouseMoveBound     = this._onHoverMouseMove.bind(this);
        this._onHoverMouseLeaveBound    = this._onHoverMouseLeave.bind(this);

        console.log('[SpaceSyntaxVisualizer] Created (dual-mode: placement + visualization).');
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Public API — Mode 1: Observer Placement
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Enter observer placement mode.
     * Attaches click/mousemove listeners, creates preview sphere and UI panel.
     */
    enterPlacementMode() {
        if (this.mode === 'placement') return;
        this.mode = 'placement';
        this.placedObservers = [];
        this.observerMarkers = [];

        const canvas = this.renderer.domElement;
        canvas.style.cursor = 'crosshair';

        // Attach placement event listeners
        canvas.addEventListener('click', this._onPlacementClickBound);
        canvas.addEventListener('mousemove', this._onPlacementMouseMoveBound);
        window.addEventListener('keydown', this._onPlacementKeyDownBound);

        // Create preview sphere (follows mouse on IFC mesh)
        this._createPreviewSphere();

        // Create the floating UI panel
        this._createOverlayPanel();

        console.log('[SpaceSyntaxVisualizer] Placement mode active.');
    }

    /**
     * Exit placement mode — remove listeners, cleanup preview sphere.
     * Does NOT remove placed markers (they persist for analysis).
     */
    exitPlacementMode() {
        if (this.mode !== 'placement') return;

        const canvas = this.renderer.domElement;
        canvas.style.cursor = 'default';
        canvas.removeEventListener('click', this._onPlacementClickBound);
        canvas.removeEventListener('mousemove', this._onPlacementMouseMoveBound);
        window.removeEventListener('keydown', this._onPlacementKeyDownBound);

        // Remove preview sphere
        if (this.previewSphere) {
            this.scene.remove(this.previewSphere);
            this.previewSphere.geometry?.dispose();
            this.previewSphere.material?.dispose();
            this.previewSphere = null;
        }

        console.log('[SpaceSyntaxVisualizer] Placement mode exited.');
    }

    /**
     * Clear all placed observer markers and reset the counter.
     */
    clearObservers() {
        // Remove sphere markers from the scene
        for (const marker of this.observerMarkers) {
            this.scene.remove(marker);
            marker.geometry?.dispose();
            marker.material?.dispose();
        }
        this.observerMarkers = [];
        this.placedObservers = [];

        // Update the UI panel counter
        this._updateOverlayPanel();
        console.log('[SpaceSyntaxVisualizer] All observers cleared.');
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Public API — Mode 2: Visualization (Instanced Heatmap)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Enter visualization mode with pre-computed solver results.
     * Builds the heatmap and enables hover interaction.
     *
     * @param {Object} data — SpaceSyntaxSolver result with observerPoints, adjacencyMatrix, perNode, gridConfig.
     */
    enterVisualizationMode(data) {
        // Clean up placement mode artifacts
        this.exitPlacementMode();
        this._removeOverlayPanel();
        this._clearObserverMarkers();

        this.mode = 'visualization';
        this.data = data;
        this.gridConfig = data.gridConfig || this.gridConfig;

        if (!this.gridConfig || !this.gridConfig.bounds) {
            console.error('[SpaceSyntaxVisualizer] Invalid gridConfig for visualization:', this.gridConfig);
            return;
        }

        // ── Extract solver data ──
        this.observerPoints  = data.observerPoints;
        this.adjacencyMatrix = data.adjacencyMatrix;
        this.integrationArr  = data.perNode?.integration ?? new Float64Array(0);
        this.N               = this.observerPoints.length;

        // ── Compute world positions ──
        this.worldPositions = new Float32Array(this.N * 3);
        this._computeWorldPositions();

        // ── Build instanced heatmap nodes ──
        this._buildNodes();

        // ── Enable hover interaction ──
        this._enableHoverInteraction();

        console.log(`[SpaceSyntaxVisualizer] Visualization mode active: ${this.N} nodes.`);
    }

    /**
     * Full teardown — remove everything, restore IFC model, unbind events.
     */
    dispose() {
        this.exitPlacementMode();
        this._disableHoverInteraction();
        this._removeOverlayPanel();
        this._clearObserverMarkers();

        // Remove instanced nodes
        if (this.nodesMesh) {
            this.scene.remove(this.nodesMesh);
            this.nodesMesh.geometry.dispose();
            this.nodesMesh.material.dispose();
            this.nodesMesh = null;
        }

        // Remove any remaining edges
        this._clearEdges();

        // Remove vision helper if any
        if (this.visionHelper) {
            this.scene.remove(this.visionHelper);
            const lines = this.visionHelper.getObjectByName('Wireframe');
            if (lines) {
                lines.geometry?.dispose();
                lines.material?.dispose();
            }
            const mesh = this.visionHelper.getObjectByName('Solid');
            if (mesh) {
                mesh.geometry?.dispose();
                mesh.material?.dispose();
            }
            this.visionHelper = null;
        }

        // Restore IFC if still ghosted
        this._unghostIFC();

        this.mode = 'idle';
        console.log('[SpaceSyntaxVisualizer] Disposed.');
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Private — Placement Mode: Raycaster Handlers
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Handle mouse move during placement — update preview sphere position.
     */
    _onPlacementMouseMove(event) {
        if (this.mode !== 'placement' || !this.previewSphere) return;

        // Ignore if clicking on UI overlays
        if (event.target.closest('#ssx-observer-overlay')) return;

        const rect = this.renderer.domElement.getBoundingClientRect();
        const newX =  ((event.clientX - rect.left) / rect.width)  * 2 - 1;
        const newY = -((event.clientY - rect.top)  / rect.height) * 2 + 1;

        // Reset depth index if mouse moved significantly (e.g. looking at a new target)
        if (Math.abs(newX - this._lastMousePos.x) > 0.01 || Math.abs(newY - this._lastMousePos.y) > 0.01) {
            this.intersectDepthIndex = 0;
            this._lastMousePos.set(newX, newY);
        }

        this.mouse.set(newX, newY);
        this._updatePreviewRaycast();
    }

    /**
     * Handle key down for depth cycling (Tab key)
     */
    _onPlacementKeyDown(event) {
        if (this.mode !== 'placement') return;
        if (event.key === 'Tab') {
            event.preventDefault(); // Prevent browser focus jumping
            this.intersectDepthIndex++;
            this._updatePreviewRaycast();
        }
    }

    /**
     * Re-run the raycast using current mouse coordinates and intersect depth
     */
    _updatePreviewRaycast() {
        if (!this.previewSphere) return;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);

        // Collect meshes to raycast against (exclude our own markers)
        const meshes = [];
        this.scene.traverse(obj => {
            if (obj.isMesh && obj.visible &&
                obj.name !== 'SSX_PreviewMarker' &&
                obj.name !== 'SSX_ObserverMarker' &&
                obj.name !== 'SSX_VGA_Nodes') {
                meshes.push(obj);
            }
        });

        const intersects = this.raycaster.intersectObjects(meshes, true);
        if (intersects.length > 0) {
            // Safety check: wrap around if depth index exceeds hits
            if (this.intersectDepthIndex >= intersects.length) {
                this.intersectDepthIndex = 0;
            }

            const hitPoint = intersects[this.intersectDepthIndex].point.clone();

            // Show preview at the clicked point + eye-level offset
            hitPoint.y += this.eyeLevelOffset;
            this.previewSphere.position.copy(hitPoint);
            this.previewSphere.visible = true;
        } else {
            this.previewSphere.visible = false;
        }
    }

    /**
     * Handle click during placement — place an observer point.
     */
    _onPlacementClick(event) {
        if (this.mode !== 'placement') return;

        // Ignore clicks on UI overlays
        if (event.target.closest('#ssx-observer-overlay')) return;

        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x =  ((event.clientX - rect.left) / rect.width)  * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top)  / rect.height) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);

        // Collect meshes to raycast against (exclude our own markers)
        const meshes = [];
        this.scene.traverse(obj => {
            if (obj.isMesh && obj.visible &&
                obj.name !== 'SSX_PreviewMarker' &&
                obj.name !== 'SSX_ObserverMarker' &&
                obj.name !== 'SSX_VGA_Nodes') {
                meshes.push(obj);
            }
        });

        const intersects = this.raycaster.intersectObjects(meshes, true);
        if (intersects.length === 0) return;

        // Safety check: wrap around if depth index exceeds hits
        if (this.intersectDepthIndex >= intersects.length) {
            this.intersectDepthIndex = 0;
        }

        const hitPoint = intersects[this.intersectDepthIndex].point.clone();

        // CRITICAL: Apply eye-level offset (+1.5m on Y axis for human vision simulation)
        const observerPoint = {
            x: hitPoint.x,
            y: hitPoint.y + this.eyeLevelOffset,
            z: hitPoint.z
        };

        // Store the world-coordinate observer point
        this.placedObservers.push(observerPoint);

        // Place a visual marker at the observer location
        this._addObserverMarker(observerPoint);

        // Update the UI panel
        this._updateOverlayPanel();

        console.log(`[SpaceSyntaxVisualizer] Observer #${this.placedObservers.length} placed at ` +
                     `(${observerPoint.x.toFixed(2)}, ${observerPoint.y.toFixed(2)}, ${observerPoint.z.toFixed(2)}) ` +
                     `[+${this.eyeLevelOffset}m eye-level]`);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Private — Placement Mode: Visual Markers
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Create the semi-transparent preview sphere that follows the mouse.
     */
    _createPreviewSphere() {
        const geometry = new THREE.SphereGeometry(0.2, 16, 16);
        const material = new THREE.MeshBasicMaterial({
            color: 0x4CAF50,
            transparent: true,
            opacity: 0.5,
            depthTest: false,
            depthWrite: false,
        });
        this.previewSphere = new THREE.Mesh(geometry, material);
        this.previewSphere.name = 'SSX_PreviewMarker';
        this.previewSphere.visible = false;
        this.previewSphere.renderOrder = 999;
        this.scene.add(this.previewSphere);
    }

    /**
     * Add a solid observer marker sphere at the given world position.
     *
     * @param {{x,y,z}} pos — World position (eye-level adjusted).
     */
    _addObserverMarker(pos) {
        const geometry = new THREE.SphereGeometry(0.2, 16, 16);
        const material = new THREE.MeshBasicMaterial({
            color: 0x2d5016,    // Biophilic green (matches --accent-primary)
            transparent: true,
            opacity: 0.9,
            depthTest: false,
            depthWrite: false,
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(pos.x, pos.y, pos.z);
        sphere.name = 'SSX_ObserverMarker';
        sphere.renderOrder = 998;
        this.scene.add(sphere);
        this.observerMarkers.push(sphere);
    }

    /**
     * Remove all observer markers from the scene.
     */
    _clearObserverMarkers() {
        for (const marker of this.observerMarkers) {
            this.scene.remove(marker);
            marker.geometry?.dispose();
            marker.material?.dispose();
        }
        this.observerMarkers = [];
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Private — Floating UI Panel (Glassmorphic Card)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Create the floating glassmorphic observer placement panel.
     * Mirrors the Acoustic Source Placement UI style.
     */
    _createOverlayPanel() {
        const container = document.getElementById('ifc-preview-container');
        if (!container) return;
        container.style.position = 'relative';

        // Remove existing panel if any
        this._removeOverlayPanel();

        const overlay = document.createElement('div');
        overlay.id = 'ssx-observer-overlay';
        overlay.className = 'glass-panel';
        overlay.style.cssText = `
            position: absolute; top: 20px; left: 50%; transform: translateX(-50%);
            padding: 16px 24px; border-radius: var(--radius-lg);
            font-family: var(--font-primary, 'Inter', sans-serif); font-size: 14px;
            z-index: 1000; text-align: center;
            min-width: 300px;
        `;
        container.appendChild(overlay);
        this.overlayPanel = overlay;

        this._updateOverlayPanel();
    }

    /**
     * Update the overlay panel content with current observer count.
     */
    _updateOverlayPanel() {
        if (!this.overlayPanel) return;

        const count = this.placedObservers.length;
        const isDisabled = count === 0;

        this.overlayPanel.innerHTML = `
            <div style="margin-bottom: 8px; font-weight: 600; font-size: 16px;">Space Syntax Observers</div>
            <div style="margin-bottom: 16px; opacity: 0.7; font-size: 13px;">Click on the model to place observer points</div>
            <div style="margin-bottom: 16px; background: rgba(0,0,0,0.05); padding: 8px; border-radius: var(--radius-md, 12px);">
                Observers placed: <strong style="color: var(--accent-primary, #2d5016);">${count}</strong>
            </div>
            <div style="display: flex; gap: 10px; justify-content: center;">
                <button id="ssx-run-btn" style="
                    background: var(--accent-primary, #2d5016); border: none; color: white; padding: 8px 20px;
                    border-radius: var(--radius-md, 12px); cursor: pointer; font-weight: 500;
                    font-family: var(--font-primary, 'Inter', sans-serif); transition: opacity 0.2s;
                    ${isDisabled ? 'opacity: 0.5; cursor: not-allowed;' : ''}
                " ${isDisabled ? 'disabled' : ''}>Run Analysis</button>
                <button id="ssx-clear-btn" style="
                    background: transparent; border: 1px solid var(--border-medium, rgba(0,0,0,0.15)); color: var(--text-primary, #333); 
                    padding: 8px 16px; border-radius: var(--radius-md, 12px); cursor: pointer;
                    font-family: var(--font-primary, 'Inter', sans-serif);
                ">Clear Points</button>
            </div>
        `;

        // Attach event listeners to the button elements
        const runBtn = document.getElementById('ssx-run-btn');
        const clearBtn = document.getElementById('ssx-clear-btn');

        if (runBtn) {
            runBtn.addEventListener('click', () => this._onRunAnalysis());
        }
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearObservers());
        }
    }

    /**
     * Show a loading state in the overlay panel during analysis.
     * Pass labels reflect the new 4-pass GPU pipeline.
     */
    _showLoadingState() {
        if (!this.overlayPanel) return;

        const count = this.placedObservers.length;
        this.overlayPanel.innerHTML = `
            <div style="font-weight: 600; margin-bottom: 8px;">⏳ Running 3D Space Syntax Analysis...</div>
            <div style="font-size: 13px;">Processing <span style="font-weight:bold; color:var(--accent-primary, #2d5016);">${count}</span> observer(s)</div>
            <div style="margin-top: 6px; font-size: 12px; opacity: 0.7;">
                🔵 Pass 0 — GPU Heading Generation<br>
                🟢 Pass A — 3D Spherical Isovist (128 rays)<br>
                🟡 Pass B — Visibility Graph Adjacency<br>
                🟠 Pass C — Graph Metrics (BFS)
            </div>
        `;
    }

    /**
     * Remove the floating overlay panel from the DOM.
     */
    _removeOverlayPanel() {
        const existing = document.getElementById('ssx-observer-overlay');
        if (existing) existing.remove();
        this.overlayPanel = null;
    }


    // ═══════════════════════════════════════════════════════════════════════
    //  Private — Run Analysis Trigger


    /**
     * Called when the user clicks "Run Analysis".
     * Sends placed observers to the solver and transitions to visualization mode.
     */
    async _onRunAnalysis() {
        if (this.placedObservers.length === 0) {
            console.warn('[SpaceSyntaxVisualizer] No observers to analyze.');
            return;
        }

        if (!this.solver || !this.voxelBuffer) {
            console.error('[SpaceSyntaxVisualizer] Solver or voxelBuffer not available.');
            alert('Space Syntax solver not available. Re-run the AHI analysis first.');
            return;
        }

        // Show loading state
        this._showLoadingState();

        try {
            console.log(`[SpaceSyntaxVisualizer] Running analysis with ${this.placedObservers.length} observers...`);

            // Call solver with user-placed world-coordinate observers
            const ssResults = await this.solver.analyze(this.voxelBuffer, {
                observers: this.placedObservers
            });

            if (!ssResults || ssResults.status === 'pending') {
                console.error('[SpaceSyntaxVisualizer] Solver returned unexpected result:', ssResults);
                this._updateOverlayPanel(); // Restore panel
                return;
            }

            console.log('[SpaceSyntaxVisualizer] Analysis complete:', ssResults);

            // Update global results — map new 3D metric names from solver
            const globalResults = {
                // ── Space Syntax graph metrics ──────────────────────────────
                integration:      ssResults.globalIntegration,
                connectivity:     ssResults.avgConnectivity,      // % of max
                rawConnectivity:  ssResults.rawConnectivity,      // average neighbour count
                meanDepth:        ssResults.meanDepth,
                intelligibility:  ssResults.intelligibility,

                // ── 3D Isovist metrics (internal voxel units — for evaluator thresholds) ──
                avgVolume:        ssResults.avgVolume,             // voxel³
                avgSurfaceArea:   ssResults.avgSurfaceArea,        // voxel²
                avgCompactness:   ssResults.avgCompactness,        // [0,1] 3D isoperimetric
                spatialChaos:     ssResults.spatialChaos,
                visualComplexity: ssResults.visualComplexity,

                // ── Physical unit mirrors (UI display only) ─────────────────
                avgVolumeM3:      ssResults.avgVolumeM3,           // m³
                avgSurfaceAreaM2: ssResults.avgSurfaceAreaM2,      // m²

                // ── Legacy aliases kept for NeuroaestheticEvaluator ─────────
                visibilityIndex:  ssResults.avgCompactness,        // was 2D, now 3D compactness
                visibility:       ssResults.avgCompactness,

                // ── Structural metadata ─────────────────────────────────────
                observerPoints:   ssResults.observerPoints   || null,
                adjacencyMatrix:  ssResults.adjacencyMatrix  || null,
                perNode:          ssResults.perNode          || null,
                gridConfig:       ssResults.gridConfig       || this.gridConfig || null,
                status: 'complete'
            };

            window.currentSpaceSyntaxData = globalResults;

            if (window.AHI_Results) {
                window.AHI_Results.spaceSyntax = globalResults;
            }

            // Immediately Trigger Save to persist new interactive metrics generated
            if (typeof window.triggerProjectSave === 'function') {
                window.triggerProjectSave();
            } else {
                window.dispatchEvent(new CustomEvent('neighbly-autosave'));
                console.log("[SpaceSyntax] Emitted neighbly-autosave event to trigger save.");
            }

            // Fire completion callback
            if (this.onAnalysisComplete) {
                this.onAnalysisComplete(globalResults);
            }

            // Transition to visualization mode
            this.enterVisualizationMode(globalResults);

            // Instantiate BehaviorSimulator natively
            if (window.behaviorSimulator) {
                window.behaviorSimulator.destroy();
            }
            window.behaviorSimulator = new BehaviorSimulator(this.scene, globalResults);
            window.behaviorSimulator.spawnAgents();

        } catch (error) {
            console.error('[SpaceSyntaxVisualizer] Analysis failed:', error);
            alert('Space Syntax analysis failed: ' + error.message);
            this._updateOverlayPanel(); // Restore panel
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Private — Coordinate Conversion (Voxel → World for Visualization)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Convert all observer voxel coords → world-space positions and cache
     * them in this.worldPositions (flat Float32Array, 3 floats per point).
     */
    _computeWorldPositions() {
        if (!this.gridConfig) {
            console.error("[SpaceSyntaxVisualizer] Received invalid gridConfig:", this.gridConfig);
            return;
        }
        let printedNodes = 0;
        
        console.log("[SpaceSyntaxVisualizer] Raw bounds object:", this.gridConfig.bounds);
        
        let minX = 0, minY = 0, minZ = 0;
        const b = this.gridConfig.bounds;
        
        if (b) {
            if (b.min !== undefined && b.min.x !== undefined) {
                // Standard Three.js Box3 or Vector3 format
                minX = b.min.x; minY = b.min.y; minZ = b.min.z;
            } else if (b.minX !== undefined) {
                // Flattened object format
                minX = b.minX; minY = b.minY; minZ = b.minZ;
            } else if (Array.isArray(b) && b.length >= 2) {
                // Array format [min, max] or [ [x,y,z], [x,y,z] ]
                const minArr = b[0];
                minX = minArr.x ?? minArr[0] ?? 0;
                minY = minArr.y ?? minArr[1] ?? 0;
                minZ = minArr.z ?? minArr[2] ?? 0;
            } else if (b.x !== undefined) {
                // Direct vector format
                minX = b.x; minY = b.y; minZ = b.z;
            }
        }
        
        const min = { x: minX, y: minY, z: minZ };
        console.log("[SpaceSyntaxVisualizer] Extracted min coordinates:", min);

        const resolution = this.gridConfig.resolution || 0.1;
        const halfRes = resolution / 2.0;

        for (let i = 0; i < this.N; i++) {
            const pt = this.observerPoints[i];
            
            // Smart Check: Are they already world coords?
            // Voxel indices are strict integers. If it has a decimal, it's likely a world float.
            const isVoxelIndex = Number.isInteger(pt.x) && Number.isInteger(pt.y) && Number.isInteger(pt.z);
            
            let wx, wy, wz;
            if (isVoxelIndex) {
                // Exact world position mapping
                wx = pt.x * resolution + min.x + halfRes;
                wy = pt.y * resolution + min.y + halfRes;
                wz = pt.z * resolution + min.z + halfRes;
            } else {
                // Already world coordinates
                wx = pt.x;
                wy = pt.y;
                wz = pt.z;
            }

            this.worldPositions[i * 3 + 0] = wx;
            this.worldPositions[i * 3 + 1] = wy;
            this.worldPositions[i * 3 + 2] = wz;

            if (printedNodes < 3) {
                console.log(`[SpaceSyntaxVisualizer] Node ${i} WorldPos = (${wx.toFixed(2)}, ${wy.toFixed(2)}, ${wz.toFixed(2)}) -- Source: {x:${pt.x}, y:${pt.y}, z:${pt.z}} type: ${isVoxelIndex ? 'voxel' : 'world'}`);
                printedNodes++;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Private — Instanced Heatmap Nodes (Mode 2)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Create the InstancedMesh for all observer nodes, colored by
     * the per-node connectivity value using an HSL gradient.
     */
    _buildNodes() {
        // Sensible radius for world coordinates so nodes fit inside rooms
        const sphereRadius = 0.15;
        const geometry = new THREE.SphereGeometry(sphereRadius, 12, 8);
        const material = new THREE.MeshBasicMaterial({ 
            color: 0xffffff,
            depthTest: false,
            depthWrite: false, // required for X-Ray rendering
            transparent: true
        });

        const mesh = new THREE.InstancedMesh(geometry, material, this.N);
        mesh.name = 'SSX_VGA_Nodes';
        mesh.frustumCulled = false; // always render all instances

        // ── Calculate degree centrality (Connectivity) on-the-fly ──
        const connArray = new Float64Array(this.N);
        let maxConn = 0, minConn = Infinity;

        // Calculate how many nodes each node can see
        for (let i = 0; i < this.N; i++) {
            let visibleCount = 0;
            for (let j = 0; j < this.N; j++) {
                // Assuming adjacencyMatrix is a 1D flat array of size N*N
                if (this.adjacencyMatrix[i * this.N + j] === 1) visibleCount++; 
            }
            connArray[i] = visibleCount;
            if (visibleCount > maxConn) maxConn = visibleCount;
            if (visibleCount < minConn) minConn = visibleCount;
        }

        // ── Set per-instance transform + HSL color ──
        const dummy  = new THREE.Object3D();
        const color  = new THREE.Color();

        for (let i = 0; i < this.N; i++) {
            // Position
            dummy.position.set(
                this.worldPositions[i * 3 + 0],
                this.worldPositions[i * 3 + 1],
                this.worldPositions[i * 3 + 2]
            );
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);

            // Color — HSL Gradient (Blue → Red based on Connectivity)
            // Normalize between 0 and 1
            const norm = maxConn === minConn ? 0.5 : (connArray[i] - minConn) / (maxConn - minConn);
            // Hue: 0.66 (Blue) down to 0.0 (Red)
            const hue = (1.0 - norm) * 0.66; 
            color.setHSL(hue, 1.0, 0.5);
            mesh.setColorAt(i, color);
        }

        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor.needsUpdate  = true;

        this.scene.add(mesh);
        this.nodesMesh = mesh;
    }

    /**
     * Jet color ramp: maps t ∈ [0,1] → Color.
     *   0.00 → deep blue   (cold / low integration)
     *   0.25 → cyan
     *   0.50 → green
     *   0.75 → yellow
     *   1.00 → red          (hot / high integration)
     *
     * @param {number}      t     — Normalised value [0, 1].
     * @param {THREE.Color} color — Output color object (mutated in-place).
     */
    _jetColor(t, color) {
        // Piecewise linear interpolation through 5 control points
        const ramp = [
            { t: 0.00, r: 0.00, g: 0.00, b: 0.70 },  // deep blue
            { t: 0.25, r: 0.00, g: 0.80, b: 1.00 },  // cyan
            { t: 0.50, r: 0.10, g: 0.90, b: 0.10 },  // green
            { t: 0.75, r: 1.00, g: 0.85, b: 0.00 },  // yellow
            { t: 1.00, r: 0.90, g: 0.00, b: 0.00 },  // red
        ];

        // Clamp
        const tc = Math.max(0, Math.min(1, t));

        for (let i = 0; i < ramp.length - 1; i++) {
            if (tc >= ramp[i].t && tc <= ramp[i + 1].t) {
                const f = (tc - ramp[i].t) / (ramp[i + 1].t - ramp[i].t);
                color.setRGB(
                    ramp[i].r + (ramp[i + 1].r - ramp[i].r) * f,
                    ramp[i].g + (ramp[i + 1].g - ramp[i].g) * f,
                    ramp[i].b + (ramp[i + 1].b - ramp[i].b) * f
                );
                return;
            }
        }
        // Fallback (shouldn't reach)
        color.setRGB(ramp[ramp.length - 1].r, ramp[ramp.length - 1].g, ramp[ramp.length - 1].b);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Private — Hover Raycaster Interaction (Mode 2)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Enable mouse hover interaction for visualization mode.
     */
    _enableHoverInteraction() {
        if (this._enabled) return;
        this._enabled = true;
        this.renderer.domElement.addEventListener('mousemove', this._onHoverMouseMoveBound);
        this.renderer.domElement.addEventListener('mouseleave', this._onHoverMouseLeaveBound);
        console.log('[SpaceSyntaxVisualizer] Hover interaction enabled.');
    }

    /**
     * Disable hover interaction without disposing geometry.
     */
    _disableHoverInteraction() {
        if (!this._enabled) return;
        this._enabled = false;
        this.renderer.domElement.removeEventListener('mousemove', this._onHoverMouseMoveBound);
        this.renderer.domElement.removeEventListener('mouseleave', this._onHoverMouseLeaveBound);
        this._clearHover();
        console.log('[SpaceSyntaxVisualizer] Hover interaction disabled.');
    }

    /**
     * mousemove handler — throttled to one raycast per animation frame.
     */
    _onHoverMouseMove(event) {
        // Compute normalised device coordinates
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x =  ((event.clientX - rect.left) / rect.width)  * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top)  / rect.height) * 2 + 1;

        // Throttle: schedule a raycast on the next animation frame
        if (!this._rafPending) {
            this._rafPending = true;
            requestAnimationFrame(() => {
                this._rafPending = false;
                this._performRaycast();
            });
        }
    }

    /**
     * mouseleave handler — clear all hover visuals when the cursor
     * leaves the renderer canvas entirely.
     */
    _onHoverMouseLeave() {
        this._clearHover();
    }

    /**
     * Execute the actual raycast against the instanced mesh and update
     * hover visuals accordingly.
     */
    _performRaycast() {
        if (!this._enabled || !this.nodesMesh) return;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const hits = this.raycaster.intersectObject(this.nodesMesh);

        if (hits.length > 0) {
            const idx = hits[0].instanceId;
            if (idx !== this.hoveredIndex) {
                // New node hovered
                this.hoveredIndex = idx;
                this._ghostIFC();
                this._drawEdges(idx);
                this._showVisionHelper(idx);
            }
        } else {
            // No hit — clear everything
            if (this.hoveredIndex !== -1) {
                this._clearHover();
            }
        }
    }

    /**
     * Reset all hover visuals (edges + ghosting).
     */
    _clearHover() {
        this.hoveredIndex = -1;
        this._clearEdges();
        this._unghostIFC();
        if (this.visionHelper) this.visionHelper.visible = false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Private — IFC Ghosting
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Make the IFC building model semi-transparent so the starburst
     * visibility lines are clearly visible.
     */
    _ghostIFC() {
        if (this._isGhosted || !this.ifcModel) return;
        this._isGhosted = true;

        this.ifcModel.traverse((child) => {
            if (!child.isMesh) return;

            // Handle both single materials and material arrays
            const materials = Array.isArray(child.material) ? child.material : [child.material];
            for (const mat of materials) {
                if (!mat) continue;

                // Cache original values (only if not already cached)
                if (!this._originalMats.has(mat)) {
                    this._originalMats.set(mat, {
                        opacity:    mat.opacity,
                        transparent: mat.transparent,
                        depthWrite:  mat.depthWrite,
                    });
                }

                // Apply ghost effect
                mat.opacity     = 0.15;
                mat.transparent = true;
                mat.depthWrite  = false;
                mat.needsUpdate = true;
            }
        });
    }

    /**
     * Restore the IFC building model to its original opacity.
     */
    _unghostIFC() {
        if (!this._isGhosted) return;
        this._isGhosted = false;

        for (const [mat, orig] of this._originalMats.entries()) {
            mat.opacity     = orig.opacity;
            mat.transparent = orig.transparent;
            mat.depthWrite  = orig.depthWrite;
            mat.needsUpdate = true;
        }
        this._originalMats.clear();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Private — Visibility Edges (Starburst)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Draw a "starburst" of line segments from the hovered node to every
     * other node that is directly visible according to the adjacency matrix.
     *
     * @param {number} nodeIdx — Index of the hovered observer node.
     */
    _drawEdges(nodeIdx) {
        // Remove previous edges first
        this._clearEdges();

        const N   = this.N;
        const adj = this.adjacencyMatrix;
        const wp  = this.worldPositions;

        // Source position
        const sx = wp[nodeIdx * 3 + 0];
        const sy = wp[nodeIdx * 3 + 1];
        const sz = wp[nodeIdx * 3 + 2];

        // Collect line segment endpoints: [sx, sy, sz, tx, ty, tz, ...]
        const vertices = [];
        const rowOffset = nodeIdx * N;

        for (let j = 0; j < N; j++) {
            if (j === nodeIdx) continue;            // skip self
            if (adj[rowOffset + j] !== 1) continue; // not visible

            vertices.push(
                sx, sy, sz,                          // from (hovered node)
                wp[j * 3 + 0], wp[j * 3 + 1], wp[j * 3 + 2]  // to (visible neighbor)
            );
        }

        if (vertices.length === 0) return; // isolated node — no edges

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));

        const material = new THREE.LineBasicMaterial({
            color:       0x00e5ff,  // bright cyan
            transparent: true,
            opacity:     0.55,
            depthWrite:  false,
            linewidth:   1,         // note: WebGL always renders 1px lines
        });

        this.edgeLines = new THREE.LineSegments(geometry, material);
        this.edgeLines.name = 'SSX_VGA_Edges';
        this.edgeLines.frustumCulled = false;
        this.scene.add(this.edgeLines);
    }

    /**
     * Remove the current starburst edge lines from the scene and
     * dispose their GPU resources.
     */
    _clearEdges() {
        if (!this.edgeLines) return;
        this.scene.remove(this.edgeLines);
        this.edgeLines.geometry.dispose();
        this.edgeLines.material.dispose();
        this.edgeLines = null;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Private — Partial Isovist (Cone of Vision) Helper
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Show the wireframe Cone of Vision helper for the hovered node
     *
     * @param {number} nodeIdx — Index of the hovered observer node.
     */
    _showVisionHelper(nodeIdx) {
        const wp = this.worldPositions;
        const pt = this.observerPoints[nodeIdx];
        
        // Use custom FOV or fall back to defaults (120 H, 90 V)
        const fovH = ((pt && pt.fov_horizontal) ? pt.fov_horizontal : 120) * (Math.PI / 180);
        const fovV = ((pt && pt.fov_vertical) ? pt.fov_vertical : 90) * (Math.PI / 180);

        this._updateVisionHelperGeometry(fovH, fovV);

        // Position at world coordinates
        const px = wp[nodeIdx * 3 + 0];
        const py = wp[nodeIdx * 3 + 1];
        const pz = wp[nodeIdx * 3 + 2];
        this.visionHelper.position.set(px, py, pz);

        // Orient along heading vector
        // Target defaults to -Z axis if no heading exists
        let headingX = 0, headingY = 0, headingZ = -1;
        if (pt && pt.heading) {
            headingX = pt.heading.x;
            headingY = pt.heading.y;
            headingZ = pt.heading.z;
        }
        
        const headingLen = Math.sqrt(headingX*headingX + headingZ*headingZ) || 1;
        const target = new THREE.Vector3(
            px + headingX / headingLen,
            py, // Force Y=0 relative to origin
            pz + headingZ / headingLen
        );
        this.visionHelper.lookAt(target);

        this.visionHelper.visible = true;
    }

    /**
     * Generate or update the rectangular pyramid / frustum wireframe geometry
     */
    _updateVisionHelperGeometry(fovH, fovV) {
        const distance = 2.5; // Shrink to 2.5 meters
        const halfWidth = Math.tan(fovH / 2) * distance;
        const halfHeight = Math.tan(fovV / 2) * distance;
        const zEdge = distance; // POSITIVE Z ensures it points TOWARDS the lookAt target!

        // ── Wireframe Vertices ──
        const lineVertices = [
            // Center to far corners
            0, 0, 0,  -halfWidth,  halfHeight, zEdge,
            0, 0, 0,   halfWidth,  halfHeight, zEdge,
            0, 0, 0,   halfWidth, -halfHeight, zEdge,
            0, 0, 0,  -halfWidth, -halfHeight, zEdge,
            
            // Center Ray (Arrow inside the frustum)
            0, 0, 0,   0, 0, zEdge,
            
            // Far plane edges
            -halfWidth,  halfHeight, zEdge,   halfWidth,  halfHeight, zEdge,
             halfWidth,  halfHeight, zEdge,   halfWidth, -halfHeight, zEdge,
             halfWidth, -halfHeight, zEdge,  -halfWidth, -halfHeight, zEdge,
            -halfWidth, -halfHeight, zEdge,  -halfWidth,  halfHeight, zEdge
        ];

        // ── Solid Mesh Vertices & Faces ──
        const meshVertices = [
            0, 0, 0,                               // 0: Origin
            -halfWidth,  halfHeight, zEdge,        // 1: TL
             halfWidth,  halfHeight, zEdge,        // 2: TR
             halfWidth, -halfHeight, zEdge,        // 3: BR
            -halfWidth, -halfHeight, zEdge         // 4: BL
        ];
        
        const meshIndices = [
            0, 1, 2,   0, 2, 3,   0, 3, 4,   0, 4, 1,   // Side walls
            1, 3, 2,   1, 4, 3                          // Far plane (triangulated)
        ];

        if (!this.visionHelper) {
            this.visionHelper = new THREE.Group();
            this.visionHelper.name = 'SSX_VisionHelper';
            this.visionHelper.visible = false;
            
            // Create Wireframe
            const lineGeom = new THREE.BufferGeometry();
            lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));
            
            const lineMat = new THREE.LineBasicMaterial({
                color: 0x000000,
                transparent: true,
                opacity: 0.3,
                depthWrite: false, // Don't occlude other transparent objects
                linewidth: 1       // Will render as 1px thin lines
            });

            const lines = new THREE.LineSegments(lineGeom, lineMat);
            lines.name = 'Wireframe';
            lines.frustumCulled = false;
            this.visionHelper.add(lines);
            
            // Create Solid Volume
            const meshGeom = new THREE.BufferGeometry();
            meshGeom.setAttribute('position', new THREE.Float32BufferAttribute(meshVertices, 3));
            meshGeom.setIndex(meshIndices);
            
            const meshMat = new THREE.MeshBasicMaterial({
                color: 0x00aaff,
                transparent: true,
                opacity: 0.1,
                depthWrite: false,
                side: THREE.DoubleSide
            });
            
            const solidMesh = new THREE.Mesh(meshGeom, meshMat);
            solidMesh.name = 'Solid';
            solidMesh.frustumCulled = false;
            this.visionHelper.add(solidMesh);

            this.scene.add(this.visionHelper);
        } else {
            // Update Wireframe
            const lines = this.visionHelper.getObjectByName('Wireframe');
            if (lines) {
                lines.geometry.dispose();
                lines.geometry = new THREE.BufferGeometry();
                lines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(lineVertices, 3));
            }
            
            // Update Solid Volume
            const solidMesh = this.visionHelper.getObjectByName('Solid');
            if (solidMesh) {
                solidMesh.geometry.dispose();
                solidMesh.geometry = new THREE.BufferGeometry();
                solidMesh.geometry.setAttribute('position', new THREE.Float32BufferAttribute(meshVertices, 3));
                solidMesh.geometry.setIndex(meshIndices);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Explicit Global Registration
//  Ensures the class is available via AHIModules even if the bundle
//  footer's typeof check fails due to strict-mode scoping.
// ═══════════════════════════════════════════════════════════════════════
if (typeof window !== 'undefined') {
    window.AHIModules = window.AHIModules || {};
    window.AHIModules.SpaceSyntaxVisualizer = SpaceSyntaxVisualizer;
    console.log('[SpaceSyntaxVisualizer] Registered in window.AHIModules');
}
