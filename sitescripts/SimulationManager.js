/**
 * AHI 2.0 Ultimate - Simulation Manager
 *
 * "Conductor" of the physics orchestra.
 * Orchestrates Voxelizer -> LBM Solver pipeline.
 */
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { createVoxelizer } from './Voxelizer.js';
import { LBMSolver } from './LBMSolver.js';
import { CHTSolver } from './CHTSolver.js';
import { voxelIndex } from './VoxelTypes.js';
export class SimulationManager {
    static instance;
    voxelizer = null;
    lbmSolver = null;
    chtSolver = null;
    acousticSolver = null;
    neuroaestheticEvaluator = null;
    spectralTracer = null;
    voxelFractalAnalyzer = null;
    spaceSyntaxSolver = null;
    device = null;
    isRunning = false;
    animationFrameId = null;
    // Real-time coupling state
    simulationTime = 0; // seconds
    lastCHTUpdate = 0;
    chtUpdateInterval = 1000; // ms (1 sec = 1 game minute)
    // Visualization canvas
    canvas = null;
    ctx = null;
    visualizationMode = 'velocity';
    constructor() {
        // Voxelizer will be created asynchronously in initialize()
    }
    static getInstance() {
        if (!SimulationManager.instance) {
            SimulationManager.instance = new SimulationManager();
        }
        return SimulationManager.instance;
    }
    /**
     * Initialize the simulation pipeline
     * 1. Setup WebGPU
     * 2. Create GPU-accelerated Voxelizer
     * 3. Generate procedural geometry
     * 4. Voxelize on GPU
     * 5. Init LBM Solver
     */
    async initialize(device, canvas) {
        console.log('[SimulationManager] Initializing...');
        this.device = device;
        if (canvas) {
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
        }
        // 1. Create GPU-accelerated Voxelizer
        this.voxelizer = await createVoxelizer(0.1, device); // 10cm resolution
        console.log('[SimulationManager] GPU Voxelizer created');
        // 2. Generate procedural geometry (Room 5x5x3m with obstacle)
        this.generateTestGeometry();
        // 3. Voxelize using GPU
        const gridConfig = await this.voxelizer.voxelizeScene();
        const voxelBuffer = this.voxelizer.getVoxelBuffer();
        // Prepare initial temperature - use config or default 20°C for test geometry
        // In production, this should come from config.simulation.cht.initialTemperature
        const initialTempC = 20; // Test geometry default
        const initialTempK = initialTempC + 273.15;
        const initialTemp = new Float32Array(gridConfig.totalVoxels).fill(initialTempK);
        console.log(`[SimulationManager] Initial temperature: ${initialTempC}°C (${initialTempK.toFixed(2)}K)`);
        // 4. Init LBM Solver
        this.lbmSolver = new LBMSolver(this.device, gridConfig);
        await this.lbmSolver.initialize(voxelBuffer, initialTemp);
        // 5. Init CHT Solver
        this.chtSolver = new CHTSolver(this.device, gridConfig);
        await this.chtSolver.initialize(this.lbmSolver.getVoxelStateBuffer(), this.lbmSolver.getVelocityBuffer(), initialTemp);
        // 6. Set initial boundary conditions (20°C walls, 10°C outside)
        await this.chtSolver.updateBoundaryConditions({
            wall_temperature: 20,
            window_temperature: 15,
            air_temperature: 20,
            external_temperature: 10,
            timestamp: 'initial'
        });
        console.log('[SimulationManager] Initialization complete with CHT coupling.');
    }
    /**
     * Generate a simple test room with:
     * - Walls, Floor, Ceiling
     * - Window
     * - Central Obstacle (e.g. furniture)
     */
    generateTestGeometry() {
        // Use public API to add objects
        // Room 5x5x3
        // @ts-ignore - unused variable
        const roomGeo = new THREE.BoxGeometry(5, 3, 5);
        const material = new THREE.MeshBasicMaterial({ color: 0x808080 });
        const glassMat = new THREE.MeshBasicMaterial({ color: 0x88ccff });
        // Floor
        const floor = new THREE.Mesh(new THREE.BoxGeometry(5.2, 0.2, 5.2), material);
        floor.position.set(0, -0.1, 0);
        floor.name = "Concrete_Floor";
        this.voxelizer.addObject(floor);
        // Ceiling
        const ceiling = new THREE.Mesh(new THREE.BoxGeometry(5.2, 0.2, 5.2), material);
        ceiling.position.set(0, 3.1, 0);
        ceiling.name = "Concrete_Ceiling";
        this.voxelizer.addObject(ceiling);
        // Walls
        const wall1 = new THREE.Mesh(new THREE.BoxGeometry(0.2, 3, 5), material);
        wall1.position.set(-2.6, 1.5, 0);
        wall1.name = "Concrete_Wall_Left";
        this.voxelizer.addObject(wall1);
        const wall2 = new THREE.Mesh(new THREE.BoxGeometry(0.2, 3, 5), material);
        wall2.position.set(2.6, 1.5, 0);
        wall2.name = "Concrete_Wall_Right";
        this.voxelizer.addObject(wall2);
        const wall3 = new THREE.Mesh(new THREE.BoxGeometry(5, 3, 0.2), material);
        wall3.position.set(0, 1.5, -2.6);
        wall3.name = "Concrete_Wall_Back";
        this.voxelizer.addObject(wall3);
        // Front wall with window
        const wall4a = new THREE.Mesh(new THREE.BoxGeometry(1.5, 3, 0.2), material);
        wall4a.position.set(-1.75, 1.5, 2.6);
        wall4a.name = "Concrete_Wall_Front_1";
        this.voxelizer.addObject(wall4a);
        const wall4b = new THREE.Mesh(new THREE.BoxGeometry(1.5, 3, 0.2), material);
        wall4b.position.set(1.75, 1.5, 2.6);
        wall4b.name = "Concrete_Wall_Front_2";
        this.voxelizer.addObject(wall4b);
        const wall4c = new THREE.Mesh(new THREE.BoxGeometry(2, 0.5, 0.2), material);
        wall4c.position.set(0, 0.25, 2.6);
        wall4c.name = "Concrete_Wall_Front_Bottom";
        this.voxelizer.addObject(wall4c);
        const wall4d = new THREE.Mesh(new THREE.BoxGeometry(2, 0.5, 0.2), material);
        wall4d.position.set(0, 2.75, 2.6);
        wall4d.name = "Concrete_Wall_Front_Top";
        this.voxelizer.addObject(wall4d);
        // Window
        const window = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 0.1), glassMat);
        window.position.set(0, 1.5, 2.6);
        window.name = "Glass_Window";
        this.voxelizer.addObject(window);
        // Obstacle (Column/Furniture)
        const obstacle = new THREE.Mesh(new THREE.BoxGeometry(1, 2, 1), material);
        obstacle.position.set(0, 1, 0);
        obstacle.name = "Wood_Furniture"; // Use wood for variety
        this.voxelizer.addObject(obstacle);
    }
    /**
     * Start the simulation loop
     */
    start() {
        if (!this.lbmSolver || this.isRunning)
            return;
        this.isRunning = true;
        this.loop();
        console.log('[SimulationManager] Simulation started.');
    }
    /**
     * Stop the simulation
     */
    stop() {
        this.isRunning = false;
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        console.log('[SimulationManager] Simulation stopped.');
    }
    async loop() {
        if (!this.isRunning || !this.lbmSolver || !this.chtSolver)
            return;
        try {
            const dt = 0.001; // 1ms timestep
            // 1. Run CHT step (heat transfer)
            await this.chtSolver.step(dt);
            // 2. Run LBM step (fluid dynamics with buoyancy from CHT)
            await this.lbmSolver.step();
            // 3. Update simulation time
            this.simulationTime += dt;
            // 4. Check if we need to sync with backend (every game hour)
            const now = Date.now();
            if (now - this.lastCHTUpdate > this.chtUpdateInterval) {
                this.lastCHTUpdate = now;
                // In production, this would call the Python backend
                console.log(`[SimulationManager] Game hour passed. Sim time: ${(this.simulationTime / 60).toFixed(1)} min`);
            }
            // 5. Visualize (adaptive based on mode)
            if (this.canvas && this.ctx) {
                if (this.visualizationMode === 'velocity') {
                    const snapshot = await this.lbmSolver.getSnapshot();
                    this.renderSlice(snapshot);
                }
                else {
                    // Temperature visualization would go here
                    // const tempField = await this.chtSolver.getTemperatureField();
                    // this.renderTemperature(tempField);
                }
            }
            this.animationFrameId = requestAnimationFrame(() => this.loop());
        }
        catch (error) {
            console.error('Simulation error:', error);
            this.stop();
        }
    }
    /**
     * Switch visualization mode
     */
    toggleVisualizationMode() {
        this.visualizationMode = this.visualizationMode === 'velocity' ? 'temperature' : 'velocity';
        console.log(`[SimulationManager] Visualization mode: ${this.visualizationMode}`);
    }
    /**
     * Run single simulation step (public for optimizer)
     */
    async step() {
        if (!this.isRunning) {
            // Allow single step even when not running
            await this.loop();
        }
    }
    /**
     * Get current simulation snapshot (public for optimizer)
     */
    async getSnapshot() {
        if (this.lbmSolver) {
            return await this.lbmSolver.getSnapshot();
        }
        throw new Error('[SimulationManager] No LBM solver initialized');
    }
    /**
     * Get voxel buffer for analysis (public for optimizer)
     */
    getVoxelBuffer() {
        return this.voxelizer.getVoxelBuffer();
    }
    /**
     * Get grid configuration (public for optimizer)
     */
    getGridConfig() {
        return this.voxelizer.getGridConfig();
    }
    /**
     * Apply new geometry genome (for optimizer)
     */
    async applyGeometry(wallPositions, windowPositions) {
        // Clear existing geometry
        this.voxelizer.clearScene();
        // Add walls from genome
        for (let i = 0; i < wallPositions.length; i += 6) {
            const geometry = new THREE.BoxGeometry(wallPositions[i + 3], // width
                wallPositions[i + 4], // height
                wallPositions[i + 5] // thickness
            );
            const material = new THREE.MeshBasicMaterial();
            const wall = new THREE.Mesh(geometry, material);
            wall.position.set(wallPositions[i], wallPositions[i + 1], wallPositions[i + 2]);
            this.voxelizer.addObject(wall);
        }
        // Add windows (as transparent boxes)
        for (let i = 0; i < windowPositions.length; i += 4) {
            const geometry = new THREE.BoxGeometry(windowPositions[i + 2], // width
                windowPositions[i + 3], // height
                0.1 // thin glass
            );
            const material = new THREE.MeshBasicMaterial({ transparent: true });
            const window = new THREE.Mesh(geometry, material);
            window.position.set(windowPositions[i], windowPositions[i + 1], 1.5 // Default height
            );
            this.voxelizer.addObject(window);
        }
        // Re-voxelize
        const gridConfig = await this.voxelizer.voxelizeScene();
        const voxelBuffer = this.voxelizer.getVoxelBuffer();
        // Re-initialize solvers with new geometry
        if (this.lbmSolver && this.chtSolver) {
            // Re-initialize with same temperature as current - don't hardcode
            const currentConfig = this.chtSolver?.getConfig?.() || {};
            const initialTempC = currentConfig.initialTemperature || 20;
            const initialTempK = initialTempC + 273.15;
            const initialTemp = new Float32Array(gridConfig.totalVoxels).fill(initialTempK);
            await this.lbmSolver.initialize(voxelBuffer, initialTemp);
            await this.chtSolver.initialize(this.lbmSolver.getVoxelStateBuffer(), this.lbmSolver.getVelocityBuffer(), initialTemp);
        }
        console.log('[SimulationManager] Applied new geometry from optimizer');
    }
    /**
     * Simple visualization: 2D Slice of Velocity Magnitude at mid-height
     */
    renderSlice(snapshot) {
        if (!this.ctx || !this.canvas)
            return;
        const { velocityField, gridConfig } = snapshot;
        const { nx, ny, nz } = gridConfig.dimensions;
        // Pick middle Z slice
        const z = Math.floor(nz / 2);
        // Resize canvas if needed
        if (this.canvas.width !== nx || this.canvas.height !== ny) {
            this.canvas.width = nx;
            this.canvas.height = ny;
        }
        const imageData = this.ctx.createImageData(nx, ny);
        const data = imageData.data;
        let maxVel = snapshot.metrics.maxVelocity || 0.1; // Avoid div by zero
        for (let j = 0; j < ny; j++) {
            for (let i = 0; i < nx; i++) {
                const idx = voxelIndex(i, j, z, nx, ny);
                // Get velocity vector
                const vx = velocityField[idx * 3];
                const vy = velocityField[idx * 3 + 1];
                const vz = velocityField[idx * 3 + 2];
                const mag = Math.sqrt(vx * vx + vy * vy + vz * vz);
                const t = Math.min(1.0, mag / maxVel); // Normalized magnitude
                // Heatmap coloring (Blue -> Red)
                // Simple RGB lerp
                const r = Math.floor(t * 255);
                const g = Math.floor((1 - Math.abs(t - 0.5) * 2) * 255); // Peak at 0.5
                const b = Math.floor((1 - t) * 255);
                const pixelIdx = ((ny - 1 - j) * nx + i) * 4; // Flip Y for canvas coords
                data[pixelIdx] = r;
                data[pixelIdx + 1] = g;
                data[pixelIdx + 2] = b;
                data[pixelIdx + 3] = 255; // Alpha
            }
        }
        this.ctx.putImageData(imageData, 0, 0);
    }
    getSolver() {
        return this.lbmSolver;
    }
}
