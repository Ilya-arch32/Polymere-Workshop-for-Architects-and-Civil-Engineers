/**
 * AHI 2.0 Ultimate - Conjugate Heat Transfer Solver
 *
 * Manages thermal coupling between solid walls and fluid (air)
 * Integrates with LBM for buoyancy-driven flows
 * Includes solar radiation based on NREL SPA
 */
import { SolarPositionAlgorithm } from './SolarPositionAlgorithm.js';
// WGSL shader loaded via window.AHIModules.shaders
/**
 * Стандартные материалы с паропроницаемостью и MBV (NORDTEST Project)
 * MBV значения из: Rode et al. "Moisture Buffering of Building Materials" (2005)
 */
export const MATERIAL_PROPERTIES = {
    AIR: {
        id: 0, name: 'Air',
        thermalConductivity: 0.026, density: 1.225, specificHeat: 1005,
        vaporPermeability: 1,
        mbv: 0 // Воздух не буферизует
    },
    CONCRETE: {
        id: 1, name: 'Concrete',
        thermalConductivity: 1.4, density: 2400, specificHeat: 880,
        vaporPermeability: 100,
        mbv: 0.38, // Limited buffer capacity
        sorptionWm: 0.02, sorptionC: 10, sorptionK: 0.75,
        hysteresisRatio: 0.85
    },
    WOOD: {
        id: 2, name: 'Wood',
        thermalConductivity: 0.15, density: 600, specificHeat: 1700,
        vaporPermeability: 20,
        mbv: 1.35, // Good buffer capacity
        sorptionWm: 0.05, sorptionC: 15, sorptionK: 0.80,
        hysteresisRatio: 0.75
    },
    GLASS: {
        id: 3, name: 'Glass',
        thermalConductivity: 1.0, density: 2500, specificHeat: 840,
        vaporPermeability: 1e6, // практически непроницаем
        mbv: 0 // No buffer capacity
    },
    INSULATION: {
        id: 4, name: 'Insulation',
        thermalConductivity: 0.04, density: 30, specificHeat: 1030,
        vaporPermeability: 5,
        mbv: 0.15, // Negligible (mineral wool)
        sorptionWm: 0.001, sorptionC: 5, sorptionK: 0.6
    },
    BRICK: {
        id: 5, name: 'Brick',
        thermalConductivity: 0.8, density: 1800, specificHeat: 900,
        vaporPermeability: 15,
        mbv: 0.48, // Limited-Moderate
        sorptionWm: 0.015, sorptionC: 12, sorptionK: 0.78,
        hysteresisRatio: 0.80
    },
    GYPSUM_BOARD: {
        id: 6, name: 'Gypsum Board',
        thermalConductivity: 0.25, density: 850, specificHeat: 1000,
        vaporPermeability: 8,
        mbv: 0.63, // Moderate - часто используется для регулирования влажности
        sorptionWm: 0.025, sorptionC: 8, sorptionK: 0.72,
        hysteresisRatio: 0.82
    },
    CLAY_PLASTER: {
        id: 7, name: 'Clay Plaster',
        thermalConductivity: 0.7, density: 1700, specificHeat: 1000,
        vaporPermeability: 10,
        mbv: 2.10, // Excellent - лучший буфер влажности
        sorptionWm: 0.04, sorptionC: 20, sorptionK: 0.85,
        hysteresisRatio: 0.70
    },
    CELLULOSE_INSULATION: {
        id: 8, name: 'Cellulose Insulation',
        thermalConductivity: 0.04, density: 50, specificHeat: 1600,
        vaporPermeability: 2,
        mbv: 1.85, // Good-Excellent
        sorptionWm: 0.08, sorptionC: 12, sorptionK: 0.82,
        hysteresisRatio: 0.78
    }
};
export class CHTSolver {
    device;
    gridConfig;
    config;
    // GPU Resources
    uniformBuffer;
    temperatureBufferA; // Double buffering
    temperatureBufferB;
    heatFluxBuffer;
    // Humidity buffers (Double Buffering) - ISO 13788
    humidityBufferA; // Относительная влажность 0.0-1.0
    humidityBufferB;
    moldRiskBuffer; // Флаги риска плесени
    moldRiskCounterBuffer; // Счетчик шагов с высокой влажностью
    // Pipelines
    diffusionPipeline;
    convectionPipeline;
    boundaryPipeline;
    buoyancyPipeline;
    // Humidity pipelines
    vaporDiffusionPipeline;
    moldRiskPipeline;
    bindGroupA;
    bindGroupB;
    humidityBindGroupA;
    humidityBindGroupB;
    currentBuffer = 'A';
    // Backend communication
    lastBackendUpdate = 0;
    boundaryConditions = null;
    // Solar calculation
    solarCalculator = null;
    solarRadiationBuffer;
    constructor(device, gridConfig, config) {
        this.device = device;
        this.gridConfig = gridConfig;

        // Calculate h_conv based on simulation mode and wind velocity
        const simulationMode = config?.simulationMode || 'outdoor';
        // REQUIRE inletVelocity from config - no fallback (must come from EPW or UI)
        const inletVelocity = config?.inletVelocity;
        if (inletVelocity === undefined || inletVelocity === null) {
            console.warn('[CHTSolver] ⚠️ inletVelocity not provided! Using minimum value 0.5 m/s for h_conv.');
        }
        const windSpeed = inletVelocity ?? 0.5; // Minimum wind for h_conv calc if missing

        let h_conv;
        if (simulationMode === 'indoor') {
            // Indoor: natural convection in still air
            h_conv = 10.0;
            console.log('[CHTSolver] Indoor mode: h_conv =', h_conv, 'W/(m²·K)');
        } else {
            // Outdoor: wind-dependent forced convection
            if (windSpeed < 3) {
                h_conv = 15.0;
            } else if (windSpeed <= 5) {
                h_conv = 50.0;
            } else if (windSpeed < 10) {
                h_conv = 50.0;
            } else {
                h_conv = 100.0;
            }
            console.log('[CHTSolver] Outdoor mode: wind =', windSpeed, 'm/s → h_conv =', h_conv, 'W/(m²·K)');
        }

        // Location must be provided via config (from EPW file or user input)
        // NO HARDCODED DEFAULTS - if location not provided, solar calculations will error
        if (!config?.location) {
            console.warn('[CHTSolver] ⚠️ No location provided in config! Solar calculations will be inaccurate.');
            console.warn('[CHTSolver] Location should come from EPW file or Analysis Parameters.');
        }

        this.config = {
            h_conv: h_conv,
            T_ref: 293.15, // 20°C reference
            beta: 3.4e-3, // Air thermal expansion
            updateInterval: 60000, // Update every minute
            // Location from config - NO DEFAULTS
            location: config?.location || null,
            // ISO 13788 parameters
            D_v: 2.5e-5, // Diffusion coefficient of water vapor in air
            moldRiskThreshold: 0.8, // 80% RH threshold for mold risk
            moldRiskSteps: 100,
            // TRL 7: MBV model
            enableMBV: true,
            mbvTimescale: 28800, // 8 hours (NORDTEST standard)
            enableHysteresis: true,
            ...config
        };

        // Initialize solar calculator only if location is provided
        // Use quiet mode (verbose: false) to avoid log spam during simulation
        if (this.config.location && this.config.location.latitude !== undefined) {
            this.solarCalculator = new SolarPositionAlgorithm(this.config.location, { verbose: false });
            console.log('[CHTSolver] Solar calculator initialized for:',
                this.config.location.latitude.toFixed(2) + '°N,',
                this.config.location.longitude.toFixed(2) + '°E');
        } else {
            console.warn('[CHTSolver] Solar calculator NOT initialized - no location data');
            this.solarCalculator = null;
        }

        // Store simulation date - must be set via setSimulationDate() before running
        this.simulationDate = null;
        this.solarUpdated = false; // Track if solar has been set for this simulation
    }
    /**
     * Initialize CHT solver with WebGPU resources
     */
    async initialize(voxelStateBuffer, velocityBuffer, initialTemperature) {
        let { nx, ny, nz, totalVoxels } = this.gridConfig.dimensions;
        // Ensure totalVoxels is integer for WebGPU buffer sizes
        totalVoxels = Math.floor(totalVoxels);

        // Handle null/undefined buffers gracefully
        if (!voxelStateBuffer) {
            console.warn('[CHTSolver] No voxelStateBuffer provided, creating fallback');
            voxelStateBuffer = this.device.createBuffer({
                label: 'Fallback Voxel Buffer',
                size: Math.floor(totalVoxels * 4),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
        }
        if (!velocityBuffer) {
            console.warn('[CHTSolver] No velocityBuffer provided, creating fallback');
            velocityBuffer = this.device.createBuffer({
                label: 'Fallback Velocity Buffer',
                size: Math.floor(totalVoxels * 3 * 4),
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
            });
        }

        // Save buffers for bind group creation
        this._voxelStateBuffer = voxelStateBuffer;
        this._velocityBuffer = velocityBuffer;
        // offset 0:  grid_size.x (u32)
        // offset 4:  grid_size.y (u32)
        // offset 8:  grid_size.z (u32)
        // offset 12: PADDING (u32) - Align to 16 bytes for next fields
        // offset 16: resolution (f32)
        // offset 20: dt (f32)
        // offset 24: alpha_solid (f32)
        // offset 28: alpha_fluid (f32)
        // offset 32: h_conv (f32)
        // offset 36: PADDING (f32) - Align gravity (vec3) to 16-byte boundary (offset 48)? 
        // WAIT: Offset 32 is 16-byte aligned. h_conv is f32 (4 bytes). Next is 36. 
        // Gravity (vec3) needs 16-byte alignment. Next multiple of 16 is 48.
        // So we need padding from 36 to 48 (12 bytes).

        // Let's force layout:
        // 0-12: grid_size
        // 12-16: PAD
        // 16: resolution
        // 20: dt
        // 24: alpha_solid
        // 28: alpha_fluid
        // 32: h_conv
        // 36-48: PAD (12 bytes)
        // 48-60: gravity (vec3)
        // 60-64: PAD to 64
        // 64: beta
        // 68: T_ref
        // 72: D_v
        // 76: moldRiskThreshold
        // 80: moldRiskSteps
        // 84: solar_azimuth (vec2 packed?)
        // Total: 144 bytes (must match shader struct)
        const uniformSize = 144; // Shader requires at least 144 bytes
        const uniformBuffer = new ArrayBuffer(uniformSize);
        const view = new DataView(uniformBuffer);

        // grid_size (vec3<u32>) - offset 0-11
        view.setUint32(0, nx, true);
        view.setUint32(4, ny, true);
        view.setUint32(8, nz, true);
        // padding at 12 (4 bytes)

        // resolution (f32) - offset 16
        view.setFloat32(16, this.gridConfig.resolution, true);

        // dt (f32) - offset 20
        view.setFloat32(20, 0.001, true);

        // Physics params - offset 24, 28, 32
        view.setFloat32(24, 2.2e-5, true); // alpha_solid
        view.setFloat32(28, 2.2e-5, true); // alpha_fluid
        view.setFloat32(32, this.config.h_conv, true); // h_conv

        // PADDING 36-48 to align gravity to 48 (16-byte aligned)

        // gravity (vec3<f32>) - offset 48
        view.setFloat32(48, 0.0, true);
        view.setFloat32(52, 0.0, true);
        view.setFloat32(56, -9.81, true);

        // beta, T_ref, D_v, moldRiskThreshold - offset 64+
        view.setFloat32(64, this.config.beta, true);
        view.setFloat32(68, this.config.T_ref, true);
        view.setFloat32(72, this.config.D_v || 2.5e-5, true);
        view.setFloat32(76, this.config.moldRiskThreshold || 0.8, true);
        view.setUint32(80, this.config.moldRiskSteps || 100, true);

        this.uniformBuffer = this.device.createBuffer({
            label: 'CHT Uniform Buffer',
            size: uniformBuffer.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformBuffer);

        // DEBUG: Log uniform buffer layout  
        console.log('[CHTSolver] Uniform buffer created:', {
            totalBytes: uniformBuffer.byteLength,
            grid_size: { nx, ny, nz },
            resolution: this.gridConfig.resolution,
            h_conv: this.config.h_conv,
            T_ref: this.config.T_ref
        });
        // Create temperature buffers (double buffering)
        const tempSize = totalVoxels * 4; // float32
        this.temperatureBufferA = this.device.createBuffer({
            label: 'Temperature Buffer A',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.temperatureBufferB = this.device.createBuffer({
            label: 'Temperature Buffer B',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        // Initialize with room temperature
        this.device.queue.writeBuffer(this.temperatureBufferA, 0, initialTemperature);
        this.device.queue.writeBuffer(this.temperatureBufferB, 0, initialTemperature);
        // DIAGNOSTIC: Log initial temperature data
        console.log('[CHTSolver DIAG] Initial temperature buffer:', {
            bufferByteLength: initialTemperature.byteLength,
            sample0: initialTemperature[0],
            sample1: initialTemperature[1],
            sample100: initialTemperature[100],
            allSame: initialTemperature[0] === initialTemperature[100]
        });
        // Create heat flux buffer
        this.heatFluxBuffer = this.device.createBuffer({
            label: 'Heat Flux Buffer',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        // ============================================
        // Humidity buffers (ISO 13788 compliance)
        // ============================================
        // Initialize humidity with 50% RH
        const initialHumidity = new Float32Array(totalVoxels).fill(0.5);
        this.humidityBufferA = this.device.createBuffer({
            label: 'Humidity Buffer A',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.device.queue.writeBuffer(this.humidityBufferA, 0, initialHumidity);
        this.humidityBufferB = this.device.createBuffer({
            label: 'Humidity Buffer B',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.device.queue.writeBuffer(this.humidityBufferB, 0, initialHumidity);
        // Mold risk flags (MOLD_RISK = 0x100 in VoxelState)
        this.moldRiskBuffer = this.device.createBuffer({
            label: 'Mold Risk Buffer',
            size: Math.floor(totalVoxels * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        // Counter for consecutive high-humidity steps
        this.moldRiskCounterBuffer = this.device.createBuffer({
            label: 'Mold Risk Counter Buffer',
            size: Math.floor(totalVoxels * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        // Load shaders
        const shaderModule = this.device.createShaderModule({
            label: 'CHT Shader Module',
            code: await this.loadShaderCode()
        });
        // Create pipelines
        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'CHT Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
            ]
        });
        const pipelineLayout = this.device.createPipelineLayout({
            label: 'CHT Pipeline Layout',
            bindGroupLayouts: [bindGroupLayout]
        });
        // Create compute pipelines for each step
        this.diffusionPipeline = this.device.createComputePipeline({
            label: 'Diffusion Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'diffusion_step'
            }
        });
        this.convectionPipeline = this.device.createComputePipeline({
            label: 'Convection Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'convection_step'
            }
        });
        this.boundaryPipeline = this.device.createComputePipeline({
            label: 'Boundary Coupling Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'boundary_coupling'
            }
        });
        this.buoyancyPipeline = this.device.createComputePipeline({
            label: 'Buoyancy Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'compute_buoyancy'
            }
        });
        // Create bind group layout for humidity buffers (group 1)
        const humidityBindGroupLayout = this.device.createBindGroupLayout({
            label: 'CHT Humidity Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // humidity_in
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // humidity_out
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // mold_risk
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }            // mold_risk_counter
            ]
        });
        this.humidityBindGroupLayout = humidityBindGroupLayout; // Store for later use

        // Pipeline layout with BOTH group 0 and group 1
        const humidityPipelineLayout = this.device.createPipelineLayout({
            label: 'CHT Humidity Pipeline Layout',
            bindGroupLayouts: [bindGroupLayout, humidityBindGroupLayout]
        });

        // Humidity pipelines use the combined layout
        this.vaporDiffusionPipeline = this.device.createComputePipeline({
            label: 'Vapor Diffusion Pipeline',
            layout: humidityPipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'vapor_diffusion_step'
            }
        });
        this.moldRiskPipeline = this.device.createComputePipeline({
            label: 'Mold Risk Pipeline',
            layout: humidityPipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'calculate_mold_risk'
            }
        });
        console.log('[CHTSolver] Humidity/mold risk pipelines created');
        // Create bind groups for double buffering (group 0)
        this.bindGroupA = this.device.createBindGroup({
            label: 'CHT Bind Group A',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: voxelStateBuffer } },
                { binding: 2, resource: { buffer: this.temperatureBufferA } },
                { binding: 3, resource: { buffer: this.temperatureBufferB } },
                { binding: 4, resource: { buffer: velocityBuffer } },
                { binding: 5, resource: { buffer: this.heatFluxBuffer } }
            ]
        });
        this.bindGroupB = this.device.createBindGroup({
            label: 'CHT Bind Group B',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: voxelStateBuffer } },
                { binding: 2, resource: { buffer: this.temperatureBufferB } },
                { binding: 3, resource: { buffer: this.temperatureBufferA } },
                { binding: 4, resource: { buffer: velocityBuffer } },
                { binding: 5, resource: { buffer: this.heatFluxBuffer } }
            ]
        });
        // Humidity bind groups for group 1 (vapor diffusion and mold risk)
        this.humidityBindGroupA = this.device.createBindGroup({
            label: 'CHT Humidity Bind Group A',
            layout: humidityBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.humidityBufferA } },
                { binding: 1, resource: { buffer: this.humidityBufferB } },
                { binding: 2, resource: { buffer: this.moldRiskBuffer } },
                { binding: 3, resource: { buffer: this.moldRiskCounterBuffer } }
            ]
        });
        this.humidityBindGroupB = this.device.createBindGroup({
            label: 'CHT Humidity Bind Group B',
            layout: humidityBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.humidityBufferB } },
                { binding: 1, resource: { buffer: this.humidityBufferA } },
                { binding: 2, resource: { buffer: this.moldRiskBuffer } },
                { binding: 3, resource: { buffer: this.moldRiskCounterBuffer } }
            ]
        });
        console.log('[CHTSolver] Initialized with humidity/mold risk support');
    }

    /**
     * Set the simulation date for solar calculations
     * Must be called before running simulation to get correct solar position
     */
    setSimulationDate(date) {
        this.simulationDate = date;
        this.solarUpdated = false; // Reset so solar will update on next step
        console.log('[CHTSolver] Simulation date set to:', date?.toLocaleString() || 'null');
    }

    /**
     * Calculate and apply solar radiation based on simulation time and location
     * Uses stored simulation date, NOT current system time!
     * Only logs on first update to avoid console spam
     */
    async updateSolarRadiation() {
        if (!this.solarCalculator) return;

        // Skip if already updated (solar position doesn't change during short simulation)
        if (this.solarUpdated) return;

        // Use simulation date, fallback to current time only if not set
        const dateForCalc = this.simulationDate || new Date();
        if (!this.simulationDate) {
            console.warn('[CHTSolver] ⚠️ No simulation date set - using current time (may be incorrect!)');
        }

        const solarPosition = this.solarCalculator.calculate(dateForCalc);
        const irradianceResult = this.solarCalculator.calculateSolarIrradianceWithWeather
            ? this.solarCalculator.calculateSolarIrradianceWithWeather(solarPosition, dateForCalc, null)
            : { ghi: this.solarCalculator.calculateSolarIrradiance(solarPosition, dateForCalc), source: 'legacy' };

        const solarIrradiance = irradianceResult.ghi || 0;

        // Update solar radiation buffer with directional intensity
        const irradianceData = new Float32Array([solarIrradiance]);
        this.device.queue.writeBuffer(this.uniformBuffer, 84, irradianceData);

        // Solar Direction (vec3) at offset 96 (aligned to 16 bytes)
        // Azimuth is clockwise from North. Elevation is from horizon.
        // Convert to Cartesian:
        // Z is UP (elevation). Y is North. X is East.
        // x = cos(el) * sin(az)
        // y = cos(el) * cos(az)
        // z = sin(el)
        const azRad = this.deg2rad(solarPosition.azimuth);
        const elRad = this.deg2rad(solarPosition.elevation);

        const directionData = new Float32Array([
            Math.cos(elRad) * Math.sin(azRad), // X
            Math.cos(elRad) * Math.cos(azRad), // Y 
            Math.sin(elRad)                    // Z (Up)
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 96, directionData);

        // Mark as updated and log ONCE
        this.solarUpdated = true;
        console.log(`[CHT] Solar set for ${dateForCalc.toLocaleDateString()}: ` +
            `Az=${solarPosition.azimuth.toFixed(1)}°, El=${solarPosition.elevation.toFixed(1)}°, ` +
            `GHI=${solarIrradiance.toFixed(0)}W/m²`);
    }
    deg2rad(deg) {
        return deg * Math.PI / 180;
    }
    /**
     * Execute one CHT timestep
     */
    async step(dt) {
        // DIAGNOSTIC: Log step inputs (every 1000 steps to avoid spam)
        if (!this._stepCount) this._stepCount = 0;
        this._stepCount++;
        if (this._stepCount % 1000 === 1) {
            console.log('[CHTSolver DIAG] step() called:', {
                stepNumber: this._stepCount,
                dt: dt,
                currentBuffer: this.currentBuffer,
                T_ref_config: this.config.T_ref,
                boundaryConditions: this.boundaryConditions
            });
        }

        // Update dt in uniforms (offset 16 = index 4 * 4 bytes)
        // FIXED LAYOUT: grid_size (0-12), pad (12-16), resolution (16-20), dt (20-24)
        const dtData = new Float32Array([dt]);
        this.device.queue.writeBuffer(this.uniformBuffer, 20, dtData); // Offset 20

        // Enable Solar Update
        await this.updateSolarRadiation();

        const commandEncoder = this.device.createCommandEncoder({
            label: 'CHT Step Command Encoder'
        });
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const workgroupSize = 4;
        const dispatchX = Math.ceil(nx / workgroupSize);
        const dispatchY = Math.ceil(ny / workgroupSize);
        const dispatchZ = Math.ceil(nz / workgroupSize);
        const bindGroup = this.currentBuffer === 'A' ? this.bindGroupA : this.bindGroupB;
        // 1. Diffusion in solids
        const diffusionPass = commandEncoder.beginComputePass({ label: 'Diffusion Pass' });
        diffusionPass.setPipeline(this.diffusionPipeline);
        diffusionPass.setBindGroup(0, bindGroup);
        diffusionPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        diffusionPass.end();
        // 2. Convection in fluids
        const convectionPass = commandEncoder.beginComputePass({ label: 'Convection Pass' });
        convectionPass.setPipeline(this.convectionPipeline);
        convectionPass.setBindGroup(0, bindGroup);
        convectionPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        convectionPass.end();
        // 3. Boundary coupling (heat exchange at interfaces)
        const boundaryPass = commandEncoder.beginComputePass({ label: 'Boundary Pass' });
        boundaryPass.setPipeline(this.boundaryPipeline);
        boundaryPass.setBindGroup(0, bindGroup);
        boundaryPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        boundaryPass.end();
        // 4. Compute buoyancy forces for LBM
        const buoyancyPass = commandEncoder.beginComputePass({ label: 'Buoyancy Pass' });
        buoyancyPass.setPipeline(this.buoyancyPipeline);
        buoyancyPass.setBindGroup(0, bindGroup);
        buoyancyPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        buoyancyPass.end();
        // 5. Vapor diffusion (ISO 13788 moisture transport)
        const humidityBindGroup = this.currentBuffer === 'A' ? this.humidityBindGroupA : this.humidityBindGroupB;
        const vaporPass = commandEncoder.beginComputePass({ label: 'Vapor Diffusion Pass' });
        vaporPass.setPipeline(this.vaporDiffusionPipeline);
        vaporPass.setBindGroup(0, bindGroup);
        vaporPass.setBindGroup(1, humidityBindGroup);
        vaporPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        vaporPass.end();

        // 6. Mold risk assessment (ISO 13788 surface condensation)
        const moldPass = commandEncoder.beginComputePass({ label: 'Mold Risk Pass' });
        moldPass.setPipeline(this.moldRiskPipeline);
        moldPass.setBindGroup(0, bindGroup);
        moldPass.setBindGroup(1, humidityBindGroup);
        moldPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        moldPass.end();
        this.device.queue.submit([commandEncoder.finish()]);
        // Swap buffers
        const prevBuffer = this.currentBuffer;
        this.currentBuffer = this.currentBuffer === 'A' ? 'B' : 'A';

        // DIAGNOSTIC: Log buffer swap (every 1000 steps)
        if (this._stepCount % 1000 === 1) {
            console.log('[CHTSolver DIAG] Buffer swapped:', prevBuffer, '->', this.currentBuffer);
        }

        // Check if we need to sync with backend
        await this.checkBackendSync();
    }
    /**
     * Update boundary conditions from RC-Network backend
     */
    async updateBoundaryConditions(conditions) {
        this.boundaryConditions = conditions;
        // Convert Celsius to Kelvin and apply to wall voxels
        const wallTempK = (conditions.wall_temperature ?? 20) + 273.15;
        const windowTempK = (conditions.window_temperature ?? conditions.wall_temperature ?? 20) + 273.15;
        const externalTempK = (conditions.external_temperature ?? 20) + 273.15;

        // Update T_ref in uniform buffer (offset 68 per WGSL struct: T_ref at offset 68, NOT 48!)
        // WGSL Layout: gravity at offset 48-60, _pad3 at 60-64, beta at 64, T_ref at 68
        if (this.uniformBuffer) {
            const tRefData = new Float32Array([externalTempK]);
            this.device.queue.writeBuffer(this.uniformBuffer, 68, tRefData);

            // Update moldRiskThreshold from EPW RH (offset 76 per WGSL struct)
            if (conditions.mold_risk_threshold !== undefined) {
                const moldThreshold = new Float32Array([conditions.mold_risk_threshold]);
                this.device.queue.writeBuffer(this.uniformBuffer, 76, moldThreshold);
                console.log(`[CHTSolver] Updated boundary conditions: Wall=${conditions.wall_temperature}°C, External(T_ref)=${conditions.external_temperature}°C (${externalTempK.toFixed(2)}K), MoldThreshold=${(conditions.mold_risk_threshold * 100).toFixed(0)}% (EPW RH)`);
            } else {
                console.log(`[CHTSolver] Updated boundary conditions: Wall=${conditions.wall_temperature}°C, External(T_ref)=${conditions.external_temperature}°C (${externalTempK.toFixed(2)}K at offset 68)`);
            }
        } else {
            console.log(`[CHTSolver] Updated boundary conditions: Wall=${conditions.wall_temperature}°C`);
        }
        this.lastBackendUpdate = Date.now();
    }
    /**
     * Check if we need to sync with Python backend
     */
    async checkBackendSync() {
        // Backend sync disabled by default to prevent boundary conditions from changing mid-simulation
        // User requested static parameters during the run.
        return;

        /* 
        // Disabled logic
        if (this.backendDisabled) return;

        const now = Date.now();
        if (now - this.lastBackendUpdate > this.config.updateInterval) {
            // Request update from backend
            await this.requestBackendUpdate();
        }
        */
    }
    /**
     * Request temperature update from RC-Network backend
     */
    async requestBackendUpdate() {
        if (this.backendDisabled) return;

        try {
            // Get current average air temperature
            const avgTemp = await this.getAverageAirTemperature();
            // Call Python backend - OPTIONAL, will fail gracefully if not available
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 2000); // 2s timeout

            const response = await fetch('/analyze/thermal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: controller.signal,
                body: JSON.stringify({
                    room: {
                        wall_area: 60, // Simplified: assume average room
                        window_area: 6,
                        floor_area: 25,
                        volume: 75,
                        wall_thickness: 0.3,
                        u_value_wall: 0.3,
                        u_value_window: 1.2,
                        air_change_rate: 0.5
                    },
                    // Use current condition or config, fallback to defaults only if missing
                    external_temperature: this.boundaryConditions?.external_temperature ?? (this.config.T_ref - 273.15),
                    solar_radiation: this.solarCalculator?.lastIrradiance ?? 0
                })
            });
            clearTimeout(timeoutId);

            if (response.ok) {
                const data = await response.json();
                await this.updateBoundaryConditions(data.boundary_conditions);
            }
        }
        catch (error) {
            // Disable backend after first failure to prevent spamming
            this.backendDisabled = true;
            console.log('[CHTSolver] Backend sync disabled (Python server not running on port 8000)');
        }
    }
    /**
     * Get average air temperature for feedback to RC-Network
     */
    async getAverageAirTemperature() {
        // Create staging buffer for readback
        const stagingBuffer = this.device.createBuffer({
            size: this.temperatureBufferA.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const commandEncoder = this.device.createCommandEncoder();
        const sourceBuffer = this.currentBuffer === 'A' ? this.temperatureBufferA : this.temperatureBufferB;
        commandEncoder.copyBufferToBuffer(sourceBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const tempData = new Float32Array(stagingBuffer.getMappedRange());
        // Calculate average (simplified - should only average FLUID voxels)
        let sum = 0;
        let count = 0;
        for (let i = 0; i < tempData.length; i++) {
            if (tempData[i] > 200 && tempData[i] < 400) { // Sanity check (200K to 400K)
                sum += tempData[i];
                count++;
            }
        }
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        return count > 0 ? (sum / count) - 273.15 : 20; // Return in Celsius
    }
    /**
     * Get heat flux buffer for LBM buoyancy calculations
     */
    getHeatFluxBuffer() {
        return this.heatFluxBuffer;
    }
    /**
     * Get current temperature buffer
     */
    getTemperatureBuffer() {
        return this.currentBuffer === 'A' ? this.temperatureBufferA : this.temperatureBufferB;
    }
    /**
     * Load shader code from heat_transfer.wgsl
     */
    async loadShaderCode() {
        const shaderCode = window.AHIModules?.shaders?.['heat_transfer.wgsl'] || '';
        if (!shaderCode) throw new Error('[CHTSolver] WGSL shader not loaded');

        // DIAGNOSTIC: Check which version of shader is loaded
        const hasFluidOnlyDiffusion = shaderCode.includes('FLUID-to-FLUID');
        const hasOldDiffusion = shaderCode.includes('Check all 6 neighbors for diffusion') && !hasFluidOnlyDiffusion;
        console.log('[CHTSolver SHADER DIAG] Loaded shader length:', shaderCode.length, 'chars');
        console.log('[CHTSolver SHADER DIAG] Has FLUID-only diffusion fix?', hasFluidOnlyDiffusion);
        if (hasOldDiffusion) {
            console.warn('[CHTSolver SHADER DIAG] ⚠️ OLD SHADER DETECTED - double heat transfer bug present!');
        }

        return shaderCode;
    }
    /**
     * Get humidity buffer for external access
     */
    getHumidityBuffer() {
        return this.currentBuffer === 'A' ? this.humidityBufferA : this.humidityBufferB;
    }
    /**
     * Get mold risk buffer for visualization
     */
    getMoldRiskBuffer() {
        return this.moldRiskBuffer;
    }
    /**
     * TRL 7: Calculate moisture buffer effect on air humidity
     * Implements NORDTEST MBV model for transient moisture response
     *
     * Δm = MBV × A × ΔRH / t_cycle
     * where:
     *   Δm = moisture flux [g/s]
     *   MBV = Moisture Buffer Value [g/(m²·%RH)]
     *   A = surface area [m²]
     *   ΔRH = change in relative humidity [%]
     *   t_cycle = cycle time [s]
     */
    async calculateMoistureBufferEffect(surfaceAreas, // materialId -> surface area [m²]
        currentRH, // Current air RH [0-1]
        targetRH, // Target/equilibrium RH [0-1]
        dt // Time step [s]
    ) {
        if (!this.config.enableMBV) {
            return {
                bufferedRH: currentRH,
                moistureFlux: 0,
                bufferCapacity: 0,
                materialContributions: new Map()
            };
        }
        const deltaRH = (targetRH - currentRH) * 100; // Convert to %
        let totalMoistureFlux = 0; // g/s
        const contributions = new Map();
        // Calculate contribution from each material
        for (const [materialId, area] of surfaceAreas) {
            const material = Object.values(MATERIAL_PROPERTIES).find(m => m.id === materialId);
            if (!material || material.mbv === 0)
                continue;
            // MBV model: Δm = MBV × A × ΔRH / t_cycle
            // Time constant τ = t_cycle / 2π for exponential response
            const timeConstant = this.config.mbvTimescale / (2 * Math.PI);
            const responseRate = 1 - Math.exp(-dt / timeConstant);
            // Moisture flux from this material
            let flux = material.mbv * area * deltaRH * responseRate;
            // Apply hysteresis if enabled (asymmetric sorption/desorption)
            if (this.config.enableHysteresis && material.hysteresisRatio) {
                // Desorption is slower than sorption
                if (deltaRH < 0) { // Desorption (RH decreasing)
                    flux *= material.hysteresisRatio;
                }
            }
            totalMoistureFlux += flux;
            contributions.set(material.name, flux);
        }
        // Convert moisture flux to RH change in air volume
        // Assume standard room: V = 75 m³, T = 20°C
        // At 20°C, saturation: 17.3 g/m³
        const roomVolume = 75; // m³ (TODO: get from actual room)
        const saturationDensity = 17.3; // g/m³ at 20°C
        const rhChange = totalMoistureFlux * dt / (roomVolume * saturationDensity);
        const bufferedRH = Math.max(0, Math.min(1, currentRH + rhChange));
        // Calculate remaining buffer capacity
        // Simplified: assume 50% saturation at current state
        let totalCapacity = 0;
        for (const [materialId, area] of surfaceAreas) {
            const material = Object.values(MATERIAL_PROPERTIES).find(m => m.id === materialId);
            if (!material || material.mbv === 0)
                continue;
            // Maximum buffer at full RH swing (0-100%)
            totalCapacity += material.mbv * area * 50; // 50% available capacity
        }
        console.log(`[CHTSolver] MBV: ΔRH=${(rhChange * 100).toFixed(2)}%, flux=${totalMoistureFlux.toFixed(2)}g/s`);
        return {
            bufferedRH,
            moistureFlux: totalMoistureFlux,
            bufferCapacity: totalCapacity,
            materialContributions: contributions
        };
    }
    /**
     * TRL 7: Calculate equilibrium moisture content using GAB sorption isotherm
     * w = (w_m × C × K × RH) / ((1 - K×RH) × (1 - K×RH + C×K×RH))
     */
    calculateSorptionIsotherm(material, rh) {
        if (!material.sorptionWm || !material.sorptionC || !material.sorptionK) {
            // Fallback: linear approximation
            return rh * 0.05; // ~5% moisture content at 100% RH
        }
        const { sorptionWm: wm, sorptionC: C, sorptionK: K } = material;
        const phi = Math.max(0.01, Math.min(0.99, rh)); // Clamp to avoid singularities
        // GAB equation
        const numerator = wm * C * K * phi;
        const denominator = (1 - K * phi) * (1 - K * phi + C * K * phi);
        return numerator / denominator; // kg water / kg dry material
    }
    /**
     * TRL 7: Get comprehensive humidity analysis
     */
    async getHumidityAnalysis() {
        // Get humidity data
        const stagingBuffer = this.device.createBuffer({
            size: this.humidityBufferA.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const commandEncoder = this.device.createCommandEncoder();
        const sourceBuffer = this.currentBuffer === 'A' ? this.humidityBufferA : this.humidityBufferB;
        commandEncoder.copyBufferToBuffer(sourceBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const humidityData = new Float32Array(stagingBuffer.getMappedRange());
        let sum = 0, min = 1, max = 0, count = 0;
        for (let i = 0; i < humidityData.length; i++) {
            const rh = humidityData[i];
            if (rh >= 0 && rh <= 1) {
                sum += rh;
                min = Math.min(min, rh);
                max = Math.max(max, rh);
                count++;
            }
        }
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        const averageRH = count > 0 ? sum / count : 0.5;
        // Get mold risk data
        const moldRisk = await this.getMoldRiskData();
        // Calculate buffer efficiency (how stable is RH)
        const rhVariance = max - min;
        const bufferEfficiency = Math.max(0, 1 - rhVariance * 2); // Lower variance = better buffering
        // Determine comfort level
        let comfortLevel;
        if (averageRH < 0.3) {
            comfortLevel = 'dry';
        }
        else if (averageRH < 0.6) {
            comfortLevel = 'optimal';
        }
        else if (averageRH < 0.8) {
            comfortLevel = 'humid';
        }
        else {
            comfortLevel = 'critical';
        }
        return {
            averageRH,
            minRH: min,
            maxRH: max,
            moldRisk,
            bufferEfficiency,
            comfortLevel
        };
    }
    /**
     * Get mold risk data for analysis
     * Returns Uint32Array where each element is mold risk flag for that voxel
     * 0 = no risk, >0 = risk detected
     */
    async getMoldRiskData() {
        if (!this.moldRiskBuffer) {
            console.warn('[CHTSolver] Mold risk buffer not initialized');
            return null;
        }

        const stagingBuffer = this.device.createBuffer({
            size: this.moldRiskBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.moldRiskBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);

        // Copy data before unmap
        const rawData = new Uint32Array(stagingBuffer.getMappedRange());
        const data = new Uint32Array(rawData.length);
        data.set(rawData);

        stagingBuffer.unmap();
        stagingBuffer.destroy();

        // Log diagnostic
        let affectedVoxels = 0;
        for (let i = 0; i < data.length; i++) {
            if (data[i] > 0) affectedVoxels++;
        }
        console.log('[CHTSolver] Mold risk data:', affectedVoxels, '/', data.length, 'voxels affected');

        return data;  // Return raw array for preview.html to filter by interior voxels
    }

    /**
     * Get snapshot of current temperature field
     */
    async getSnapshot() {
        if (!this.temperatureBufferA || !this.device) {
            console.warn('[CHTSolver] Cannot get snapshot - no temperature buffer');
            return null;
        }

        const { totalVoxels } = this.gridConfig.dimensions || this.gridConfig;
        const stagingBuffer = this.device.createBuffer({
            label: 'CHT Temperature Staging',
            size: Math.floor(totalVoxels * 4),
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        // CRITICAL FIX: Use correct buffer based on currentBuffer state
        const sourceBuffer = this.currentBuffer === 'A' ? this.temperatureBufferA : this.temperatureBufferB;

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            sourceBuffer, 0,
            stagingBuffer, 0,
            Math.floor(totalVoxels * 4)
        );
        this.device.queue.submit([commandEncoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));

        // EXPOSE FOR VISUALIZATION
        window.AHI_Results = window.AHI_Results || {};
        window.AHI_Results.thermal = {
            temperatureField: data,
            gridConfig: this.gridConfig
        };
        console.log('[CHTSolver] Cached temperatureField in window.AHI_Results.thermal');

        stagingBuffer.unmap();
        stagingBuffer.destroy();

        // Calculate min/max without spread operator (stack overflow on 1M elements)
        let minT = Infinity, maxT = -Infinity;
        let count0 = 0, count293 = 0, countOther = 0;
        for (let i = 0; i < Math.min(data.length, 10000); i++) {  // Sample first 10k for speed
            if (data[i] < minT) minT = data[i];
            if (data[i] > maxT) maxT = data[i];
            // Count temperature buckets
            if (data[i] === 0) count0++;
            else if (Math.abs(data[i] - 293.15) < 1) count293++;
            else countOther++;
        }
        console.log('[CHTSolver] Temperature snapshot: min=', minT.toFixed(1), 'max=', maxT.toFixed(1));
        console.log('[CHTSolver] Temperature distribution (first 10k): zeros=', count0, 'near293K=', count293, 'other=', countOther);
        console.log('[CHTSolver] Sample temperatures [0-4]:', data[0]?.toFixed(1), data[1]?.toFixed(1), data[2]?.toFixed(1), data[3]?.toFixed(1), data[4]?.toFixed(1));

        // DIAGNOSTIC: Check voxel states for first 5 indices
        // Read voxelStateBuffer to see what state these voxels have
        const { nx, ny, nz } = this.gridConfig.dimensions;
        console.log('[CHTSolver DIAG] Grid dimensions:', nx, 'x', ny, 'x', nz);
        console.log('[CHTSolver DIAG] First 5 voxels are at coords:');
        for (let i = 0; i < 5; i++) {
            const x = i % nx;
            const y = Math.floor(i / nx) % ny;
            const z = Math.floor(i / (nx * ny));
            console.log(`  [${i}]: (${x},${y},${z}) -> Temp=${data[i]?.toFixed(1)}K = ${(data[i] - 273.15)?.toFixed(1)}°C`);
        }

        // DIAGNOSTIC: Check if temperature changed from initial 293.15K
        const initialTemp = 293.15;
        let changedCount = 0;
        for (let i = 0; i < Math.min(data.length, 1000); i++) {
            if (Math.abs(data[i] - initialTemp) > 0.01) changedCount++;
        }
        // DIAGNOSTIC: Analyze temperatures by VOXEL TYPE (SOLID vs FLUID)
        // This requires reading voxelStateBuffer alongside temperature
        if (this._voxelStateBuffer) {
            try {
                // Read FULL voxel buffer to find SOLID walls (they're not in first 10k)
                const voxelStagingBuffer = this.device.createBuffer({
                    label: 'Voxel State Staging',
                    size: this._voxelStateBuffer.size,
                    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
                });
                const voxelEncoder = this.device.createCommandEncoder();
                voxelEncoder.copyBufferToBuffer(this._voxelStateBuffer, 0, voxelStagingBuffer, 0, voxelStagingBuffer.size);
                this.device.queue.submit([voxelEncoder.finish()]);
                await voxelStagingBuffer.mapAsync(GPUMapMode.READ);
                const voxelData = new Float32Array(voxelStagingBuffer.getMappedRange());

                const VOXEL_STRIDE = 8;
                let solidCount = 0, fluidCount = 0;
                let solidTempSum = 0, fluidTempSum = 0;
                let solidMinT = Infinity, solidMaxT = -Infinity;
                let fluidMinT = Infinity, fluidMaxT = -Infinity;
                let interiorFluidCount = 0, exteriorFluidCount = 0;

                const totalVoxels = Math.floor(voxelData.length / VOXEL_STRIDE);
                const step = Math.max(1, Math.floor(totalVoxels / 20000)); // Sample ~20k evenly distributed

                for (let i = 0; i < totalVoxels; i += step) {
                    const state = voxelData[i * VOXEL_STRIDE];
                    const tempK = data[i]; // From temperature buffer

                    if (state > 0.5 && state < 1.5) { // SOLID = 1
                        solidCount++;
                        solidTempSum += tempK;
                        if (tempK < solidMinT) solidMinT = tempK;
                        if (tempK > solidMaxT) solidMaxT = tempK;
                    } else if (state > 1.5) { // FLUID = 2
                        fluidCount++;
                        fluidTempSum += tempK;
                        if (tempK < fluidMinT) fluidMinT = tempK;
                        if (tempK > fluidMaxT) fluidMaxT = tempK;
                        // Count interior vs exterior based on temperature
                        if (Math.abs(tempK - 293.15) < 1) {
                            interiorFluidCount++; // Still at ~20°C = interior
                        } else {
                            exteriorFluidCount++; // Changed = exterior
                        }
                    }
                }

                voxelStagingBuffer.unmap();
                voxelStagingBuffer.destroy();

                console.log('[CHTSolver DIAG] Temperature by voxel type (sample 10k):');
                console.log('[CHTSolver DIAG]   SOLID walls:', solidCount, 'voxels, avg=',
                    solidCount > 0 ? ((solidTempSum / solidCount) - 273.15).toFixed(2) + '°C' : 'N/A',
                    'range:', solidMinT < Infinity ? (solidMinT - 273.15).toFixed(1) + ' to ' + (solidMaxT - 273.15).toFixed(1) + '°C' : 'N/A');
                console.log('[CHTSolver DIAG]   FLUID air:', fluidCount, 'voxels, avg=',
                    fluidCount > 0 ? ((fluidTempSum / fluidCount) - 273.15).toFixed(2) + '°C' : 'N/A',
                    'range:', fluidMinT < Infinity ? (fluidMinT - 273.15).toFixed(1) + ' to ' + (fluidMaxT - 273.15).toFixed(1) + '°C' : 'N/A');

                // Check if SOLID walls cooled at all from initial 20°C
                const wallCooled = solidMaxT < 292.15; // More than 1 degree below initial
                const airCooled = fluidMaxT < 292.15;
                console.log('[CHTSolver DIAG]   Walls cooled from 20°C?', wallCooled ? 'YES' : 'NO ⚠️');
                console.log('[CHTSolver DIAG]   Air cooled from 20°C?', airCooled ? 'YES' : 'NO ⚠️');

            } catch (diagErr) {
                console.warn('[CHTSolver DIAG] Voxel type analysis failed:', diagErr.message);
            }
        }

        console.log('[CHTSolver DIAG] getSnapshot() result:', {
            totalSteps: this._stepCount || 0,
            sourceBuffer: this.currentBuffer,
            changedFromInitial: changedCount + '/1000 voxels',
            minTemp: minT.toFixed(2) + 'K (' + (minT - 273.15).toFixed(2) + '°C)',
            maxTemp: maxT.toFixed(2) + 'K (' + (maxT - 273.15).toFixed(2) + '°C)'
        });

        return {
            temperatureField: data
        };
    }

    /**
     * Cleanup resources
     */
    destroy() {
        this.uniformBuffer?.destroy();
        this.temperatureBufferA?.destroy();
        this.temperatureBufferB?.destroy();
        this.heatFluxBuffer?.destroy();
        this.humidityBufferA?.destroy();
        this.humidityBufferB?.destroy();
        this.moldRiskBuffer?.destroy();
        this.moldRiskCounterBuffer?.destroy();
        console.log('[CHTSolver] Destroyed');
    }
}
