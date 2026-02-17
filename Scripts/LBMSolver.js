/**
 * AHI 2.0 Ultimate - LBM Solver TypeScript Wrapper
 *
 * Управляет WebGPU compute pipeline для LBM D3Q19 симуляции
 */
import { PHYSICS_CONSTANTS } from './VoxelTypes.js';
// WGSL shader loaded via window.AHIModules.shaders
export class LBMSolver {
    device;
    gridConfig;
    config;
    // GPU Buffers
    uniformBuffer;
    fInBuffer;
    fOutBuffer;
    densityBuffer;
    velocityBuffer;
    temperatureBuffer;
    voxelStateBuffer;
    // Compute pipelines
    collisionPipeline;
    streamingPipeline;
    inletBCPipeline;
    outletBCPipeline;
    bindGroup;
    currentStep = 0;
    initialized = false;
    // TRL 7: Adaptive time stepping state
    currentDt = 0.001;
    lastMaxVelocity = 0;
    lastMachNumber = 0;
    dtHistory = [];
    // LBM lattice constants
    CS = 1.0 / Math.sqrt(3); // Lattice speed of sound
    constructor(device, gridConfig, config) {
        this.device = device;
        this.gridConfig = gridConfig;
        // Default configuration
        this.config = {
            tau: 0.6,
            nu: 1.5e-5, // Air at 20°C
            rho0: PHYSICS_CONSTANTS.AIR_DENSITY,
            gravity: [0, 0, -9.81],
            dt: 0.001, // 1ms
            enableBuoyancy: true,
            beta: 3.4e-3, // Air thermal expansion
            smagorinskyConstant: 0.15, // Стандартное значение для зданий (0.1-0.2)
            enableLES: true, // Включаем LES по умолчанию для высоких Re
            // TRL 7: Adaptive time stepping defaults
            enableAdaptiveDt: true,
            maxMach: 0.1, // Critical for incompressible LBM validity
            cflFactor: 0.7, // Conservative CFL factor
            dtMin: 1e-6, // 1 microsecond minimum
            dtMax: 0.01, // 10 millisecond maximum
            dtUpdateInterval: 10, // Update dt every 10 steps
            // UI parameters - MUST be provided by config, no defaults
            // inletVelocity: required from EPW or UI
            // terrainRoughness: required from UI
            ...config,
        };
        this.currentDt = this.config.dt;
        console.log('[LBMSolver] Initialized with config:', this.config);
    }
    /**
     * Инициализация WebGPU ресурсов
     */
    async initialize(voxelStateData, initialTemperature) {
        console.time('[LBMSolver] Initialization');
        const { nx, ny, nz, totalVoxels } = this.gridConfig.dimensions;
        const D3Q19_DIRECTIONS = 19;
        // Создаем буферы
        this.createBuffers(totalVoxels, D3Q19_DIRECTIONS, voxelStateData, initialTemperature);
        // Компилируем шейдеры (loaded via AHI loader)
        const lbmShaderCode = window.AHIModules?.shaders?.['lbm_solver.wgsl'] || '';
        if (!lbmShaderCode) throw new Error('[LBMSolver] WGSL shader not loaded');
        const shaderModule = this.device.createShaderModule({
            label: 'LBM D3Q19 Shader',
            code: lbmShaderCode,
        });
        // Создаем bind group layout
        const bindGroupLayout = this.device.createBindGroupLayout({
            label: 'LBM Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ],
        });
        // Создаем pipeline layout
        const pipelineLayout = this.device.createPipelineLayout({
            label: 'LBM Pipeline Layout',
            bindGroupLayouts: [bindGroupLayout],
        });
        // Создаем compute pipelines
        this.collisionPipeline = this.device.createComputePipeline({
            label: 'LBM Collision Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'collisionStep',
            },
        });
        this.streamingPipeline = this.device.createComputePipeline({
            label: 'LBM Streaming Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'streamingStep',
            },
        });
        this.inletBCPipeline = this.device.createComputePipeline({
            label: 'LBM Inlet BC Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'applyInletBC',
            },
        });
        this.outletBCPipeline = this.device.createComputePipeline({
            label: 'LBM Outlet BC Pipeline',
            layout: pipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'applyOutletBC',
            },
        });
        // Создаем bind group
        this.bindGroup = this.device.createBindGroup({
            label: 'LBM Bind Group',
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.fInBuffer } },
                { binding: 2, resource: { buffer: this.fOutBuffer } },
                { binding: 3, resource: { buffer: this.densityBuffer } },
                { binding: 4, resource: { buffer: this.velocityBuffer } },
                { binding: 5, resource: { buffer: this.temperatureBuffer } },
                { binding: 6, resource: { buffer: this.voxelStateBuffer } },
            ],
        });
        this.initialized = true;
        console.timeEnd('[LBMSolver] Initialization');
        console.log(`[LBMSolver] Ready for simulation (${totalVoxels} voxels, ${D3Q19_DIRECTIONS} velocities)`);
    }
    /**
     * Создание GPU буферов
     */
    createBuffers(totalVoxels, directions, voxelStateData, initialTemperature) {
        // Ensure totalVoxels is integer for WebGPU buffer sizes
        totalVoxels = Math.floor(totalVoxels);

        const { nx, ny, nz } = this.gridConfig.dimensions;
        // Uniform buffer (параметры симуляции)
        // CRITICAL: Must use DataView for mixed u32/f32 types!
        // WGSL SimulationParams struct layout (with explicit vec3 padding!):
        // offset 0-15:  nx, ny, nz, resolution
        // offset 16-31: tau, omega, rho0, nu
        // offset 32-47: gravity (vec3, 12 bytes) + _padding_gravity (4 bytes) = CRITICAL for alignment!
        // offset 48-63: dt, enableBuoyancy, beta, smagorinskyConstant
        // offset 64-79: enableLES, inletVelocity, terrainRoughness, windDirX
        // offset 80-95: windDirY, windDirZ, inletPlane, outletPlane
        // offset 96-111: padding to satisfy WGSL 16-byte struct alignment
        // Total: 112 bytes (must be multiple of 16 for uniform buffers!)

        const uniformBuffer = new ArrayBuffer(112);
        const view = new DataView(uniformBuffer);
        let offset = 0;

        // Row 1: nx, ny, nz (u32), resolution (f32)
        view.setUint32(offset, nx, true); offset += 4;
        view.setUint32(offset, ny, true); offset += 4;
        view.setUint32(offset, nz, true); offset += 4;
        view.setFloat32(offset, this.gridConfig.resolution, true); offset += 4;

        // Row 2: tau, omega, rho0, nu (all f32)
        view.setFloat32(offset, this.config.tau, true); offset += 4;
        view.setFloat32(offset, 1.0 / this.config.tau, true); offset += 4; // omega
        view.setFloat32(offset, this.config.rho0, true); offset += 4;
        view.setFloat32(offset, this.config.nu, true); offset += 4;

        // Row 3: gravity vec3 + padding (WGSL vec3 has 16-byte alignment in uniform buffers!)
        view.setFloat32(offset, this.config.gravity[0], true); offset += 4;  // offset 32
        view.setFloat32(offset, this.config.gravity[1], true); offset += 4;  // offset 36
        view.setFloat32(offset, this.config.gravity[2], true); offset += 4;  // offset 40
        view.setFloat32(offset, 0.0, true); offset += 4;  // offset 44 PADDING! WGSL vec3 alignment!

        // Row 4: dt, enableBuoyancy, beta, smagorinskyConstant
        view.setFloat32(offset, this.config.dt, true); offset += 4;  // offset 48
        view.setUint32(offset, this.config.enableBuoyancy ? 1 : 0, true); offset += 4;  // offset 52
        view.setFloat32(offset, this.config.beta, true); offset += 4;  // offset 56
        view.setFloat32(offset, this.config.smagorinskyConstant, true); offset += 4;  // offset 60

        // Calculate inlet/outlet planes based on wind direction
        // REQUIRE windDirection - no fallback
        if (this.config.windDirection === undefined || this.config.windDirection === null) {
            throw new Error('[LBMSolver] windDirection is required. Load EPW file or select wind direction in UI.');
        }
        const windDirDeg = this.config.windDirection;
        const dir = ((windDirDeg % 360) + 360) % 360;

        let inletPlane, outletPlane, windDirX, windDirY, windDirZ;

        // Determine primary flow direction based on wind coming FROM direction
        if (dir >= 315 || dir < 45) {
            // North wind (from N) → flow towards South (+Z direction)
            inletPlane = 4;  // Z_MIN
            outletPlane = 5; // Z_MAX
            windDirX = 0.0;
            windDirZ = 1.0;
        } else if (dir >= 45 && dir < 135) {
            // East wind (from E) → flow towards West (-X direction)
            inletPlane = 1;  // X_MAX
            outletPlane = 0; // X_MIN
            windDirX = -1.0;
            windDirZ = 0.0;
        } else if (dir >= 135 && dir < 225) {
            // South wind (from S) → flow towards North (-Z direction)
            inletPlane = 5;  // Z_MAX
            outletPlane = 4; // Z_MIN
            windDirX = 0.0;
            windDirZ = -1.0;
        } else {
            // West wind (from W) → flow towards East (+X direction)
            inletPlane = 0;  // X_MIN
            outletPlane = 1; // X_MAX
            windDirX = 1.0;
            windDirZ = 0.0;
        }

        // CRITICAL: Store for step() dispatch calculation
        this.inletPlane = inletPlane;
        this.outletPlane = outletPlane;

        windDirY = 0.0; // Horizontal wind (no vertical component)

        // REQUIRE inletVelocity and terrainRoughness - no fallbacks
        if (this.config.inletVelocity === undefined || this.config.inletVelocity === null) {
            throw new Error('[LBMSolver] inletVelocity is required. Load EPW file or set wind speed in UI.');
        }
        if (this.config.terrainRoughness === undefined || this.config.terrainRoughness === null) {
            throw new Error('[LBMSolver] terrainRoughness is required. Default value (0.35) should be provided by application.');
        }
        const windSpeed = this.config.inletVelocity;

        // Row 5: enableLES, inletVelocity, terrainRoughness, windDirX
        view.setUint32(offset, this.config.enableLES ? 1 : 0, true); offset += 4;  // offset 64
        view.setFloat32(offset, windSpeed, true); offset += 4;  // offset 68
        view.setFloat32(offset, this.config.terrainRoughness, true); offset += 4;  // offset 72
        view.setFloat32(offset, windDirX, true); offset += 4;  // offset 76

        // Row 6: windDirY, windDirZ, inletPlane, outletPlane
        view.setFloat32(offset, windDirY, true); offset += 4;  // offset 80
        view.setFloat32(offset, windDirZ, true); offset += 4;  // offset 84
        view.setUint32(offset, inletPlane, true); offset += 4;  // offset 88
        view.setUint32(offset, outletPlane, true); offset += 4;  // offset 92

        // Row 7: padding to 112 bytes
        view.setUint32(offset, 0, true); offset += 4;  // offset 96
        view.setUint32(offset, 0, true); offset += 4;  // offset 100
        view.setUint32(offset, 0, true); offset += 4;  // offset 104
        view.setUint32(offset, 0, true); offset += 4;  // offset 108

        console.log('[LBMSolver] Uniform buffer:', {
            nx, ny, nz,
            resolution: this.gridConfig.resolution,
            tau: this.config.tau,
            omega: 1.0 / this.config.tau,
            enableBuoyancy: this.config.enableBuoyancy,
            inletVelocity: windSpeed,
            windDirection: windDirDeg,
            windDirX: windDirX.toFixed(3),
            windDirZ: windDirZ.toFixed(3),
            inletPlane,
            outletPlane,
            terrainRoughness: this.config.terrainRoughness,
            totalBytes: offset
        });
        console.log(`[LBMSolver] CRITICAL: totalBytes=${offset}, windDirX=${windDirX}, windDirZ=${windDirZ}, inletPlane=${inletPlane}`);

        this.uniformBuffer = this.device.createBuffer({
            label: 'LBM Uniform Buffer',
            size: uniformBuffer.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformBuffer);
        // Distribution functions (f_in, f_out)
        const fSize = Math.floor(totalVoxels * directions * 4); // float32, ensure integer
        this.fInBuffer = this.device.createBuffer({
            label: 'LBM f_in Buffer',
            size: fSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.fOutBuffer = this.device.createBuffer({
            label: 'LBM f_out Buffer',
            size: fSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        // Инициализация equilibrium distribution
        const initialF = new Float32Array(totalVoxels * directions);
        for (let i = 0; i < totalVoxels; i++) {
            // f_eq для покоя: rho * w_i
            initialF[i * directions + 0] = this.config.rho0 * (1.0 / 3.0); // rest particle
            for (let q = 1; q < directions; q++) {
                initialF[i * directions + q] = this.config.rho0 * (q < 7 ? 1.0 / 18.0 : 1.0 / 36.0);
            }
        }
        this.device.queue.writeBuffer(this.fInBuffer, 0, initialF);
        this.device.queue.writeBuffer(this.fOutBuffer, 0, initialF);
        // Macroscopic fields - ensure all sizes are integers
        const densitySize = Math.floor(totalVoxels * 4);
        const velocitySize = Math.floor(totalVoxels * 3 * 4);
        const tempSize = initialTemperature?.byteLength || Math.floor(totalVoxels * 4);

        console.log('[LBMSolver] Buffer sizes:', { fSize, densitySize, velocitySize, tempSize, totalVoxels });

        this.densityBuffer = this.device.createBuffer({
            label: 'LBM Density Buffer',
            size: densitySize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this.velocityBuffer = this.device.createBuffer({
            label: 'LBM Velocity Buffer',
            size: velocitySize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Initialize density to rho0 and velocity to zero (CRITICAL for stability!)
        const initialDensity = new Float32Array(totalVoxels);
        initialDensity.fill(this.config.rho0);
        this.device.queue.writeBuffer(this.densityBuffer, 0, initialDensity);

        const initialVelocity = new Float32Array(totalVoxels * 3);
        initialVelocity.fill(0.0); // All velocities start at zero
        this.device.queue.writeBuffer(this.velocityBuffer, 0, initialVelocity);
        this.temperatureBuffer = this.device.createBuffer({
            label: 'LBM Temperature Buffer',
            size: tempSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        if (initialTemperature) {
            this.device.queue.writeBuffer(this.temperatureBuffer, 0, initialTemperature);
        }

        // Handle both GPUBuffer and TypedArray for voxelStateData
        if (voxelStateData instanceof GPUBuffer) {
            // Already a GPUBuffer, use directly
            this.voxelStateBuffer = voxelStateData;
        } else if (voxelStateData && (voxelStateData.byteLength || voxelStateData.size)) {
            // TypedArray - create buffer and copy data
            const bufferSize = voxelStateData.byteLength || voxelStateData.size;
            this.voxelStateBuffer = this.device.createBuffer({
                label: 'Voxel State Buffer',
                size: bufferSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.device.queue.writeBuffer(this.voxelStateBuffer, 0, voxelStateData);
        } else {
            // NO FALLBACK - throw error
            throw new Error('[LBMSolver] No voxelStateData provided. LBM requires valid voxel buffer from Voxelizer.');
        }
    }
    /**
     * TRL 7: Calculate adaptive time step based on CFL and Mach number constraints
     *
     * CFL condition: dt ≤ CFL_factor × dx / u_max
     * Mach constraint: u_max / c_s ≤ Ma_max  =>  dt ≤ Ma_max × c_s × dx / u_max
     *
     * For LBM: c_s = 1/sqrt(3) in lattice units, physical c_s = dx/dt × 1/sqrt(3)
     */
    async calculateAdaptiveDt() {
        if (!this.config.enableAdaptiveDt) {
            return this.config.dt;
        }
        // Get maximum velocity from field
        const maxVel = await this.getMaxVelocity();
        this.lastMaxVelocity = maxVel;
        const dx = this.gridConfig.resolution;
        // Physical speed of sound at 20°C: ~343 m/s
        // In lattice units: c_s = 1/sqrt(3) ≈ 0.577
        const c_s_physical = 343; // m/s
        const c_s_lattice = this.CS;
        // Calculate current Mach number
        this.lastMachNumber = maxVel / c_s_physical;
        if (maxVel < 1e-6) {
            // Very low velocity, use maximum dt
            return this.config.dtMax;
        }
        // CFL constraint: dt ≤ CFL × dx / u_max
        const dt_cfl = this.config.cflFactor * dx / maxVel;
        // Mach constraint: Ma = u / c_s < Ma_max
        // In LBM, lattice velocity u_lat = u_phys × dt / dx
        // For Ma < 0.1: u_lat / c_s_lat < 0.1
        // => u_phys × dt / dx / (1/sqrt(3)) < 0.1
        // => dt < 0.1 × dx × sqrt(3) / u_phys
        const dt_mach = this.config.maxMach * dx * Math.sqrt(3) / maxVel;
        // Take minimum of constraints
        let newDt = Math.min(dt_cfl, dt_mach);
        // Apply bounds
        newDt = Math.max(this.config.dtMin, Math.min(newDt, this.config.dtMax));
        // Smooth transition (exponential moving average to avoid oscillations)
        const alpha = 0.3; // Smoothing factor
        newDt = alpha * newDt + (1 - alpha) * this.currentDt;
        // Store history for analysis
        this.dtHistory.push(newDt);
        if (this.dtHistory.length > 100) {
            this.dtHistory.shift();
        }
        return newDt;
    }
    /**
     * Get maximum velocity in the field
     */
    async getMaxVelocity() {
        const stagingBuffer = this.device.createBuffer({
            size: this.velocityBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.velocityBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const velocityData = new Float32Array(stagingBuffer.getMappedRange());
        let maxVel = 0;
        const totalVoxels = velocityData.length / 3;
        // Sample every 4th voxel for performance
        for (let i = 0; i < totalVoxels; i += 4) {
            const vx = velocityData[i * 3];
            const vy = velocityData[i * 3 + 1];
            const vz = velocityData[i * 3 + 2];
            const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
            maxVel = Math.max(maxVel, vMag);
        }
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        return maxVel;
    }
    /**
     * TRL 7: Get stability metrics for monitoring
     */
    async getStabilityMetrics() {
        const maxVel = this.lastMaxVelocity;
        const dx = this.gridConfig.resolution;
        const c_s_physical = 343; // m/s
        const machNumber = maxVel / c_s_physical;
        const cflNumber = maxVel * this.currentDt / dx;
        // Stability margin: how far from critical values
        const machMargin = 1 - (machNumber / this.config.maxMach);
        const cflMargin = 1 - (cflNumber / 1.0); // CFL < 1 for stability
        const stabilityMargin = Math.min(machMargin, cflMargin);
        const isStable = machNumber < this.config.maxMach && cflNumber < 1.0;
        return {
            currentDt: this.currentDt,
            maxVelocity: maxVel,
            machNumber,
            cflNumber,
            isStable,
            stabilityMargin: Math.max(0, stabilityMargin)
        };
    }
    /**
     * Update uniforms with new dt value
     */
    updateDtUniform(dt) {
        this.currentDt = dt;
        // dt is at offset 11 in uniform buffer (after gravity vec3)
        const dtData = new Float32Array([dt]);
        this.device.queue.writeBuffer(this.uniformBuffer, 11 * 4, dtData);
    }
    /**
     * Выполнить один шаг симуляции (collision + streaming + BC)
     * TRL 7: Now with adaptive time stepping
     */
    async step() {
        if (!this.initialized) {
            throw new Error('[LBMSolver] Not initialized. Call initialize() first.');
        }
        // TRL 7: Adaptive time stepping
        if (this.config.enableAdaptiveDt &&
            this.currentStep % this.config.dtUpdateInterval === 0) {
            const newDt = await this.calculateAdaptiveDt();
            if (Math.abs(newDt - this.currentDt) / this.currentDt > 0.01) {
                this.updateDtUniform(newDt);
                console.log(`[LBMSolver] Adaptive dt: ${(newDt * 1000).toFixed(3)}ms (Ma=${this.lastMachNumber.toFixed(4)})`);
            }
        }
        const commandEncoder = this.device.createCommandEncoder({
            label: 'LBM Step Command Encoder',
        });
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const workgroupSize = 4;
        const dispatchX = Math.ceil(nx / workgroupSize);
        const dispatchY = Math.ceil(ny / workgroupSize);
        const dispatchZ = Math.ceil(nz / workgroupSize);

        // Standard LBM step order: Streaming → BC → Collision

        // 1. Streaming step: f_in[neighbor] → f_out (propagate distributions)
        const streamingPass = commandEncoder.beginComputePass({ label: 'Streaming Pass' });
        streamingPass.setPipeline(this.streamingPipeline);
        streamingPass.setBindGroup(0, this.bindGroup);
        streamingPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        streamingPass.end();

        // 2. Copy streamed f_out → f_in
        commandEncoder.copyBufferToBuffer(this.fOutBuffer, 0, this.fInBuffer, 0, this.fOutBuffer.size);

        // 3. Boundary conditions (writes to f_out at inlet/outlet)
        // BC kernels use @workgroup_size(8, 8) in WGSL
        const bcWorkgroupSize = 8;

        // CRITICAL FIX: Dispatch dimensions must match inlet/outlet plane orientation
        // inletPlane 0,1 = X plane (dispatch over YZ), inletPlane 4,5 = Z plane (dispatch over XY)
        const inletOnXPlane = this.inletPlane < 2;
        const [inletDispX, inletDispY] = inletOnXPlane ? [ny, nz] : [nx, ny];
        const outletOnXPlane = this.outletPlane < 2;
        const [outletDispX, outletDispY] = outletOnXPlane ? [ny, nz] : [nx, ny];

        console.log(`[LBMSolver] BC dispatch: inlet=${this.inletPlane} (${inletDispX}x${inletDispY}), outlet=${this.outletPlane} (${outletDispX}x${outletDispY})`);

        const bcPass = commandEncoder.beginComputePass({ label: 'BC Pass' });
        bcPass.setPipeline(this.inletBCPipeline);
        bcPass.setBindGroup(0, this.bindGroup);
        bcPass.dispatchWorkgroups(Math.ceil(inletDispX / bcWorkgroupSize), Math.ceil(inletDispY / bcWorkgroupSize), 1);
        bcPass.setPipeline(this.outletBCPipeline);
        bcPass.setBindGroup(0, this.bindGroup);
        bcPass.dispatchWorkgroups(Math.ceil(outletDispX / bcWorkgroupSize), Math.ceil(outletDispY / bcWorkgroupSize), 1);
        bcPass.end();

        // 4. Copy BC-applied f_out → f_in
        commandEncoder.copyBufferToBuffer(this.fOutBuffer, 0, this.fInBuffer, 0, this.fOutBuffer.size);

        // 5. Collision step: f_in → f_out, updates velocity buffer
        const collisionPass = commandEncoder.beginComputePass({ label: 'Collision Pass' });
        collisionPass.setPipeline(this.collisionPipeline);
        collisionPass.setBindGroup(0, this.bindGroup);
        collisionPass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
        collisionPass.end();

        // 6. Final copy: f_out → f_in for next iteration
        commandEncoder.copyBufferToBuffer(this.fOutBuffer, 0, this.fInBuffer, 0, this.fOutBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        this.currentStep++;
    }
    /**
     * Get current time step value
     */
    getCurrentDt() {
        return this.currentDt;
    }
    /**
     * Get dt history for analysis
     */
    getDtHistory() {
        return [...this.dtHistory];
    }
    /**
     * Получить текущий snapshot симуляции для visualization
     */
    async getSnapshot() {
        const { totalVoxels } = this.gridConfig;
        // Readback buffers
        const stagingDensity = this.device.createBuffer({
            size: this.densityBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const stagingVelocity = this.device.createBuffer({
            size: this.velocityBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.densityBuffer, 0, stagingDensity, 0, stagingDensity.size);
        commandEncoder.copyBufferToBuffer(this.velocityBuffer, 0, stagingVelocity, 0, stagingVelocity.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await stagingDensity.mapAsync(GPUMapMode.READ);
        await stagingVelocity.mapAsync(GPUMapMode.READ);
        const densityData = new Float32Array(stagingDensity.getMappedRange()).slice();
        const velocityData = new Float32Array(stagingVelocity.getMappedRange()).slice();

        // EXPOSE FOR VISUALIZATION
        window.AHI_Results = window.AHI_Results || {};
        window.AHI_Results.airflow = {
            velocityField: velocityData,
            gridConfig: {
                ...this.gridConfig,
                inletPlane: this.inletPlane,
                outletPlane: this.outletPlane
            }
        };
        console.log('[LBMSolver] Cached velocityData in window.AHI_Results.airflow, inletPlane:', this.inletPlane);

        stagingDensity.unmap();
        stagingVelocity.unmap();
        stagingDensity.destroy();
        stagingVelocity.destroy();
        // Readback temperature if available
        let tempArray = null;
        if (this.temperatureBuffer) {
            const stagingTemp = this.device.createBuffer({
                size: this.temperatureBuffer.size,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const cmd = this.device.createCommandEncoder();
            cmd.copyBufferToBuffer(this.temperatureBuffer, 0, stagingTemp, 0, stagingTemp.size);
            this.device.queue.submit([cmd.finish()]);
            await stagingTemp.mapAsync(GPUMapMode.READ);
            tempArray = new Float32Array(stagingTemp.getMappedRange()).slice();
            stagingTemp.unmap();
            stagingTemp.destroy();
        }

        // Calculate aggregated metrics
        let sumTemp = 0;
        let maxVel = 0;
        let sumVel = 0;
        let nonZeroVelCount = 0;
        let nanCount = 0;

        for (let i = 0; i < totalVoxels; i++) {
            if (tempArray) sumTemp += tempArray[i]; // Use actual temperature
            const vx = velocityData[i * 3];
            const vy = velocityData[i * 3 + 1];
            const vz = velocityData[i * 3 + 2];
            const vmag = Math.sqrt(vx * vx + vy * vy + vz * vz);

            if (isNaN(vmag)) {
                nanCount++;
            } else {
                if (vmag > maxVel) maxVel = vmag;
                if (vmag > 0.001) {
                    nonZeroVelCount++;
                    sumVel += vmag;
                }
            }
        }

        const avgVel = nonZeroVelCount > 0 ? sumVel / nonZeroVelCount : 0;
        const avgTemp = tempArray ? (sumTemp / totalVoxels) : 20.0; // Default to 20C if no CHT

        console.log(`[LBMSolver] getSnapshot DEBUG:`);
        console.log(`  - avgTemperature: ${avgTemp.toFixed(2)} K (${(avgTemp - 273.15).toFixed(2)} C)`);
        console.log(`  - avgDensity (was confused with temp): ${(densityData.reduce((a, b) => a + b, 0) / totalVoxels).toFixed(4)}`);

        // Sample first few non-zero velocities
        let sampleCount = 0;
        for (let i = 0; i < totalVoxels && sampleCount < 5; i++) {
            const vx = velocityData[i * 3];
            const vy = velocityData[i * 3 + 1];
            const vz = velocityData[i * 3 + 2];
            const vmag = Math.sqrt(vx * vx + vy * vy + vz * vz);
            if (vmag > 0.001) {
                console.log(`  - Sample[${i}]: vx=${vx.toFixed(3)}, vy=${vy.toFixed(3)}, vz=${vz.toFixed(3)}, |v|=${vmag.toFixed(3)}`);
                sampleCount++;
            }
        }

        // Calculate entropy metric
        const flowComplexity = this.calculateEntropyMetric(velocityData);
        return {
            timestamp: this.currentStep * this.config.dt,
            gridConfig: this.gridConfig,
            temperatureField: new Float32Array(0), // TODO: link to CHT
            velocityField: velocityData,
            comfortField: new Float32Array(0), // TODO: compute PMV
            metrics: {
                avgTemperature: avgTemp - 273.15, // Return Celsius for UI
                maxVelocity: maxVel,
                avgVelocity: avgVel,
                nonZeroVelCount: nonZeroVelCount,
                avgPMV: 0,
                co2Max: 0,
                energyConsumption: 0,
                flowComplexityIndex: flowComplexity,
            },
        };
    }
    /**
     * Calculate Flow Complexity Index based on velocity field entropy.
     * "Edge of Chaos" analysis.
     */
    calculateEntropyMetric(velocityData) {
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const totalVoxels = nx * ny * nz;
        // 1. Calculate local variability (vorticity approximation)
        // We'll use a simplified approach: magnitude of curl (vorticity) or just gradient magnitude
        // For speed, let's use gradient of velocity magnitude
        let totalGradient = 0;
        let maxGradient = 0;
        const gradients = []; // Sampled gradients for entropy
        // Sampling step to avoid O(N) heavy calc if N is huge, but for GPU readback we already have the data
        // We'll sample every 2nd voxel to save CPU time
        const step = 2;
        for (let k = 1; k < nz - 1; k += step) {
            for (let j = 1; j < ny - 1; j += step) {
                for (let i = 1; i < nx - 1; i += step) {
                    const idx = i + j * nx + k * nx * ny;
                    // Center velocity
                    const vx = velocityData[idx * 3];
                    const vy = velocityData[idx * 3 + 1];
                    const vz = velocityData[idx * 3 + 2];
                    const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
                    // Neighbors (just 6-connectivity for gradient)
                    // x+1
                    const idx_px = (i + 1) + j * nx + k * nx * ny;
                    const vMag_px = Math.sqrt(velocityData[idx_px * 3] ** 2 + velocityData[idx_px * 3 + 1] ** 2 + velocityData[idx_px * 3 + 2] ** 2);
                    // y+1
                    const idx_py = i + (j + 1) * nx + k * nx * ny;
                    const vMag_py = Math.sqrt(velocityData[idx_py * 3] ** 2 + velocityData[idx_py * 3 + 1] ** 2 + velocityData[idx_py * 3 + 2] ** 2);
                    // z+1
                    const idx_pz = i + j * nx + (k + 1) * nx * ny;
                    const vMag_pz = Math.sqrt(velocityData[idx_pz * 3] ** 2 + velocityData[idx_pz * 3 + 1] ** 2 + velocityData[idx_pz * 3 + 2] ** 2);
                    // Gradient magnitude approx
                    const grad = Math.sqrt((vMag_px - vMag) ** 2 + (vMag_py - vMag) ** 2 + (vMag_pz - vMag) ** 2);
                    if (grad > 0.001) { // Ignore empty/still areas
                        gradients.push(grad);
                        maxGradient = Math.max(maxGradient, grad);
                    }
                }
            }
        }
        if (gradients.length === 0)
            return 0.0;
        // 2. Calculate Shannon Entropy of the gradient distribution
        // Histogram with 20 bins
        const bins = 20;
        const histogram = new Float32Array(bins);
        for (const grad of gradients) {
            const binIdx = Math.min(bins - 1, Math.floor((grad / maxGradient) * bins));
            histogram[binIdx]++;
        }
        // Normalize and compute entropy
        let entropy = 0;
        const totalSamples = gradients.length;
        for (let i = 0; i < bins; i++) {
            if (histogram[i] > 0) {
                const p = histogram[i] / totalSamples;
                entropy -= p * Math.log2(p);
            }
        }
        // Max possible entropy for 'bins' is log2(bins)
        const maxEntropy = Math.log2(bins); // ≈ 4.32 for 20 bins
        // Normalize index to 0-1
        const normalizedEntropy = entropy / maxEntropy;
        // 3. Interpret as "Life" metric
        // Too low (0.0-0.3) = Laminar/Dead
        // Too high (0.8-1.0) = Chaotic/Stress
        // Optimal (0.4-0.7) = "Edge of Chaos" (High score)
        // Let's return the raw entropy for now as the requested "FlowComplexityIndex"
        // The user asked for 0.0-1.0 complexity index.
        return normalizedEntropy;
    }
    /**
     * Get voxel state buffer for CHT coupling
     */
    getVoxelStateBuffer() {
        return this.voxelStateBuffer;
    }
    /**
     * Get velocity buffer for CHT coupling
     */
    getVelocityBuffer() {
        return this.velocityBuffer;
    }
    /**
     * Cleanup
     */
    destroy() {
        this.uniformBuffer?.destroy();
        this.fInBuffer?.destroy();
        this.fOutBuffer?.destroy();
        this.densityBuffer?.destroy();
        this.velocityBuffer?.destroy();
        this.temperatureBuffer?.destroy();
        this.voxelStateBuffer?.destroy();
        console.log('[LBMSolver] Destroyed');
    }
}
