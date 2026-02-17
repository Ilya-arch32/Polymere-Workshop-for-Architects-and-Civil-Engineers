/**
 * AHI 2.0 Ultimate - 3D FDTD Acoustic Solver
 *
 * Finite-Difference Time-Domain solver for acoustic wave propagation
 * Models diffraction, low-frequency resonances critical for atmosphere perception
 */
/**
 * Стандартные коэффициенты звукопоглощения
 */
export const MATERIAL_ABSORPTION = {
    AIR: { id: 0, name: 'Air', alpha: 0.0 }, // Воздух - не поглощает
    CONCRETE: { id: 1, name: 'Concrete', alpha: 0.02 }, // Бетон - очень отражающий
    WOOD: { id: 2, name: 'Wood', alpha: 0.15 }, // Дерево - умеренное поглощение
    GLASS: { id: 3, name: 'Glass', alpha: 0.04 }, // Стекло - отражающее
    CARPET: { id: 4, name: 'Carpet', alpha: 0.35 }, // Ковер - хорошее поглощение
    ACOUSTIC_PANEL: { id: 5, name: 'Acoustic Panel', alpha: 0.85 }, // Акустические панели
    BRICK: { id: 6, name: 'Brick', alpha: 0.03 }, // Кирпич
    GYPSUM: { id: 7, name: 'Gypsum Board', alpha: 0.10 }, // Гипсокартон
    CURTAIN: { id: 8, name: 'Heavy Curtain', alpha: 0.55 }, // Тяжелые шторы
};
export class AcousticSolver {
    device;
    gridConfig;
    // Wave field buffers (double buffering)
    pressureBufferA;
    pressureBufferB;
    velocityBuffer;
    // Compute pipelines
    velocityPipeline;
    pressurePipeline;
    analysisPipeline;
    // Physical constants
    c = 343.0; // Speed of sound m/s
    rho = 1.225; // Air density kg/m³
    K; // Bulk modulus
    // Voxel data for Sabine calculation
    voxelStateBuffer = null;
    voxelStateData = null;
    // Cached surface area data
    surfaceAreaByMaterial = new Map();
    totalVolume = 0;
    surfaceAreaCalculated = false;
    // Room Impulse Response recording
    rirBuffer = null;
    rirSampleRate = 1000; // Hz (recording sample rate - 1000 samples/sec for efficiency)
    rirDuration = 2.0; // seconds (343m/s * 2.0s = 686m max propagation, enough for large rooms)
    rirData = null;
    rirRecorded = false;
    // Receiver position for RIR recording
    receiverPosition = { x: 5, y: 5, z: 1.5 };

    // Pressure field visualization snapshots
    pressureSnapshots = [];        // Array of { timestamp, pressureData: Float32Array }
    snapshotInterval = 50;         // Capture every N simulation steps
    maxSnapshots = 100;            // Limit memory usage
    materialAbsorptionMap = null;  // Map<voxelIndex, absorptionCoeff> for solid voxels
    soundSources = [];             // Array of { x, y, z } source positions
    constructor(device, gridConfig) {
        this.device = device;
        this.gridConfig = gridConfig;
        this.K = this.rho * this.c * this.c;
        this.cancelled = false;
        // Инициализируем коэффициенты поглощения по умолчанию
        for (const mat of Object.values(MATERIAL_ABSORPTION)) {
            this.surfaceAreaByMaterial.set(mat.id, 0);
        }
    }
    /**
     * Initialize FDTD solver with WebGPU resources
     */
    async initialize(voxelBuffer) {
        const { totalVoxels } = this.gridConfig;
        const { nx, ny, nz } = this.gridConfig.dimensions;
        // Сохраняем ссылку на voxel buffer для расчета площади поверхности
        this.voxelStateBuffer = voxelBuffer;
        // Allocate wave field buffers
        this.pressureBufferA = this.device.createBuffer({
            label: 'Pressure Buffer A',
            size: totalVoxels * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.pressureBufferB = this.device.createBuffer({
            label: 'Pressure Buffer B',
            size: totalVoxels * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        // Velocity has 3 components (vec4 with padding)
        this.velocityBuffer = this.device.createBuffer({
            label: 'Velocity Buffer',
            size: totalVoxels * 16, // vec4 = 16 bytes
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        // CRITICAL: Initialize all buffers to zeros!
        // This is essential for FDTD stability - starting from rest
        const zeroDataP = new Float32Array(totalVoxels);
        zeroDataP.fill(0.0);
        this.device.queue.writeBuffer(this.pressureBufferA, 0, zeroDataP);
        this.device.queue.writeBuffer(this.pressureBufferB, 0, zeroDataP);

        const zeroDataV = new Float32Array(totalVoxels * 4);
        zeroDataV.fill(0.0);
        this.device.queue.writeBuffer(this.velocityBuffer, 0, zeroDataV);

        console.log(`[AcousticSolver] Buffers initialized to zeros: pressure=${totalVoxels * 4} bytes, velocity=${totalVoxels * 16} bytes`);
        // Create FDTD compute shaders
        const shaderModule = this.device.createShaderModule({
            label: 'FDTD Acoustics Shaders',
            code: this.generateFDTDShaders()
        });

        // Create EXPLICIT bind group layout (auto-layout excludes unused bindings per entrypoint)
        this.fdtdBindGroupLayout = this.device.createBindGroupLayout({
            label: 'FDTD Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }
            ]
        });

        const fdtdPipelineLayout = this.device.createPipelineLayout({
            label: 'FDTD Pipeline Layout',
            bindGroupLayouts: [this.fdtdBindGroupLayout]
        });

        // Velocity update pipeline
        this.velocityPipeline = this.device.createComputePipeline({
            label: 'Velocity Update',
            layout: fdtdPipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'updateVelocity'
            }
        });
        // Pressure update pipeline
        this.pressurePipeline = this.device.createComputePipeline({
            label: 'Pressure Update',
            layout: fdtdPipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'updatePressure'
            }
        });
        // Analysis pipeline for metrics
        this.analysisPipeline = this.device.createComputePipeline({
            label: 'Acoustic Analysis',
            layout: fdtdPipelineLayout,
            compute: {
                module: shaderModule,
                entryPoint: 'analyzeAcoustics'
            }
        });

        // Create uniform buffer for FDTD
        // Uniforms struct: grid_size:vec3<u32>, pad, dt, c, rho, K, resolution, pad2, pad3, pad4
        this.uniformBuffer = this.device.createBuffer({
            label: 'Acoustic Uniforms',
            size: 48, // 12 floats (3 vec4 aligned blocks)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Create bind groups using EXPLICIT layout - all 5 bindings required
        // Bindings: 0=uniforms, 1=pressure_in, 2=pressure_out, 3=velocity, 4=voxels

        this.fdtdBindGroup = this.device.createBindGroup({
            label: 'FDTD Bind Group',
            layout: this.fdtdBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.pressureBufferA } },
                { binding: 2, resource: { buffer: this.pressureBufferB } },
                { binding: 3, resource: { buffer: this.velocityBuffer } },
                { binding: 4, resource: { buffer: voxelBuffer } }
            ]
        });

        // Use same bind group for both pipelines since they share layout
        this.velocityBindGroup = this.fdtdBindGroup;
        this.pressureBindGroup = this.fdtdBindGroup;

        // Initialize uniforms - MUST match WGSL struct layout:
        // struct Uniforms { grid_size: vec3<u32>, _pad, dt, c, rho, K, resolution, ... }
        // vec3<u32> at offset 0 (16 bytes with padding)
        // floats at offset 16: dt, c, rho, K, resolution
        const gridData = new Uint32Array([nx, ny, nz, 0]); // grid_size + padding at offset 0
        const resolution = this.gridConfig.resolution || 0.7; // Physical voxel size in meters
        const uniformData = new Float32Array([
            1.0 / 44100, // dt (time step based on sample rate)
            343.0,       // c (speed of sound m/s)
            1.225,       // rho (air density kg/m³)
            142000.0,    // K (bulk modulus of air Pa)
            resolution,  // Physical voxel size in meters
            0.0, 0.0, 0.0 // padding to 48 bytes total
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, gridData);  // offset 0
        this.device.queue.writeBuffer(this.uniformBuffer, 16, uniformData); // offset 16
        console.log(`[AcousticSolver] Uniforms: resolution=${resolution}m, grid=${nx}x${ny}x${nz}`);

        console.log('[AcousticSolver] Initialized 3D FDTD solver');
        // Allocate RIR buffer for Schroeder integration
        const rirSamples = Math.ceil(this.rirSampleRate * this.rirDuration);
        this.rirBuffer = this.device.createBuffer({
            label: 'RIR Buffer',
            size: rirSamples * 4, // float32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        this.rirData = new Float32Array(rirSamples);
    }
    /**
     * Reset pressure and velocity buffers to zero
     */
    resetState() {
        const { totalVoxels } = this.gridConfig;

        const zeroDataP = new Float32Array(totalVoxels);
        zeroDataP.fill(0.0);
        this.device.queue.writeBuffer(this.pressureBufferA, 0, zeroDataP);
        this.device.queue.writeBuffer(this.pressureBufferB, 0, zeroDataP);

        const zeroDataV = new Float32Array(totalVoxels * 4);
        zeroDataV.fill(0.0);
        this.device.queue.writeBuffer(this.velocityBuffer, 0, zeroDataV);

        console.log('[AcousticSolver] Simulation state reset (buffers cleared)');
    }
    /**
     * Set receiver position for RIR measurement
     */
    setReceiverPosition(x, y, z) {
        this.receiverPosition = { x, y, z };
        this.rirRecorded = false; // Invalidate cached RIR
    }
    /**
     * Run FDTD simulation step
     */
    async step(dt) {
        const commandEncoder = this.device.createCommandEncoder();
        const { nx, ny, nz } = this.gridConfig.dimensions;
        // Update velocity from pressure gradient
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.velocityPipeline);
            pass.setBindGroup(0, this.velocityBindGroup);
            pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
            pass.end();
        }
        // Update pressure from velocity divergence
        {
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.pressurePipeline);
            pass.setBindGroup(0, this.pressureBindGroup);
            pass.dispatchWorkgroups(Math.ceil(nx / 4), Math.ceil(ny / 4), Math.ceil(nz / 4));
            pass.end();
        }
        // Swap buffers: Copy pressureBufferB back to pressureBufferA
        // NOTE: Pointer swap breaks bind groups! Must use buffer copy.
        commandEncoder.copyBufferToBuffer(
            this.pressureBufferB, 0,
            this.pressureBufferA, 0,
            this.pressureBufferB.size
        );
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
    }
    /**
     * Inject acoustic impulse (for room response)
     */
    async injectImpulse(x, y, z) {
        const { nx, ny, nz, totalVoxels } = this.gridConfig.dimensions || this.gridConfig;

        // Clamp source position to grid bounds
        const safeX = Math.max(0, Math.min(nx - 1, x));
        const safeY = Math.max(0, Math.min(ny - 1, y));
        const safeZ = Math.max(0, Math.min(nz - 1, z));

        const idx = safeX + safeY * nx + safeZ * nx * ny;
        const maxIdx = totalVoxels || (nx * ny * nz);

        if (idx >= maxIdx || idx < 0) {
            console.error(`[AcousticSolver] Impulse index ${idx} out of bounds (max ${maxIdx})`);
            return;
        }

        // Gaussian pulse injection
        const impulseData = new Float32Array(1);
        impulseData[0] = 1000.0; // Pressure amplitude
        console.log(`[AcousticSolver] Injecting impulse at voxel ${idx} (${safeX}, ${safeY}, ${safeZ})`);
        this.device.queue.writeBuffer(this.pressureBufferA, idx * 4, impulseData);
    }
    /**
     * Apply Perfectly Matched Layer (PML) boundaries
     * 
     * NOTE: PML is implemented directly in the WGSL shader (updateVelocity function)
     * for GPU efficiency. The shader applies exponential damping in boundary regions:
     * - pml_width = 10 voxels from each grid edge
     * - sigma = 10.0 damping coefficient (exp(-10*dt) ≈ 0.5% loss per step)
     * 
     * This method exists for API compatibility but the actual PML is GPU-side.
     * @see generateFDTDShaders() lines 1026-1036 for WGSL implementation
     */
    applyPML() {
        // PML absorbing boundaries implemented in WGSL shader (updateVelocity)
        // See: sigma damping at grid boundaries to prevent wave reflections
    }
    /**
     * Расчет площади поверхности для каждого материала
     * Подсчитывает граничные воксели (Solid рядом с Fluid)
     */
    async calculateSurfaceArea() {
        if (!this.voxelStateBuffer) {
            console.warn('[AcousticSolver] Voxel buffer not initialized');
            return this.surfaceAreaByMaterial;
        }
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const totalVoxels = nx * ny * nz;
        const voxelVolume = Math.pow(this.gridConfig.resolution, 3);
        const faceArea = Math.pow(this.gridConfig.resolution, 2);
        // Считываем данные вокселей из GPU
        const stagingBuffer = this.device.createBuffer({
            size: this.voxelStateBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.voxelStateBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
        this.device.queue.submit([commandEncoder.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        // Voxel buffer is 8 floats per voxel: [state, material, pad, temp, vx, vy, vz, pad]
        this.voxelStateData = new Float32Array(stagingBuffer.getMappedRange()).slice();
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        // Сбрасываем счетчики площадей (but preserve volume if set from RoomDetector)
        this.surfaceAreaByMaterial.clear();
        for (const mat of Object.values(MATERIAL_ABSORPTION)) {
            this.surfaceAreaByMaterial.set(mat.id, 0);
        }

        // Only calculate volume from voxels if not already set from RoomDetector
        const presetVolume = this.totalVolume;
        let calculatedVolume = 0;

        // Проходим по всем вокселям
        const VOXEL_FLUID = 2;
        const VOXEL_SOLID = 1;
        const stride = 8; // 8 floats per voxel as per Voxelizer format
        const neighbors = [
            [-1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, 1, 0],
            [0, 0, -1], [0, 0, 1]
        ];
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    // Voxel format: [state, material, pad, temp, vx, vy, vz, pad]
                    const state = Math.round(this.voxelStateData[idx * stride]);
                    const material = Math.round(this.voxelStateData[idx * stride + 1]);
                    // Считаем объем воздуха (Fluid voxels)
                    if (state === VOXEL_FLUID) {
                        calculatedVolume += voxelVolume;
                    }
                    // Ищем границы Solid-Fluid
                    if (state === VOXEL_SOLID) {
                        for (const [dx, dy, dz] of neighbors) {
                            const nx2 = x + dx;
                            const ny2 = y + dy;
                            const nz2 = z + dz;
                            // Проверка границ
                            if (nx2 >= 0 && nx2 < nx &&
                                ny2 >= 0 && ny2 < ny &&
                                nz2 >= 0 && nz2 < nz) {
                                const neighborIdx = nx2 + ny2 * nx + nz2 * nx * ny;
                                const neighborState = Math.round(this.voxelStateData[neighborIdx * stride]);
                                // Если сосед - воздух, эта грань - поверхность
                                if (neighborState === VOXEL_FLUID) {
                                    const currentArea = this.surfaceAreaByMaterial.get(material) || 0;
                                    this.surfaceAreaByMaterial.set(material, currentArea + faceArea);
                                }
                            }
                        }
                    }
                }
            }
        }
        this.surfaceAreaCalculated = true;

        // Diagnostic: count voxel states (scan ALL voxels, not just first 1000)
        let stateCount = { 0: 0, 1: 0, 2: 0, other: 0 };
        const totalVoxelCount = Math.floor(this.voxelStateData.length / stride);
        for (let i = 0; i < totalVoxelCount; i++) {
            const s = Math.round(this.voxelStateData[i * stride]);
            if (s === 0) stateCount[0]++;
            else if (s === 1) stateCount[1]++;
            else if (s === 2) stateCount[2]++;
            else stateCount.other++;
        }
        console.log(`[AcousticSolver] Voxel state (ALL ${totalVoxelCount}): EMPTY=${stateCount[0]}, SOLID=${stateCount[1]}, FLUID=${stateCount[2]}, other=${stateCount.other}`);

        // Find and log first SOLID voxel
        let firstSolidIdx = -1;
        for (let i = 0; i < totalVoxelCount && firstSolidIdx < 0; i++) {
            if (Math.round(this.voxelStateData[i * stride]) === 1) {
                firstSolidIdx = i;
            }
        }
        if (firstSolidIdx >= 0) {
            console.log(`[AcousticSolver] First SOLID voxel at idx=${firstSolidIdx}:`,
                Array.from(this.voxelStateData.slice(firstSolidIdx * stride, firstSolidIdx * stride + 8)));
        } else {
            console.warn(`[AcousticSolver] WARNING: No SOLID voxels found! Check voxelization.`);
        }

        // Use pre-set volume from RoomDetector if available, otherwise use calculated from voxels
        if (presetVolume > 0) {
            console.log(`[AcousticSolver] Using RoomDetector volume: ${presetVolume.toFixed(2)}m³ (calculated was: ${calculatedVolume.toFixed(2)}m³)`);
            // totalVolume already set, don't overwrite
        } else {
            this.totalVolume = calculatedVolume;
            console.log(`[AcousticSolver] Using calculated volume: ${this.totalVolume.toFixed(2)}m³`);
        }

        console.log(`[AcousticSolver] Surface area calculated: V=${this.totalVolume.toFixed(2)}m³`);
        return this.surfaceAreaByMaterial;
    }
    /**
     * Record Room Impulse Response from FDTD simulation
     * Runs full FDTD simulation and records pressure at receiver position
     */
    async recordRIR(sourceX, sourceY, sourceZ, onProgress = null, numSteps = null) {
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const res = this.gridConfig.resolution;

        // CFL-based time step: dt < dx / c for stability
        // Using 0.5 * dx / c for safety margin
        const cflDt = 0.5 * res / this.c; // CFL condition: 0.5 * 0.7m / 343m/s ≈ 1ms
        const dt = Math.min(cflDt, 0.001); // Max 1ms for performance
        const totalSteps = numSteps || Math.ceil(this.rirDuration / dt);

        // Source coordinates are already in VOXEL UNITS from preview.html
        // (calculated as room centroid in voxel space)
        const srcVoxelX = Math.floor(sourceX);  // Already voxel index
        const srcVoxelY = Math.floor(sourceY);
        const srcVoxelZ = Math.floor(sourceZ);

        console.log(`[AcousticSolver] recordRIR using CFL dt=${(dt * 1000).toFixed(2)}ms (res=${res}m)`);
        console.log(`  - source: (${srcVoxelX}, ${srcVoxelY}, ${srcVoxelZ}), grid: ${nx}x${ny}x${nz}`);
        console.log(`  - totalSteps: ${totalSteps}, duration: ${this.rirDuration}s`);

        // CRITICAL: Update uniform buffer with correct dt!
        // Without this, shader uses wrong dt and sound doesn't propagate
        const resolution = this.gridConfig.resolution || 0.7;
        const uniformData = new Float32Array([
            dt,          // Updated dt (CFL-based, ~1ms)
            this.c,      // Speed of sound (343 m/s)
            this.rho,    // Air density (1.225 kg/m³)
            this.K,      // Bulk modulus (142000 Pa)
            resolution,  // Physical voxel size in meters
            0.0, 0.0, 0.0 // padding
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 16, uniformData);
        console.log(`[AcousticSolver] Updated uniforms: dt=${dt.toFixed(6)}s, c=${this.c}, rho=${this.rho}, K=${this.K}`);

        // CRITICAL: Update rirSampleRate based on actual sampling interval
        // RIR samples every 10 simulation steps (see line ~507)
        const rirSamplingInterval = 10; // steps between samples
        this.rirSampleRate = 1.0 / (rirSamplingInterval * dt);
        console.log(`[AcousticSolver] RIR effective sample rate: ${this.rirSampleRate.toFixed(1)}Hz (interval=${rirSamplingInterval}*${(dt * 1000).toFixed(2)}ms)`);

        console.log(`[AcousticSolver] Recording RIR: ${totalSteps} steps at dt=${(dt * 1000).toFixed(2)}ms`);

        // Inject impulse at source position
        await this.injectImpulse(srcVoxelX, srcVoxelY, srcVoxelZ);

        // Verify impulse was injected
        const srcIdx = srcVoxelX + srcVoxelY * nx + srcVoxelZ * nx * ny;
        const srcPressure = await this.readPressureAtPoint(srcIdx);
        console.log(`[AcousticSolver] Impulse check: pressure at source voxel ${srcIdx} = ${srcPressure}`);

        // Receiver voxel index - place 10m from source to ensure sound reaches it
        const totalVoxels = nx * ny * nz;

        // Calculate receiver position: 10m from source in -X direction (or use default if manual set)
        const desiredOffset = 10 / res; // 10 meters in voxel units
        let recX, recY, recZ;

        // Check if receiver was explicitly set far from default
        const defaultRec = { x: 5, y: 5, z: 1.5 };
        const isDefaultReceiver =
            Math.abs(this.receiverPosition.x - defaultRec.x) < 0.1 &&
            Math.abs(this.receiverPosition.y - defaultRec.y) < 0.1 &&
            Math.abs(this.receiverPosition.z - defaultRec.z) < 0.1;

        if (isDefaultReceiver) {
            // Move receiver to be 10m from source
            recX = Math.max(1, Math.min(nx - 2, srcVoxelX - Math.floor(desiredOffset)));
            recY = srcVoxelY; // Same Y
            recZ = srcVoxelZ; // Same Z
            console.log(`[AcousticSolver] Auto-positioned receiver ${desiredOffset.toFixed(0)} voxels from source`);
        } else {
            // Use manually set receiver position
            recX = Math.max(0, Math.min(nx - 1, Math.floor(this.receiverPosition.x / res)));
            recY = Math.max(0, Math.min(ny - 1, Math.floor(this.receiverPosition.y / res)));
            recZ = Math.max(0, Math.min(nz - 1, Math.floor(this.receiverPosition.z / res)));
        }

        const recIdx = recX + recY * nx + recZ * nx * ny;
        const distanceVoxels = Math.sqrt(Math.pow(srcVoxelX - recX, 2) + Math.pow(srcVoxelY - recY, 2) + Math.pow(srcVoxelZ - recZ, 2));
        const distanceMeters = distanceVoxels * res;
        const timeToReach = distanceMeters / 343; // Speed of sound

        console.log(`[AcousticSolver] Receiver: (${recX}, ${recY}, ${recZ}) -> idx ${recIdx}`);
        console.log(`[AcousticSolver] Distance source→receiver: ${distanceMeters.toFixed(1)}m, time to reach: ${(timeToReach * 1000).toFixed(0)}ms`);

        // Validate recIdx doesn't exceed buffer bounds
        const maxIdx = totalVoxels;
        if (recIdx >= maxIdx || recIdx < 0) {
            console.error(`[AcousticSolver] Receiver index ${recIdx} exceeds grid size ${maxIdx}! Using center.`);
            const centerIdx = Math.floor(maxIdx / 2);
            console.log(`[AcousticSolver] Using center voxel: ${centerIdx}`);
        }
        const safeRecIdx = Math.max(0, Math.min(maxIdx - 1, recIdx));

        // Run simulation and record pressure at receiver
        // FIXED: Use compact array with only sampled values (every samplingInterval steps)
        const samplingInterval = 10;
        const numRirSamples = Math.ceil(totalSteps / samplingInterval);
        const rirSamples = new Float32Array(numRirSamples);
        let sampleIdx = 0;
        let maxPressure = 0;
        let nonZeroCount = 0;

        console.log(`[AcousticSolver] Starting FDTD simulation: ${totalSteps} steps (should complete in ~${(totalSteps * 0.002).toFixed(1)}s)...`);
        const startTime = performance.now();

        // Clear previous snapshots and store source
        this.clearSnapshots();
        this.soundSources = [{ x: srcVoxelX, y: srcVoxelY, z: srcVoxelZ }];

        // Build material absorption map for visualization (async, but don't await - run in background)
        this.buildMaterialAbsorptionMap().catch(e => console.warn('[AcousticSolver] Material map failed:', e));

        for (let step = 0; step < totalSteps; step++) {
            if (this.cancelled) { console.log('[AcousticSolver] Cancelled at step', step); break; }
            await this.step(dt);

            if (onProgress && step % 50 === 0) {
                onProgress(Math.round((step / totalSteps) * 100));
            }

            // DEBUG: Track pressure propagation over multiple steps
            // FDTD propagation: pressure takes ~2 steps to propagate 1 voxel
            if (step === 1 || step === 2 || step === 5 || step === 10 || step === 20 || step === 50) {
                const srcP = await this.readPressureAtPoint(srcIdx);
                const neighborP = await this.readPressureAtPoint(srcIdx + 1);
                const neighborVel = await this.readVelocityAtPoint(srcIdx + 1);
                console.log(`[AcousticSolver] DEBUG step ${step}: srcP=${srcP.toFixed(2)}, neighborP=${neighborP.toFixed(4)}, neighborVel=${neighborVel.toFixed(6)}`);
            }

            // Read pressure at receiver position every samplingInterval steps
            if (step % samplingInterval === 0) {
                const pressure = await this.readPressureAtPoint(safeRecIdx);
                rirSamples[sampleIdx] = pressure;
                sampleIdx++;
                if (pressure !== 0) {
                    nonZeroCount++;
                    if (Math.abs(pressure) > maxPressure) {
                        maxPressure = Math.abs(pressure);
                    }
                }
            }

            // Capture pressure snapshots for 3D visualization (every snapshotInterval steps)
            if (step % this.snapshotInterval === 0 && this.pressureSnapshots.length < this.maxSnapshots) {
                const timestampMs = step * dt * 1000;
                await this.capturePressureSnapshot(timestampMs);
            }
        }
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        console.log(`[AcousticSolver] RIR complete in ${elapsed}s: maxP=${maxPressure.toFixed(4)}, nonZero=${nonZeroCount}/${numRirSamples}, samples=${sampleIdx}`);
        this.rirData = rirSamples;
        this.rirRecorded = true;
        console.log('[AcousticSolver] RIR recording complete');
        return {
            pressure: rirSamples,
            sampleRate: this.rirSampleRate,
            duration: this.rirDuration,
            sourcePosition: { x: sourceX, y: sourceY, z: sourceZ },
            receiverPosition: this.receiverPosition
        };
    }
    /**
     * Read pressure value at a specific voxel index
     */
    async readPressureAtPoint(voxelIdx) {
        const stagingBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.pressureBufferA, voxelIdx * 4, stagingBuffer, 0, 4);
        this.device.queue.submit([commandEncoder.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange());
        const pressure = data[0];
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        return pressure;
    }
    /**
     * Read velocity value at a specific voxel index (returns magnitude)
     */
    async readVelocityAtPoint(voxelIdx) {
        const stagingBuffer = this.device.createBuffer({
            size: 16, // vec4<f32>
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.velocityBuffer, voxelIdx * 16, stagingBuffer, 0, 16);
        this.device.queue.submit([commandEncoder.finish()]);
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange());
        const vx = data[0], vy = data[1], vz = data[2];
        const magnitude = Math.sqrt(vx * vx + vy * vy + vz * vz);
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        return magnitude;
    }

    /**
     * Read the entire pressure field from GPU for visualization
     * @returns {Promise<Float32Array>} Full pressure field data
     */
    async readFullPressureField() {
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const totalVoxels = nx * ny * nz;
        const bufferSize = totalVoxels * 4; // float32

        const stagingBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.pressureBufferA, 0, stagingBuffer, 0, bufferSize);
        this.device.queue.submit([commandEncoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange().slice(0));
        stagingBuffer.unmap();
        stagingBuffer.destroy();

        return data;
    }

    /**
     * Capture current pressure field as a snapshot for visualization
     * @param {number} timestampMs - Current simulation time in milliseconds
     */
    async capturePressureSnapshot(timestampMs) {
        if (this.pressureSnapshots.length >= this.maxSnapshots) {
            // Evict oldest snapshot to stay within memory limit
            this.pressureSnapshots.shift();
        }

        const pressureData = await this.readFullPressureField();
        this.pressureSnapshots.push({
            timestamp: timestampMs,
            pressureData: pressureData
        });

        console.log(`[AcousticSolver] Captured snapshot at ${timestampMs.toFixed(1)}ms (total: ${this.pressureSnapshots.length})`);
    }

    /**
     * Read voxel state data from GPU buffer into CPU memory
     */
    async readVoxelStateData() {
        if (!this.voxelStateBuffer) {
            console.warn('[AcousticSolver] No voxel state buffer available');
            return null;
        }

        const { nx, ny, nz } = this.gridConfig.dimensions;
        const totalVoxels = nx * ny * nz;
        const stride = 8; // 8 floats per voxel (matching VoxelState struct)
        const bufferSize = totalVoxels * stride * 4; // float32

        const stagingBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.voxelStateBuffer, 0, stagingBuffer, 0, bufferSize);
        this.device.queue.submit([commandEncoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        this.voxelStateData = new Float32Array(stagingBuffer.getMappedRange().slice(0));
        stagingBuffer.unmap();
        stagingBuffer.destroy();

        console.log(`[AcousticSolver] Read voxel state data: ${totalVoxels} voxels`);
        return this.voxelStateData;
    }

    /**
     * Get material absorption map for solid voxels (for wall soundproofing visualization)
     * @returns {Map<number, {material: number, alpha: number, x: number, y: number, z: number}>}
     */
    async buildMaterialAbsorptionMap() {
        // Ensure voxel state data is loaded
        if (!this.voxelStateData) {
            await this.readVoxelStateData();
        }

        if (!this.voxelStateData) {
            console.warn('[AcousticSolver] No voxel state data available for material map');
            return new Map();
        }

        const { nx, ny, nz } = this.gridConfig.dimensions;
        const stride = 8; // 8 floats per voxel
        const VOXEL_SOLID = 1;
        const map = new Map();

        // Get absorption coefficients lookup
        const alphaLookup = {};
        for (const mat of Object.values(MATERIAL_ABSORPTION)) {
            alphaLookup[mat.id] = mat.alpha;
        }

        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    const state = Math.round(this.voxelStateData[idx * stride]);
                    const material = Math.round(this.voxelStateData[idx * stride + 1]);

                    if (state === VOXEL_SOLID) {
                        const alpha = alphaLookup[material] !== undefined ? alphaLookup[material] : 0.02;
                        map.set(idx, { material, alpha, x, y, z });
                    }
                }
            }
        }

        this.materialAbsorptionMap = map;
        console.log(`[AcousticSolver] Built material absorption map: ${map.size} solid voxels`);
        return map;
    }

    /**
     * Get visualization data for acoustics (pressure snapshots + material map)
     * Note: buildMaterialAbsorptionMap should be called before this method
     */
    getVisualizationData() {
        return {
            snapshots: this.pressureSnapshots,
            materialMap: this.materialAbsorptionMap || new Map(),
            sources: this.soundSources,
            gridConfig: this.gridConfig
        };
    }

    /**
     * Clear all pressure snapshots (to free memory)
     */
    clearSnapshots() {
        this.pressureSnapshots = [];
        console.log('[AcousticSolver] Cleared pressure snapshots');
    }

    /**
     * Add a sound source position
     */
    addSoundSource(x, y, z) {
        this.soundSources.push({ x, y, z });
        console.log(`[AcousticSolver] Added sound source at (${x}, ${y}, ${z}), total: ${this.soundSources.length}`);
    }

    /**
     * Run simulation for multiple sources in parallel on GPU
     * Each source injects an impulse, and we capture snapshots of the combined field
     */
    async runMultiSourceSimulation(numSteps) {
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const res = this.gridConfig.resolution;
        const cflDt = 0.5 * res / this.c;
        const dt = Math.min(cflDt, 0.001);
        const totalSteps = numSteps || Math.ceil(this.rirDuration / dt);

        console.log(`[AcousticSolver] Running multi-source simulation: ${this.soundSources.length} sources, ${totalSteps} steps`);

        // Clear previous snapshots
        this.clearSnapshots();

        // Reset simulation state (clear buffers) to ensure clean wave propagation
        this.resetState();

        // Build material absorption map (async - reads voxel data from GPU)
        await this.buildMaterialAbsorptionMap();

        // Inject impulses at ALL source positions simultaneously
        for (const source of this.soundSources) {
            await this.injectImpulse(Math.floor(source.x), Math.floor(source.y), Math.floor(source.z));
        }

        // Run simulation and capture snapshots
        const snapshotTimestep = Math.max(1, Math.floor(totalSteps / this.maxSnapshots));
        const startTime = performance.now();

        for (let step = 0; step < totalSteps; step++) {
            if (this.cancelled) { console.log('[AcousticSolver] Cancelled at step', step); break; }
            await this.step(dt);

            // Capture snapshot at intervals
            if (step % this.snapshotInterval === 0 || step === totalSteps - 1) {
                const timestampMs = step * dt * 1000;
                await this.capturePressureSnapshot(timestampMs);
            }
        }

        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        console.log(`[AcousticSolver] Multi-source simulation complete in ${elapsed}s, ${this.pressureSnapshots.length} snapshots`);

        return this.getVisualizationData();
    }

    /**
     * Schroeder Integration for reverberation analysis
     * E(t) = ∫_t^∞ p²(τ)dτ
     *
     * This is the backward integration method from Schroeder (1965)
     * that gives the ensemble-averaged decay curve from a single RIR
     */
    calculateSchroederCurve(rir) {
        const n = rir.length;
        const energyCurve = new Float32Array(n);
        // Backward integration: E(t) = ∫_t^∞ p²(τ)dτ
        let cumulativeEnergy = 0;
        for (let i = n - 1; i >= 0; i--) {
            cumulativeEnergy += rir[i] * rir[i];
            energyCurve[i] = cumulativeEnergy;
        }
        // Normalize to start at 0 dB
        const maxEnergy = energyCurve[0];
        if (maxEnergy > 0) {
            for (let i = 0; i < n; i++) {
                energyCurve[i] = energyCurve[i] / maxEnergy;
            }
        }
        return energyCurve;
    }
    /**
     * Calculate reverberation time from Schroeder curve
     * Uses linear regression on the decay curve
     *
     * @param startDb Start of measurement range (e.g., -5 for T30)
     * @param endDb End of measurement range (e.g., -35 for T30)
     * @returns Reverberation time in seconds (extrapolated to -60dB)
     */
    calculateRTFromSchroeder(energyCurve, sampleRate, startDb = -5, endDb = -35) {
        const n = energyCurve.length;
        // Convert to dB
        const dbCurve = new Float32Array(n);
        let minDb = 0, maxDb = -100;
        for (let i = 0; i < n; i++) {
            dbCurve[i] = energyCurve[i] > 1e-10 ? 10 * Math.log10(energyCurve[i]) : -100;
            if (dbCurve[i] > maxDb) maxDb = dbCurve[i];
            if (dbCurve[i] < minDb && dbCurve[i] > -100) minDb = dbCurve[i];
        }
        console.log(`[AcousticSolver] Schroeder curve: n=${n}, maxDb=${maxDb.toFixed(1)}, minDb=${minDb.toFixed(1)}, sampleRate=${sampleRate.toFixed(1)}Hz`);

        // Find indices for start and end of decay range
        let startIdx = 0;
        let endIdx = n - 1;
        for (let i = 0; i < n; i++) {
            if (dbCurve[i] <= startDb && startIdx === 0) {
                startIdx = i;
            }
            if (dbCurve[i] <= endDb) {
                endIdx = i;
                break;
            }
        }
        console.log(`[AcousticSolver] Decay range: startIdx=${startIdx} (${(startIdx / sampleRate).toFixed(3)}s), endIdx=${endIdx} (${(endIdx / sampleRate).toFixed(3)}s), samples=${endIdx - startIdx}`);

        // Need at least 10 samples for reliable regression
        if (endIdx - startIdx < 10) {
            const errorMsg = `[AcousticSolver] ERROR: Insufficient decay range for RT60 calculation. ` +
                `Found ${endIdx - startIdx} samples (need 10+). ` +
                `Decay curve range: ${minDb.toFixed(1)}dB to ${maxDb.toFixed(1)}dB, ` +
                `but need ${startDb}dB to ${endDb}dB range. ` +
                `Possible causes: room too small, simulation duration too short, or no reflective surfaces.`;
            console.error(errorMsg);
            throw new Error(errorMsg);
        }
        // Linear regression on dB vs time
        // y = slope * x + intercept
        let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        const count = endIdx - startIdx + 1;
        for (let i = startIdx; i <= endIdx; i++) {
            const t = i / sampleRate;
            sumX += t;
            sumY += dbCurve[i];
            sumXY += t * dbCurve[i];
            sumX2 += t * t;
        }
        const slope = (count * sumXY - sumX * sumY) / (count * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / count;
        console.log(`[AcousticSolver] Regression: slope=${slope.toFixed(2)} dB/s, intercept=${intercept.toFixed(2)}`);

        // RT60 = time for 60dB decay
        // slope is dB/second, so RT60 = -60 / slope
        const rt60 = slope < -0.1 ? -60 / slope : -1;
        // EDT: Early Decay Time (first 10dB of decay)
        let edtIdx = 0;
        for (let i = 0; i < n; i++) {
            if (dbCurve[i] <= -10) {
                edtIdx = i;
                break;
            }
        }
        const edt = edtIdx > 0 ? (edtIdx / sampleRate) * 6 : rt60; // Extrapolate to 60dB
        // Calculate R² for confidence
        let ssTot = 0, ssRes = 0;
        const meanY = sumY / count;
        for (let i = startIdx; i <= endIdx; i++) {
            const t = i / sampleRate;
            const predicted = slope * t + intercept;
            ssTot += (dbCurve[i] - meanY) ** 2;
            ssRes += (dbCurve[i] - predicted) ** 2;
        }
        const rSquared = 1 - (ssRes / ssTot);
        console.log(`[AcousticSolver] RT60=${rt60.toFixed(2)}s (raw), EDT=${edt.toFixed(2)}s, R²=${rSquared.toFixed(3)}`);

        return {
            rt60: Math.max(0.1, Math.min(rt60, 10)),
            edt: Math.max(0.1, Math.min(edt, 10)),
            confidence: Math.max(0, rSquared)
        };
    }
    /**
     * Calculate Clarity indices (C50, C80) from RIR
     * C_t = 10 * log10(∫_0^t p²(τ)dτ / ∫_t^∞ p²(τ)dτ)
     */
    calculateClarity(rir, sampleRate, limitMs) {
        const limitSamples = Math.floor(sampleRate * limitMs / 1000);
        const n = rir.length;
        let earlyEnergy = 0;
        let lateEnergy = 0;
        for (let i = 0; i < n; i++) {
            const energy = rir[i] * rir[i];
            if (i < limitSamples) {
                earlyEnergy += energy;
            }
            else {
                lateEnergy += energy;
            }
        }
        if (lateEnergy < 1e-10) {
            return 20; // Very high clarity (dry room)
        }
        return 10 * Math.log10(earlyEnergy / lateEnergy);
    }
    /**
     * Calculate Definition D50 from RIR
     * D50 = ∫_0^50ms p²(τ)dτ / ∫_0^∞ p²(τ)dτ
     */
    calculateDefinition(rir, sampleRate) {
        const limit50ms = Math.floor(sampleRate * 0.05);
        const n = rir.length;
        let earlyEnergy = 0;
        let totalEnergy = 0;
        for (let i = 0; i < n; i++) {
            const energy = rir[i] * rir[i];
            totalEnergy += energy;
            if (i < limit50ms) {
                earlyEnergy += energy;
            }
        }
        return totalEnergy > 0 ? earlyEnergy / totalEnergy : 0;
    }
    // NOTE: Sabine formula (calculateRT60Sabine) and empirical C50/C80 fallbacks REMOVED
    // per TRL 7/8 requirements. FDTD simulation with Schroeder integration is the only
    // valid calculation method. Run recordRIR() before calling calculateMetrics().
    /**
     * Расчет модальной плотности (Modal Density)
     * N(f) = (4πVf²/c³) + (πSf/2c²) + (Lf/8c)
     * Упрощенная версия для низких частот
     */
    calculateModalDensity(frequency = 125) {
        if (this.totalVolume === 0)
            return 0;
        // Общая площадь поверхности
        let totalSurface = 0;
        for (const area of this.surfaceAreaByMaterial.values()) {
            totalSurface += area;
        }
        // Характерный размер (приближение)
        const L = Math.pow(this.totalVolume, 1 / 3) * 4;
        const c = this.c;
        const f = frequency;
        // Формула модальной плотности
        const modalDensity = (4 * Math.PI * this.totalVolume * f * f) / (c * c * c) +
            (Math.PI * totalSurface * f) / (2 * c * c) +
            (L * f) / (8 * c);
        return modalDensity;
    }
    /**
     * Calculate acoustic metrics - TRL 7/8 implementation
     * Primary: Schroeder Integration from FDTD simulation
     * Fallback: Sabine equation for quick estimates
     */
    async calculateMetrics(useSchroeder = true) {
        // Рассчитываем площадь поверхности если еще не сделано
        if (!this.surfaceAreaCalculated) {
            await this.calculateSurfaceArea();
        }
        let RT60;
        let T30;
        let EDT;
        let C50;
        let C80;
        let D50;
        let confidence;
        let calculationMethod;
        // Try Schroeder integration if RIR is available
        if (useSchroeder && this.rirRecorded && this.rirData && this.rirData.length > 0) {
            console.log('[AcousticSolver] Using Schroeder Integration for metrics');
            // Calculate Schroeder curve
            const schroederCurve = this.calculateSchroederCurve(this.rirData);
            // Extract reverberation times from Schroeder curve
            const rtResult = this.calculateRTFromSchroeder(schroederCurve, this.rirSampleRate, -5, -35 // T30 range
            );
            T30 = rtResult.rt60;

            // Schroeder integration MUST succeed - no fallbacks
            RT60 = T30; // T30 is more reliable than direct RT60
            EDT = rtResult.edt;
            confidence = rtResult.confidence;
            calculationMethod = 'schroeder';

            // Calculate Clarity from RIR directly - no empirical fallbacks
            C50 = this.calculateClarity(this.rirData, this.rirSampleRate, 50);
            C80 = this.calculateClarity(this.rirData, this.rirSampleRate, 80);
            D50 = this.calculateDefinition(this.rirData, this.rirSampleRate);

            // Validate clarity results
            if (!isFinite(C50) || C50 < -60) {
                const errorMsg = `[AcousticSolver] ERROR: Clarity (C50) calculation failed. ` +
                    `Got C50=${C50}. RIR data may be corrupted or simulation incomplete. ` +
                    `RIR samples: ${this.rirData.length}, sample rate: ${this.rirSampleRate}Hz`;
                console.error(errorMsg);
                throw new Error(errorMsg);
            }

            console.log(`[AcousticSolver] Results: RT60=${RT60.toFixed(2)}s, C50=${C50.toFixed(1)}dB, method=${calculationMethod}`);
        }
        else {
            // No RIR data - FDTD simulation must be run first
            const errorMsg = `[AcousticSolver] ERROR: No RIR data available. ` +
                `You must call recordRIR() before calculateMetrics(). ` +
                `The FDTD simulation must complete to generate room impulse response data.`;
            console.error(errorMsg);
            throw new Error(errorMsg);
        }
        const modalDensity = this.calculateModalDensity();
        // Spatial Impression зависит от ранних отражений
        // Упрощенная формула: выше RT60 = больше ощущение пространства
        const spatialImpression = Math.min(1.0, RT60 / 2.0);
        // Acoustic Intimacy - обратно пропорциональна объему
        const intimacyVolume = 200; // Оптимальный объем для интимности
        const acousticIntimacy = Math.max(0, Math.min(1.0, 1.0 - (this.totalVolume - intimacyVolume) / 1000));
        return {
            RT60,
            T30,
            EDT,
            C50,
            C80,
            D50,
            modalDensity,
            spatialImpression,
            acousticIntimacy,
            calculationMethod,
            confidence
        };
    }
    /**
     * Run full acoustic analysis with Schroeder integration
     * This is the TRL 8 method that should be used for validation
     */
    async runFullAnalysis(sourceX, sourceY, sourceZ, onProgress = null) {
        // Record RIR from FDTD simulation
        await this.recordRIR(sourceX, sourceY, sourceZ, onProgress);
        // Calculate metrics using Schroeder integration
        return this.calculateMetrics(true);
    }
    /**
     * Получить данные о площади поверхности по материалам
     */
    getSurfaceAreaData() {
        const result = [];
        for (const mat of Object.values(MATERIAL_ABSORPTION)) {
            const area = this.surfaceAreaByMaterial.get(mat.id) || 0;
            if (area > 0) {
                result.push({
                    material: mat.name,
                    area,
                    alpha: mat.alpha
                });
            }
        }
        return result;
    }
    /**
     * Получить объем помещения
     */
    getRoomVolume() {
        return this.totalVolume;
    }
    /**
     * Generate WGSL shaders for FDTD
     */
    generateFDTDShaders() {
        return `
            struct Uniforms {
                grid_size: vec3<u32>,
                _pad0: u32,
                dt: f32,
                c: f32,
                rho: f32,
                K: f32,
                resolution: f32,  // Physical voxel size in meters
                _pad1: f32,
                _pad2: f32,
                _pad3: f32,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> pressure_in: array<f32>;
            @group(0) @binding(2) var<storage, read_write> pressure_out: array<f32>;
            @group(0) @binding(3) var<storage, read_write> velocity: array<vec4<f32>>;
            @group(0) @binding(4) var<storage, read> voxels: array<f32>;
            
            @compute @workgroup_size(4, 4, 4)
            fn updateVelocity(@builtin(global_invocation_id) gid: vec3<u32>) {
                if (gid.x >= uniforms.grid_size.x || 
                    gid.y >= uniforms.grid_size.y || 
                    gid.z >= uniforms.grid_size.z) {
                    return;
                }
                
                let idx = gid.x + gid.y * uniforms.grid_size.x + 
                         gid.z * uniforms.grid_size.x * uniforms.grid_size.y;
                
                // Check voxel state - ONLY FLUID (state & 2 != 0) allows acoustic propagation
                // States are bitfields: EMPTY=0, SOLID=1, FLUID=2, GLASS=4, etc.
                let state = u32(voxels[idx * 8u]);
                let is_fluid = (state & 2u) != 0u;
                let is_solid = (state & 1u) != 0u;
                
                // Skip non-fluid voxels
                if (!is_fluid) { 
                    return; 
                }
                
                // Calculate pressure gradient using FORWARD difference (staggered grid)
                // This places velocity between pressure cells for proper wave propagation
                var grad_p = vec3<f32>(0.0);
                
                // Use physical voxel size in meters
                let dx = uniforms.resolution;
                let stride_y = uniforms.grid_size.x;
                let stride_z = uniforms.grid_size.x * uniforms.grid_size.y;
                
                // X gradient: forward difference p[i+1] - p[i]
                // Zero gradient if neighbor is SOLID (rigid wall normal velocity = 0)
                if (gid.x < uniforms.grid_size.x - 1u) {
                    let neighbor_state = u32(voxels[(idx + 1u) * 8u]);
                    let neighbor_is_solid = neighbor_state == 1u;
                    if (!neighbor_is_solid) {
                        let p_plus = pressure_in[idx + 1u];
                        let p_current = pressure_in[idx];
                        grad_p.x = (p_plus - p_current) / dx;
                    }
                    // else: grad_p.x stays 0 (Neumann BC: dp/dn = 0)
                }
                
                // Y gradient: forward difference
                if (gid.y < uniforms.grid_size.y - 1u) {
                    let neighbor_state = u32(voxels[(idx + stride_y) * 8u]);
                    let neighbor_is_solid = neighbor_state == 1u;
                    if (!neighbor_is_solid) {
                        let p_plus = pressure_in[idx + stride_y];
                        let p_current = pressure_in[idx];
                        grad_p.y = (p_plus - p_current) / dx;
                    }
                }
                
                // Z gradient: forward difference
                if (gid.z < uniforms.grid_size.z - 1u) {
                    let neighbor_state = u32(voxels[(idx + stride_z) * 8u]);
                    let neighbor_is_solid = neighbor_state == 1u;
                    if (!neighbor_is_solid) {
                        let p_plus = pressure_in[idx + stride_z];
                        let p_current = pressure_in[idx];
                        grad_p.z = (p_plus - p_current) / dx;
                    }
                }
                
                // Update velocity: dv/dt = -(1/rho) * grad(p)
                let v_current = velocity[idx];
                let v_new = v_current.xyz - (uniforms.dt / uniforms.rho) * grad_p;
                
                // Apply PML damping in boundary regions (absorbing outer boundary)
                var sigma = 0.0;
                let pml_width = 10u;
                if (gid.x < pml_width || gid.x > uniforms.grid_size.x - pml_width ||
                    gid.y < pml_width || gid.y > uniforms.grid_size.y - pml_width ||
                    gid.z < pml_width || gid.z > uniforms.grid_size.z - pml_width) {
                    // Damping coefficient: 10 gives exp(-10*0.0005) = 0.995 per step (0.5% loss)
                    // Previous value 100 was too aggressive (4.3% loss per step)
                    sigma = 10.0;
                }
                
                velocity[idx] = vec4<f32>(v_new * exp(-sigma * uniforms.dt), 0.0);
            }
            
            @compute @workgroup_size(4, 4, 4)
            fn updatePressure(@builtin(global_invocation_id) gid: vec3<u32>) {
                if (gid.x >= uniforms.grid_size.x || 
                    gid.y >= uniforms.grid_size.y || 
                    gid.z >= uniforms.grid_size.z) {
                    return;
                }
                
                let idx = gid.x + gid.y * uniforms.grid_size.x + 
                         gid.z * uniforms.grid_size.x * uniforms.grid_size.y;
                
                // Check voxel state using bitfields
                let state = u32(voxels[idx * 8u]);
                let is_fluid = (state & 2u) != 0u;
                let is_solid_only = state == 1u; // Pure solid (walls)
                
                // RIGID WALL BOUNDARY: Use Neumann BC (dp/dn = 0)
                // For rigid walls, normal velocity = 0 but pressure reflects
                // We implement this by copying pressure from neighboring fluid cell
                if (is_solid_only) {
                    // Find a neighboring fluid cell and copy its pressure (reflection)
                    var neighbor_p = 0.0;
                    var found = false;
                    
                    // Check all 6 neighbors for a fluid cell
                    let stride_y = uniforms.grid_size.x;
                    let stride_z = uniforms.grid_size.x * uniforms.grid_size.y;
                    
                    if (!found && gid.x > 0u) {
                        let n_state = u32(voxels[(idx - 1u) * 8u]);
                        if ((n_state & 2u) != 0u) { neighbor_p = pressure_in[idx - 1u]; found = true; }
                    }
                    if (!found && gid.x < uniforms.grid_size.x - 1u) {
                        let n_state = u32(voxels[(idx + 1u) * 8u]);
                        if ((n_state & 2u) != 0u) { neighbor_p = pressure_in[idx + 1u]; found = true; }
                    }
                    if (!found && gid.y > 0u) {
                        let n_state = u32(voxels[(idx - stride_y) * 8u]);
                        if ((n_state & 2u) != 0u) { neighbor_p = pressure_in[idx - stride_y]; found = true; }
                    }
                    if (!found && gid.y < uniforms.grid_size.y - 1u) {
                        let n_state = u32(voxels[(idx + stride_y) * 8u]);
                        if ((n_state & 2u) != 0u) { neighbor_p = pressure_in[idx + stride_y]; found = true; }
                    }
                    if (!found && gid.z > 0u) {
                        let n_state = u32(voxels[(idx - stride_z) * 8u]);
                        if ((n_state & 2u) != 0u) { neighbor_p = pressure_in[idx - stride_z]; found = true; }
                    }
                    if (!found && gid.z < uniforms.grid_size.z - 1u) {
                        let n_state = u32(voxels[(idx + stride_z) * 8u]);
                        if ((n_state & 2u) != 0u) { neighbor_p = pressure_in[idx + stride_z]; found = true; }
                    }
                    
                    // Apply absorption coefficient (concrete alpha ~0.02 = 98% reflection)
                    let reflection_coeff = 0.98; // 1.0 = perfect reflection, 0.0 = full absorption
                    pressure_out[idx] = neighbor_p * reflection_coeff;
                    return;
                }
                
                // Skip EMPTY voxels (state=0) - they are outside the room
                if (state == 0u) {
                    pressure_out[idx] = 0.0;
                    return;
                }
                
                // Process only FLUID voxels
                if (!is_fluid) {
                    return;
                }
                
                // Calculate velocity divergence using BACKWARD difference (staggered grid)
                // This pairs with forward difference in velocity update for wave propagation
                var div_v = 0.0;
                let dx = uniforms.resolution;
                
                // X divergence: backward difference v[i] - v[i-1]
                if (gid.x > 0u) {
                    let v_current = velocity[idx].x;
                    let v_minus = velocity[idx - 1u].x;
                    div_v += (v_current - v_minus) / dx;
                }
                
                // Y divergence: backward difference
                if (gid.y > 0u) {
                    let v_current = velocity[idx].y;
                    let v_minus = velocity[idx - uniforms.grid_size.x].y;
                    div_v += (v_current - v_minus) / dx;
                }
                
                // Z divergence: backward difference
                if (gid.z > 0u) {
                    let stride_z = uniforms.grid_size.x * uniforms.grid_size.y;
                    let v_current = velocity[idx].z;
                    let v_minus = velocity[idx - stride_z].z;
                    div_v += (v_current - v_minus) / dx;
                }
                
                // Update pressure: dp/dt = -K * div(v)
                let p_current = pressure_in[idx];
                pressure_out[idx] = p_current - uniforms.K * uniforms.dt * div_v;
            }
            
            @compute @workgroup_size(1)
            fn analyzeAcoustics(@builtin(global_invocation_id) gid: vec3<u32>) {
                // Calculate RT60, clarity indices from impulse response
                // This would integrate the energy decay curve
                // Simplified placeholder for now
            }
        `;
    }
    /**
     * Cleanup
     */
    destroy() {
        this.pressureBufferA?.destroy();
        this.pressureBufferB?.destroy();
        this.velocityBuffer?.destroy();
        this.voxelStateData = null;
        this.voxelStateBuffer = null;
        console.log('[AcousticSolver] Destroyed');
    }
}
