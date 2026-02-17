/**
 * AHI 2.0 Ultimate - Space Syntax Solver
 *
 * Hybrid Approach:
 * 1. Graph Connectivity: Exact N^2 Pairwise Line-of-Sight (Truthful Connections)
 * 2. Isovist Metrics: Stochastic Raycasting (Volume/Distance estimation)
 */
export class SpaceSyntaxSolver {
    device;
    gridConfig;

    // GPU Resources
    bindGroupLayout;
    pairwisePipeline;   // New: For N^2 Graph Generation
    metricsPipeline;    // Renamed: For Isovist Volume/Dist

    visibilityBuffer;   // The Graph (Adjacency Matrix as Bitmask)
    metricsBuffer;      // Isovist Properties
    uniformBuffer;
    samplePosBuffer;    // New: Stores exact coordinates of samples [x, y, z, pad]

    // Configuration
    config = {
        numRays: 256,        // Rays per viewpoint for Isovist Metrics
        maxSamples: 1000,    // Hard cap on VGA nodes
        useFixedPoint: true
    };

    constructor(device, gridConfig, config = {}) {
        this.device = device;
        this.gridConfig = gridConfig;
        this.config = { ...this.config, ...config };
    }

    async initialize() {
        // Shader Code
        const shaderModule = this.device.createShaderModule({
            label: 'Space Syntax Shaders',
            code: this.generateSpaceSyntaxShaders()
        });

        // Bind Group Layout
        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'Space Syntax Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // Voxels
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // Visibility Graph (Bits)
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // Metrics
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }  // Sample Positions (vec4)
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'Space Syntax Pipeline Layout',
            bindGroupLayouts: [this.bindGroupLayout]
        });

        // 1. Pairwise LOS Pipeline (Graph Generation)
        this.pairwisePipeline = this.device.createComputePipeline({
            label: 'Pairwise LOS Pipeline',
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_graph_los' }
        });

        // 2. Metrics Pipeline (Isovist Volume)
        this.metricsPipeline = this.device.createComputePipeline({
            label: 'Metrics Pipeline',
            layout: pipelineLayout,
            compute: { module: shaderModule, entryPoint: 'compute_isovist_metrics' }
        });

        // Allocate Buffers
        // Visibility Graph: 1000x1000 bits
        const visBufferSize = Math.ceil((this.config.maxSamples * this.config.maxSamples) / 32) * 4;
        this.visibilityBuffer = this.device.createBuffer({
            label: 'Visibility Buffer',
            size: Math.max(256, visBufferSize),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Metrics: [Volume, Distance] per sample
        this.metricsBuffer = this.device.createBuffer({
            label: 'Metrics Buffer',
            size: this.config.maxSamples * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        // Sample Positions: 1000 * vec4<f32> (16 bytes)
        this.samplePosBuffer = this.device.createBuffer({
            label: 'Sample Positions Buffer',
            size: this.config.maxSamples * 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        // Uniforms
        const uniformSize = 64;
        this.uniformBuffer = this.device.createBuffer({
            label: 'Uniform Buffer',
            size: uniformSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Init Uniforms
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const gridDiagonal = Math.sqrt(nx * nx + ny * ny + nz * nz);

        const uniformData = new ArrayBuffer(uniformSize);
        const view = new DataView(uniformData);
        view.setUint32(0, nx, true);
        view.setUint32(4, ny, true);
        view.setUint32(8, nz, true);
        view.setFloat32(32, gridDiagonal, true);
        view.setUint32(36, this.config.maxSamples, true);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        console.log(`[SpaceSyntaxSolver] Initialized. Samples: ${this.config.maxSamples} (Pairwise)`);
    }

    async analyze(voxelBuffer) {
        if (!this.visibilityBuffer) return null;
        this._voxelBuffer = voxelBuffer;

        // 1. Sample Fluid Voxels & Upload Positions
        const samples = await this.sampleFluidVoxels(voxelBuffer);
        const sampleCount = samples.length;  // May be less than maxSamples

        if (sampleCount === 0) return { globalIntegration: 0, avgConnectivity: 0, visibilityIndex: 0 };

        // Upload Sample positions to GPU (vec4 array)
        const posData = new Float32Array(this.config.maxSamples * 4); // x, y, z, pad
        for (let i = 0; i < sampleCount; i++) {
            posData[i * 4 + 0] = samples[i].x;
            posData[i * 4 + 1] = samples[i].y;
            posData[i * 4 + 2] = samples[i].z;
            posData[i * 4 + 3] = 0; // padding
        }
        this.device.queue.writeBuffer(this.samplePosBuffer, 0, posData);
        this.device.queue.writeBuffer(this.uniformBuffer, 36, new Uint32Array([sampleCount])); // Update count

        // 2. Clear Result Buffers
        this.device.queue.writeBuffer(this.visibilityBuffer, 0, new Uint32Array(this.visibilityBuffer.size / 4));
        this.device.queue.writeBuffer(this.metricsBuffer, 0, new Uint32Array(this.metricsBuffer.size / 4));

        // 3. Create Bind Group
        const bindGroup = this.device.createBindGroup({
            label: 'Space Syntax Bind Group',
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: voxelBuffer } },
                { binding: 2, resource: { buffer: this.visibilityBuffer } },
                { binding: 3, resource: { buffer: this.metricsBuffer } },
                { binding: 4, resource: { buffer: this.samplePosBuffer } }
            ]
        });

        // 4. Dispatch Compute Passes
        const commandEncoder = this.device.createCommandEncoder();

        // Pass A: Pairwise LOS Graph Generation (N x N)
        // Workgroup size 16x16.
        const workgroups = Math.ceil(sampleCount / 16);

        const passA = commandEncoder.beginComputePass({ label: 'Pairwise Graph Pass' });
        passA.setPipeline(this.pairwisePipeline);
        passA.setBindGroup(0, bindGroup);
        passA.dispatchWorkgroups(workgroups, workgroups, 1);
        passA.end();

        // Pass B: Isovist Metrics (Volume/Dist)
        // One workgroup per sample (same as before) for 256 rays
        const passB = commandEncoder.beginComputePass({ label: 'Metrics Raycast Pass' });
        passB.setPipeline(this.metricsPipeline);
        passB.setBindGroup(0, bindGroup);
        passB.dispatchWorkgroups(sampleCount, 1, 1); // 1 workgroup (256 threads) per sample
        passB.end();

        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        // 5. Read Back & Analyze Results
        const isovistMetrics = await this.readMetricsBuffer(0, sampleCount * 2);
        const connectivityCounts = await this.calculateGraphConnectivity(sampleCount); // Returns degrees
        const integrationData = await this.calculateIntegrationCPU(connectivityCounts, sampleCount);

        // Averages
        const avgConn = connectivityCounts.reduce((a, b) => a + b, 0) / sampleCount;
        const avgInt = integrationData.reduce((a, b) => a + b, 0) / sampleCount;

        // Connectivity % (Normalized by max possible connections in graph)
        const maxNodes = Math.max(1, sampleCount - 1);
        const normConn = (avgConn / maxNodes) * 100;

        // Visibility Index (Volume)
        let totalVolume = 0;
        for (let i = 0; i < sampleCount; i++) totalVolume += isovistMetrics[i * 2];
        const avgVolume = totalVolume / sampleCount;
        const theoreticalMax = this.config.numRays * (Math.sqrt(this.gridConfig.totalVoxels) / 2);
        const visIndex = Math.min(1.0, avgVolume / Math.max(1, theoreticalMax));

        console.log(`[SpaceSyntaxSolver] Analysis Complete (Hybrid). Samples: ${sampleCount}`);
        console.log(`  - Graph Connectivity (LOS): ${avgConn.toFixed(1)} / ${maxNodes} (${normConn.toFixed(1)}%)`);
        console.log(`  - Integration (HH): ${avgInt.toFixed(4)}`);
        console.log(`  - Isovist Volume Idx: ${visIndex.toFixed(3)}`);

        return {
            globalIntegration: avgInt,
            avgConnectivity: normConn,
            rawConnectivity: avgConn,
            isovistArea: avgVolume,
            visibilityIndex: visIndex,
            sampleCount: sampleCount,
            intelligibility: this.calculateIntelligibility(connectivityCounts, integrationData),
            // Synergy: Optional - Calculate local vs global
            synergy: this.calculateSynergy(integrationData, connectivityCounts, sampleCount)
        };
    }

    // --- Helpers ---

    async sampleFluidVoxels(voxelBuffer) {
        // Read buffer to CPU
        const staging = this.device.createBuffer({
            size: voxelBuffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(voxelBuffer, 0, staging, 0, voxelBuffer.size);
        this.device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange());

        const { nx, ny, nz } = this.gridConfig.dimensions;
        const positions = [];

        // Find fluid voxels (stride 8, offset 0 is state)
        for (let i = 0; i < data.length; i += 8) {
            if (data[i] > 1.5 && data[i] < 2.5) {
                const idx = i / 8;
                positions.push({
                    x: idx % nx,
                    y: Math.floor(idx / nx) % ny,
                    z: Math.floor(idx / (nx * ny))
                });
            }
        }
        staging.unmap(); staging.destroy();

        // Shuffle & Cap
        for (let i = positions.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [positions[i], positions[j]] = [positions[j], positions[i]];
        }
        return positions.slice(0, this.config.maxSamples);
    }

    async calculateGraphConnectivity(numSamples) {
        const bitCount = Math.ceil((numSamples * numSamples) / 32);
        // Align read size to 4
        const readSize = bitCount * 4;

        const staging = this.device.createBuffer({
            size: readSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.visibilityBuffer, 0, staging, 0, staging.size); // Use staging.size to be safe
        this.device.queue.submit([enc.finish()]);

        // Wait for mapping
        await staging.mapAsync(GPUMapMode.READ);
        const bits = new Uint32Array(staging.getMappedRange());

        // Extract graph degrees
        const degrees = new Float32Array(numSamples);
        for (let i = 0; i < numSamples; i++) {
            let count = 0;
            for (let k = 0; k < numSamples; k++) {
                if (i === k) continue;
                const idx = i * numSamples + k;
                if ((bits[Math.floor(idx / 32)] >>> (idx % 32)) & 1) count++;
            }
            degrees[i] = count;
        }
        staging.unmap(); staging.destroy();
        return degrees;
    }

    async calculateIntegrationCPU(degrees, numSamples) {
        // Reread bits for BFS (Optim: could reuse buffer from above but simplicity first)
        const bitCount = Math.ceil((numSamples * numSamples) / 32);
        const staging = this.device.createBuffer({
            size: bitCount * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.visibilityBuffer, 0, staging, 0, staging.size);
        this.device.queue.submit([enc.finish()]);

        await staging.mapAsync(GPUMapMode.READ);
        const bits = new Uint32Array(staging.getMappedRange());

        // Build Adjacency List
        const adj = new Array(numSamples).fill(0).map(() => []);
        for (let i = 0; i < numSamples; i++) {
            for (let k = 0; k < numSamples; k++) {
                if (i === k) continue;
                const idx = i * numSamples + k;
                if ((bits[Math.floor(idx / 32)] >>> (idx % 32)) & 1) adj[i].push(k);
            }
        }
        staging.unmap(); staging.destroy();

        // Calculate Mean Depth -> Integration
        const integration = new Float32Array(numSamples);
        for (let i = 0; i < numSamples; i++) {
            const dists = new Int32Array(numSamples).fill(-1);
            const q = [i]; dists[i] = 0;
            let totalDist = 0; let reachable = 0;
            let head = 0;

            while (head < q.length) {
                const u = q[head++];
                for (const v of adj[u]) {
                    if (dists[v] === -1) {
                        dists[v] = dists[u] + 1;
                        totalDist += dists[v];
                        reachable++;
                        q.push(v);
                    }
                }
            }
            // Integration (Inverse Mean Depth) relative to system
            if (reachable > 1) {
                const md = totalDist / reachable;
                integration[i] = md > 1 ? 1.0 / (md - 1.0) : 1.0;
            }
        }
        return integration;
    }

    async readMetricsBuffer(offset, count) {
        const size = count * 4;
        const staging = this.device.createBuffer({
            size: size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.metricsBuffer, offset * 4, staging, 0, size);
        this.device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap(); staging.destroy();
        return data;
    }

    calculateIntelligibility(conn, int) {
        let n = conn.length; if (n < 2) return 0;
        let sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
        for (let i = 0; i < n; i++) {
            let x = conn[i], y = int[i];
            sx += x; sy += y; sxy += x * y; sx2 += x * x; sy2 += y * y;
        }
        let num = (n * sxy - sx * sy);
        let den = Math.sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy));
        if (den === 0) return 0;
        return Math.pow(num / den, 2);
    }

    calculateSynergy(integration, connectivity, n) {
        // Synergy currently placeholder (requires local integration)
        return 0;
    }

    generateSpaceSyntaxShaders() {
        return `
            struct Uniforms {
                grid_size: vec3<u32>,
                _pad0: u32,
                _pad1: vec4<f32>,
                max_distance: f32,
                sample_count: u32,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            @group(0) @binding(1) var<storage, read> voxels: array<f32>;
            @group(0) @binding(2) var<storage, read_write> visibility: array<atomic<u32>>;
            @group(0) @binding(3) var<storage, read_write> metrics: array<f32>;
            @group(0) @binding(4) var<storage, read> sample_pos: array<vec4<f32>>; 

            // --- Pass 1: Pairwise Graph Generation (N^2) ---
            @compute @workgroup_size(16, 16)
            fn compute_graph_los(@builtin(global_invocation_id) gid: vec3<u32>) {
                let from_idx = gid.x;
                let to_idx = gid.y;
                let count = uniforms.sample_count;
                
                if (from_idx >= count || to_idx >= count || from_idx == to_idx) { return; }
                
                let start_pos = sample_pos[from_idx].xyz;
                let end_pos = sample_pos[to_idx].xyz;
                
                // Raymarch from start to end
                let diff = end_pos - start_pos;
                let dist = length(diff);
                let dir = normalize(diff); // Can be NaN if start==end (covered by check above)
                
                let step_size = 0.5; // Fine step for LOS
                var t = 1.0; 
                var visible = true;
                
                // Safety clamp
                if (dist < 1.0) { visible = true; } 
                else {
                    let max_t = dist - 0.5; // Don't march into the target voxel itself
                    
                    let sx = i32(uniforms.grid_size.x);
                    let sy = i32(uniforms.grid_size.y);
                    let sz = i32(uniforms.grid_size.z);
                    let slayer = sx * sy;
                    
                    loop {
                        if (t >= max_t) { break; }
                        let p = start_pos + dir * t;
                        
                        let ix = i32(round(p.x));
                        let iy = i32(round(p.y));
                        let iz = i32(round(p.z));
                        
                        // Check bounds
                        if (ix >= 0 && ix < sx && iy >= 0 && iy < sy && iz >= 0 && iz < sz) {
                            let idx = ix + iy * sx + iz * slayer;
                            let state = u32(voxels[u32(idx) * 8u]);
                            if ((state & 1u) != 0u) { // Solid?
                                visible = false;
                                break;
                            }
                        }
                        t += step_size;
                    }
                }
                
                // If visible, set the bit in adjacency matrix
                if (visible) {
                    let bit_idx = from_idx * count + to_idx;
                    atomicOr(&visibility[bit_idx / 32u], 1u << (bit_idx % 32u));
                }
            }

            // --- Pass 2: Isovist Metrics (Random Rays) ---
            var<workgroup> wg_vol: atomic<u32>;
            var<workgroup> wg_dist: atomic<u32>;

            @compute @workgroup_size(256)
            fn compute_isovist_metrics(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
                let sample_idx = wid.x; // One workgroup per sample
                let ray_id = lid.x;
                
                if (sample_idx >= uniforms.sample_count) { return; }

                if (ray_id == 0u) {
                    atomicStore(&wg_vol, 0u);
                    atomicStore(&wg_dist, 0u);
                }
                workgroupBarrier();

                let origin = sample_pos[sample_idx].xyz;
                
                // Ray Gen (Fibonacci Sphere / Uniform)
                let theta = f32(ray_id % 16u) * 6.28318 / 16.0;
                let phi = f32(ray_id / 16u) * 3.14159 / 16.0;
                let dir = vec3<f32>(sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta));
                
                var t = 1.0;
                let step = 0.8;
                var dist_accum = 0.0;
                var vol_accum = 0u;
                
                let sx = i32(uniforms.grid_size.x);
                let sy = i32(uniforms.grid_size.y);
                let sz = i32(uniforms.grid_size.z);
                let slayer = sx * sy;
                let max_d = uniforms.max_distance;

                loop {
                    if (t >= max_d) { break; }
                    let p = origin + dir * t;
                    let ix = i32(round(p.x));
                    let iy = i32(round(p.y));
                    let iz = i32(round(p.z));
                    
                    if (ix < 0 || ix >= sx || iy < 0 || iy >= sy || iz < 0 || iz >= sz) {
                        t += step; continue;
                    }
                    
                    let idx = ix + iy * sx + iz * slayer;
                    let state = u32(voxels[u32(idx) * 8u]);
                    
                    if ((state & 1u) != 0u) { break; } // Hit Solid
                    if ((state & 2u) != 0u) { // Hit Fluid
                        vol_accum++;
                        dist_accum += t;
                    }
                    t += step;
                }
                
                atomicAdd(&wg_vol, vol_accum);
                atomicAdd(&wg_dist, u32(dist_accum * 100.0));
                
                workgroupBarrier();
                
                if (ray_id == 0u) {
                    let total_vol = f32(atomicLoad(&wg_vol));
                    let total_dist = f32(atomicLoad(&wg_dist)) / 100.0;
                    metrics[sample_idx * 2u] = total_vol;
                    metrics[sample_idx * 2u + 1u] = total_dist;
                }
            }
        `;
    }

    destroy() {
        this.visibilityBuffer?.destroy();
        this.metricsBuffer?.destroy();
        this.samplePosBuffer?.destroy();
        this.uniformBuffer?.destroy();
        console.log('[SpaceSyntaxSolver] Destroyed');
    }
}
