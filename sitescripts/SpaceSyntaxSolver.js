/**
 * Polymère — Space Syntax Solver (User-Driven Observer Placement)
 *
 * Interactive Eye-Level Isovist Analysis (DepthmapX-style VGA).
 * The user places observer points on the IFC model; the solver runs
 * the GPU analysis pipeline only on those user-defined locations.
 *
 * ════════════════════════════════════════════════════════════════════════
 *  GPU PIPELINE (4 passes in command-encoder order):
 *
 *  Pass 0 — compute_headings  [NEW]
 *    • 1 thread per observer, workgroup 64×1×1
 *    • Casts 16 radial DDA rays in XZ plane, writes longest-LOS heading
 *      directly to headingBuffer (eliminates CPU _findLongestLineOfSight)
 *    • Bindings (group 0): uniforms, voxels, observers, headings (r/w)
 *
 *  Pass A — compute_isovist   [UPGRADED to 3D Spherical]
 *    • 1 thread per observer, workgroup 64×1×1
 *    • Casts 128 rays distributed over a full unit sphere via Fibonacci
 *      lattice; computes 3D Volume (voxel³) and Surface Area (voxel²)
 *    • Compactness = (36π × V²) / S³  clamped to [0,1]
 *    • Bindings (group 0): uniforms, voxels, observers, headings (r), output_metrics (r/w)
 *    • NOTE: headings is now pass-through read — Pass 0 already filled it.
 *
 *  Pass B — compute_adjacency [UNCHANGED]
 *    • 1 thread per (A,B) pair, workgroup 16×16×1
 *    • Amanatides & Woo 3D DDA with horizontal FOV cone culling
 *    • Bindings (group 0): uniforms, voxels, observers, headings (r), adjacency (r/w atomic)
 *
 *  Pass C — GPU BFS [MOCKED SCAFFOLD]
 *    • WGSL frontier-BFS shader compiled and pipeline created
 *    • JS dispatch loop stubbed — CPU BFS (_calculateGraphMetrics) is
 *      retained as functional fallback until multi-pass indirect dispatch
 *      is validated.
 *
 * ════════════════════════════════════════════════════════════════════════
 *  Metrics buffer output schema (CHANGED from 2D):
 *    [i*4+0] volume       (voxel³)
 *    [i*4+1] surface_area (voxel²)
 *    [i*4+2] compactness  (dimensionless, [0,1])
 *    [i*4+3] 0.0          (padding)
 *
 *  Voxel stride: 8 floats [state, material, pad, temp, vx, vy, vz, pad]
 *  States: 0 = EMPTY, 1 = SOLID (wall/floor), 2 = FLUID (indoor air)
 */
export class SpaceSyntaxSolver {
    device;
    gridConfig;

    // ── GPU Resources — Pass 0 (Heading Generation) ──────────────────────
    headingsPipeline;
    headingsBindGroupLayout;

    // ── GPU Resources — Pass A (3D Isovist) ──────────────────────────────
    isovistPipeline;
    isovistBindGroupLayout;
    outputMetricsBuffer;

    // ── GPU Resources — Pass B (Adjacency / Visibility Graph) ────────────
    adjacencyPipeline;
    adjacencyBindGroupLayout;
    adjacencyBuffer;

    // ── GPU Resources — Pass C (BFS Scaffold — mocked) ───────────────────
    bfsPipeline;
    bfsBindGroupLayout;
    bfsIntegrationBuffer;   // output: per-node integration (f32 array)
    bfsFrontierBuffer;      // working: frontier bit-mask (u32 array)
    bfsDistanceBuffer;      // working: per-node BFS depth (i32 array)

    // ── Shared Dynamic Buffers ────────────────────────────────────────────
    uniformBuffer;
    observerBuffer;
    headingBuffer; // written by Pass 0, read by Pass A & B

    // ── Configuration ─────────────────────────────────────────────────────
    config = {
        eyeLevelMeters:    1.5,   // Eye-level offset above floor (metres)
        observerStepMeters: 1.0,  // Observer grid spacing (metres), unused in user-placement mode
        maxRayDistMeters:  50.0,  // Maximum isovist ray distance (metres)
    };

    constructor(device, gridConfig, config = {}) {
        this.device     = device;
        this.gridConfig = gridConfig;
        this.config     = { ...this.config, ...config };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — initialize()
    //  Must be called once before analyze(). Creates all pipelines.
    // ═══════════════════════════════════════════════════════════════════════

    async initialize() {
        // ── Load the shared WGSL shader (avga_raycaster.wgsl) ──
        const shaderCode = await this._loadShaderCode();

        // Validate shader compilation immediately to surface WGSL errors early.
        this.device.pushErrorScope('validation');
        const sharedModule = this.device.createShaderModule({
            label: 'AVGA Shared Compute Shader',
            code: shaderCode
        });
        const shaderError = await this.device.popErrorScope();
        if (shaderError) {
            console.error('[SpaceSyntaxSolver] WGSL shader compilation error:', shaderError.message);
            throw new Error('WGSL shader compilation failed: ' + shaderError.message);
        }
        console.log('[SpaceSyntaxSolver] Shared WGSL shader compiled successfully.');

        // ── Shared Uniform Buffer (48 bytes, 16-byte aligned) ──
        // Layout:
        //   u32 grid_size_x    (offset  0)
        //   u32 grid_size_y    (offset  4)
        //   u32 grid_size_z    (offset  8)
        //   u32 observer_count (offset 12)
        //   f32 resolution     (offset 16)  ← metres per voxel
        //   f32 max_ray_dist   (offset 20)  ← in voxel units
        //   f32 ray_step       (offset 24)  ← in voxel units (0.5 = half-voxel)
        //   f32 fov_horizontal (offset 28)  ← degrees (default 120)
        //   f32 fov_vertical   (offset 32)  ← degrees (default 90)
        //   u32 _pad3–_pad5    (offset 36–44)
        this.uniformBuffer = this.device.createBuffer({
            label: 'SpaceSyntax Uniform Buffer',
            size:  48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // ════════════════════════════════════════════════════════════════
        //  SHARED 6-SLOT BIND GROUP LAYOUT
        //
        //  The WGSL shader (avga_raycaster.wgsl) declares all 6 binding
        //  slots globally so that all three entry points compile from a
        //  single module with a consistent reflection layout.
        //
        //  All three passes are built from the same layout.
        //  Each pass supplies all 6 buffers; unused slots for a given
        //  entry-point receive the small stub_buffer (16 bytes, read-only).
        //
        //  Slot assignments:
        //    0 — uniforms         (uniform)
        //    1 — voxels           (read-only-storage)
        //    2 — observers        (read-only-storage)
        //    3 — headings         (storage r/w)      — Pass 0 writes, A/B read
        //    4 — output_metrics   (storage r/w)      — Pass A writes, 0/B unused
        //    5 — adjacency        (storage r/w)      — Pass B writes, 0/A unused
        // ════════════════════════════════════════════════════════════════
        this.sharedBindGroupLayout = this.device.createBindGroupLayout({
            label: 'Shared 6-Slot Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // headings (r/w)
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // output_metrics (r/w)
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // adjacency (r/w atomic)
            ]
        });

        const sharedPipelineLayout = this.device.createPipelineLayout({
            label: 'Shared Pipeline Layout',
            bindGroupLayouts: [this.sharedBindGroupLayout]
        });

        // ── Pass 0 — Headings pipeline ────────────────────────────────────
        this.headingsPipeline = this.device.createComputePipeline({
            label: 'Headings Compute Pipeline (Pass 0)',
            layout: sharedPipelineLayout,
            compute: { module: sharedModule, entryPoint: 'compute_headings' }
        });

        // ── Pass A — 3D Spherical Isovist pipeline ────────────────────────
        this.isovistPipeline = this.device.createComputePipeline({
            label: 'Isovist Compute Pipeline (Pass A — 3D Spherical)',
            layout: sharedPipelineLayout,
            compute: { module: sharedModule, entryPoint: 'compute_isovist' }
        });

        // ── Pass B — Adjacency pipeline (same module, same layout) ────────
        this.adjacencyPipeline = this.device.createComputePipeline({
            label: 'Adjacency Compute Pipeline (Pass B)',
            layout: sharedPipelineLayout,
            compute: { module: sharedModule, entryPoint: 'compute_adjacency' }
        });

        // ── Two distinct stub buffers for unused writable storage slots ─────
        // WebGPU forbids aliasing: the same buffer cannot appear in two
        // different writable storage bindings within a single bind group.
        // _stubBuffer4 fills slot 4 (output_metrics) when unused by a pass.
        // _stubBuffer5 fills slot 5 (adjacency)       when unused by a pass.
        this._stubBuffer4 = this.device.createBuffer({
            label: 'Stub Buffer — Slot 4 (output_metrics)',
            size:  16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._stubBuffer5 = this.device.createBuffer({
            label: 'Stub Buffer — Slot 5 (adjacency)',
            size:  16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // ════════════════════════════════════════════════════════════════
        //  PASS C — GPU BFS Scaffold (strongly mocked)
        //
        //  The WGSL BFS shader is generated and compiled here.
        //  Pipeline and buffer layouts are fully created and ready.
        //  The actual multi-pass frontier dispatch loop is stubbed — the
        //  CPU BFS (_calculateGraphMetrics) remains the functional path.
        //
        //  TODO: replace CPU BFS by iterating bfsPipeline dispatches
        //        once WebGPU indirect dispatch + readback signalling is
        //        available in this environment.
        // ════════════════════════════════════════════════════════════════
        const bfsShaderCode = this._generateBFSShader();
        const bfsModule = this.device.createShaderModule({
            label: 'GPU BFS Compute Shader (Pass C — mock)',
            code: bfsShaderCode
        });

        // Bind group layout for BFS:
        //   0: uniforms (u32 observer_count, u32 current_depth, u32 _pad[2])
        //   1: adjacency    — bit-packed (read-only)
        //   2: frontier     — u32 array, bit-packed (read_write)
        //   3: next_frontier— u32 array, bit-packed (read_write)
        //   4: distances    — i32 array (read_write); -1 = unvisited
        //   5: integration  — f32 array output (read_write)
        this.bfsBindGroupLayout = this.device.createBindGroupLayout({
            label: 'BFS Bind Group Layout (Pass C)',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // adjacency
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // frontier
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // next_frontier
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // distances
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // integration output
            ]
        });

        this.bfsPipeline = this.device.createComputePipeline({
            label: 'BFS Compute Pipeline (Pass C — mock)',
            layout: this.device.createPipelineLayout({
                label: 'BFS Pipeline Layout',
                bindGroupLayouts: [this.bfsBindGroupLayout]
            }),
            compute: { module: bfsModule, entryPoint: 'compute_bfs_frontier' }
        });

        console.log('[SpaceSyntaxSolver] Initialized. Passes: 0 (Headings) + A (3D Isovist) + B (Adjacency) + C (BFS mock).');
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — analyze(voxelBuffer, options)
    //
    //  Runs the full 4-pass GPU pipeline against user-placed observers.
    //  Returns a result object with 3D isovist metrics + graph metrics.
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * @param {GPUBuffer} voxelBuffer  — GPU buffer with voxel state data (stride 8 f32).
     * @param {Object}    options      — { observers: Array<{x,y,z}> } in WORLD coordinates.
     * @returns {Object} Aggregated metrics or a pending stub.
     */
    async analyze(voxelBuffer, options = {}) {
        if (!this.isovistPipeline) return null;

        // ── Stub path: no observers yet ──
        const userObservers = options.observers || [];
        if (userObservers.length === 0) {
            console.log('[SpaceSyntaxSolver] No observers — returning pending stub.');
            return this._pendingResult();
        }

        const { nx, ny, nz } = this.gridConfig.dimensions;
        const resolution = this.gridConfig.resolution; // metres per voxel

        // ─────────────────────────────────────────────────────────────────
        // STEP 1 — Convert world → voxel coordinates (CPU, trivial)
        // ─────────────────────────────────────────────────────────────────
        const observerPoints = userObservers.map(wp => this._worldToVoxel(wp));
        const N = observerPoints.length;

        console.log(`[SpaceSyntaxSolver] Dispatching pipeline for ${N} observer(s)...`);

        // ─────────────────────────────────────────────────────────────────
        // STEP 2 — Upload uniforms & observer positions to GPU
        //          (no CPU voxel readback needed — Pass 0 handles headings)
        // ─────────────────────────────────────────────────────────────────

        // Observer buffer: vec4<f32> per point
        const observerData = new Float32Array(N * 4);
        for (let i = 0; i < N; i++) {
            observerData[i * 4 + 0] = observerPoints[i].x;
            observerData[i * 4 + 1] = observerPoints[i].y;
            observerData[i * 4 + 2] = observerPoints[i].z;
            observerData[i * 4 + 3] = 0.0;
        }

        // Destroy previous dynamic buffers (observer count may change)
        this.observerBuffer?.destroy();
        this.outputMetricsBuffer?.destroy();
        this.adjacencyBuffer?.destroy();
        this.headingBuffer?.destroy();

        this.observerBuffer = this.device.createBuffer({
            label: 'Observer Positions Buffer',
            size:  observerData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.observerBuffer, 0, observerData);

        // Heading buffer (GPU-only read_write — Pass 0 populates it)
        this.headingBuffer = this.device.createBuffer({
            label: 'Headings Buffer (GPU-written by Pass 0)',
            size:  Math.max(16, N * 4 * 4), // vec4<f32> per observer, min 16 bytes
            usage: GPUBufferUsage.STORAGE,   // no COPY_DST needed — GPU writes it
        });

        // Output metrics buffer: 4 f32 per observer [volume, surface, compact, 0]
        this.outputMetricsBuffer = this.device.createBuffer({
            label: 'Isovist Output Metrics Buffer',
            size:  Math.max(16, N * 4 * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        // Zero-fill to avoid stale data
        this.device.queue.writeBuffer(this.outputMetricsBuffer, 0, new Float32Array(N * 4));

        // Adjacency matrix: bit-packed atomic<u32>
        const totalPairs = N * N;
        const u32Count   = Math.ceil(totalPairs / 32);
        this.adjacencyBuffer = this.device.createBuffer({
            label: 'Adjacency Matrix Buffer',
            size:  Math.max(4, u32Count * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.adjacencyBuffer, 0, new Uint32Array(Math.max(1, u32Count)));

        // Write uniforms
        const maxRayDistVoxels = this.config.maxRayDistMeters / resolution;
        const rayStepVoxels    = 0.5; // half-voxel precision

        const uniformData = new ArrayBuffer(48);
        const uv          = new DataView(uniformData);
        uv.setUint32(  0, nx,               true); // grid_size_x
        uv.setUint32(  4, ny,               true); // grid_size_y
        uv.setUint32(  8, nz,               true); // grid_size_z
        uv.setUint32( 12, N,                true); // observer_count
        uv.setFloat32(16, resolution,       true); // resolution (m/voxel)
        uv.setFloat32(20, maxRayDistVoxels, true); // max_ray_dist (voxels)
        uv.setFloat32(24, rayStepVoxels,    true); // ray_step (voxels)
        uv.setFloat32(28, 120.0,            true); // fov_horizontal (degrees)
        uv.setFloat32(32, 90.0,             true); // fov_vertical (degrees)
        // Bytes 36–47 remain zero-padding
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        // ─────────────────────────────────────────────────────────────────
        // STEP 3 — Build bind groups for each pass
        // ─────────────────────────────────────────────────────────────────

        // ── Pass 0 bind group ─────────────────────────────────────────────
        // headings (3) r/w — Pass 0 writes it.
        // output_metrics (4) unused → _stubBuffer4.
        // adjacency (5)      unused → _stubBuffer5  (DISTINCT buffer, no aliasing).
        const headingsBindGroup = this.device.createBindGroup({
            label: 'Pass 0 — Headings Bind Group',
            layout: this.sharedBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer       } }, // uniforms
                { binding: 1, resource: { buffer: voxelBuffer              } }, // voxels
                { binding: 2, resource: { buffer: this.observerBuffer      } }, // observers
                { binding: 3, resource: { buffer: this.headingBuffer       } }, // headings (r/w)
                { binding: 4, resource: { buffer: this._stubBuffer4        } }, // output_metrics stub
                { binding: 5, resource: { buffer: this._stubBuffer5        } }, // adjacency stub
            ]
        });

        // ── Pass A bind group ─────────────────────────────────────────────
        // headings (3) r — filled by Pass 0.
        // output_metrics (4) r/w — Pass A writes isovist metrics.
        // adjacency (5) unused → _stubBuffer5.
        const isovistBindGroup = this.device.createBindGroup({
            label: 'Pass A — Isovist Bind Group',
            layout: this.sharedBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer       } }, // uniforms
                { binding: 1, resource: { buffer: voxelBuffer              } }, // voxels
                { binding: 2, resource: { buffer: this.observerBuffer      } }, // observers
                { binding: 3, resource: { buffer: this.headingBuffer       } }, // headings (r)
                { binding: 4, resource: { buffer: this.outputMetricsBuffer } }, // output_metrics (r/w)
                { binding: 5, resource: { buffer: this._stubBuffer5        } }, // adjacency stub
            ]
        });

        // ── Pass B bind group ─────────────────────────────────────────────
        // headings (3) r — filled by Pass 0.
        // output_metrics (4) uses _stubBuffer4 to avoid aliasing with slot 5.
        // adjacency (5) r/w — Pass B writes via atomicOr.
        const adjacencyBindGroup = this.device.createBindGroup({
            label: 'Pass B — Adjacency Bind Group',
            layout: this.sharedBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer       } }, // uniforms
                { binding: 1, resource: { buffer: voxelBuffer              } }, // voxels
                { binding: 2, resource: { buffer: this.observerBuffer      } }, // observers
                { binding: 3, resource: { buffer: this.headingBuffer       } }, // headings (r)
                { binding: 4, resource: { buffer: this._stubBuffer4        } }, // output_metrics stub
                { binding: 5, resource: { buffer: this.adjacencyBuffer     } }, // adjacency (r/w atomic)
            ]
        });

        // ─────────────────────────────────────────────────────────────────
        // STEP 4 — Encode & submit Pass 0 (must complete before Pass A/B
        //          read headingBuffer)
        // ─────────────────────────────────────────────────────────────────
        const pass0WGCount = Math.ceil(N / 64);

        const enc0 = this.device.createCommandEncoder({ label: 'Pass 0 Encoder' });
        const pass0 = enc0.beginComputePass({ label: 'Pass 0 — Heading Generation' });
        pass0.setPipeline(this.headingsPipeline);
        pass0.setBindGroup(0, headingsBindGroup);
        pass0.dispatchWorkgroups(pass0WGCount, 1, 1);
        pass0.end();
        this.device.queue.submit([enc0.finish()]);
        await this.device.queue.onSubmittedWorkDone(); // barrier: headings must be ready

        console.log(`[SpaceSyntaxSolver] Pass 0 (Headings): dispatched ${pass0WGCount} workgroup(s). ✓`);

        // ─────────────────────────────────────────────────────────────────
        // STEP 5 — Encode & submit Pass A (3D Isovist) + Pass B (Adjacency)
        //          in a single command buffer (both read headings, neither
        //          writes headings — no WAW hazard)
        // ─────────────────────────────────────────────────────────────────
        const passAWGCount = Math.ceil(N / 64);
        const passBWGCount = Math.ceil(N / 16);

        const enc1  = this.device.createCommandEncoder({ label: 'Pass A+B Encoder' });

        // Pass A — 3D Isovist
        const passA = enc1.beginComputePass({ label: 'Pass A — 3D Spherical Isovist' });
        passA.setPipeline(this.isovistPipeline);
        passA.setBindGroup(0, isovistBindGroup);
        passA.dispatchWorkgroups(passAWGCount, 1, 1);
        passA.end();

        // Pass B — Adjacency (reads headings, writes adjacencyBuffer)
        const passB = enc1.beginComputePass({ label: 'Pass B — Visibility Graph Adjacency' });
        passB.setPipeline(this.adjacencyPipeline);
        passB.setBindGroup(0, adjacencyBindGroup);
        passB.dispatchWorkgroups(passBWGCount, passBWGCount, 1);
        passB.end();

        this.device.queue.submit([enc1.finish()]);
        await this.device.queue.onSubmittedWorkDone();

        console.log(`[SpaceSyntaxSolver] Pass A (3D Isovist): ${passAWGCount} wg. ✓`);
        console.log(`[SpaceSyntaxSolver] Pass B (Adjacency): ${passBWGCount}×${passBWGCount} wg. ✓`);

        // ─────────────────────────────────────────────────────────────────
        // STEP 6 — Pass C (GPU BFS)
        // ─────────────────────────────────────────────────────────────────
        console.log('[SpaceSyntaxSolver] Pass C (GPU BFS): executing dispatch loop...');

        this.bfsFrontierBuffer?.destroy();
        this.bfsNextFrontierBuffer?.destroy();
        this.bfsDistanceBuffer?.destroy();
        this.bfsIntegrationBuffer?.destroy();
        this.bfsUniformBuffer?.destroy();

        const u32CountBfs = Math.ceil(N / 32);
        const frontierSize = Math.max(4, u32CountBfs * 4);
        const nodeArraySize = Math.max(4, N * 4);

        this.bfsFrontierBuffer = this.device.createBuffer({
            label: 'BFS Frontier Buffer',
            size: frontierSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.bfsNextFrontierBuffer = this.device.createBuffer({
            label: 'BFS Next Frontier Buffer',
            size: frontierSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.bfsDistanceBuffer = this.device.createBuffer({
            label: 'BFS Distance Buffer',
            size: nodeArraySize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.bfsIntegrationBuffer = this.device.createBuffer({
            label: 'BFS Integration Buffer',
            size: nodeArraySize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.bfsUniformBuffer = this.device.createBuffer({
            label: 'BFS Uniforms',
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const connectivity = new Float64Array(N);
        const meanDepth    = new Float64Array(N);
        const integration  = new Float64Array(N);

        const initialDistances = new Int32Array(N);
        const zeroFrontier = new Uint32Array(u32CountBfs);
        const maxDepth = Math.min(N, 30);
        const passCWGCount = Math.ceil(N / 64);

        for (let S = 0; S < N; S++) {
            initialDistances.fill(-1);
            initialDistances[S] = 0;
            this.device.queue.writeBuffer(this.bfsDistanceBuffer, 0, initialDistances);
            
            const initialFrontier = new Uint32Array(u32CountBfs);
            const u32Idx = Math.floor(S / 32);
            const bitIdx = S % 32;
            initialFrontier[u32Idx] = 1 << bitIdx;
            this.device.queue.writeBuffer(this.bfsFrontierBuffer, 0, initialFrontier);

            let currentFrontierBuf = this.bfsFrontierBuffer;
            let currentNextFrontierBuf = this.bfsNextFrontierBuffer;

            for (let depth = 0; depth < maxDepth; depth++) {
                this.device.queue.writeBuffer(currentNextFrontierBuf, 0, zeroFrontier);
                
                const bfsUniformData = new ArrayBuffer(16);
                const bfsDv = new DataView(bfsUniformData);
                bfsDv.setUint32(0, N, true);
                bfsDv.setUint32(4, depth, true);
                bfsDv.setUint32(8, S, true);
                this.device.queue.writeBuffer(this.bfsUniformBuffer, 0, bfsUniformData);

                const bfsBindGroup = this.device.createBindGroup({
                    label: `Pass C Bind Group (Depth ${depth}, Source ${S})`,
                    layout: this.bfsBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.bfsUniformBuffer } },
                        { binding: 1, resource: { buffer: this.adjacencyBuffer } },
                        { binding: 2, resource: { buffer: currentFrontierBuf } },
                        { binding: 3, resource: { buffer: currentNextFrontierBuf } },
                        { binding: 4, resource: { buffer: this.bfsDistanceBuffer } },
                        { binding: 5, resource: { buffer: this.bfsIntegrationBuffer } }
                    ]
                });

                const enc2 = this.device.createCommandEncoder();
                const passC = enc2.beginComputePass({ label: `Pass C — BFS (Depth ${depth})` });
                passC.setPipeline(this.bfsPipeline);
                passC.setBindGroup(0, bfsBindGroup);
                passC.dispatchWorkgroups(passCWGCount, 1, 1);
                passC.end();
                this.device.queue.submit([enc2.finish()]);

                const temp = currentFrontierBuf;
                currentFrontierBuf = currentNextFrontierBuf;
                currentNextFrontierBuf = temp;
            }

            const staging = this.device.createBuffer({
                size: nodeArraySize,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const encDist = this.device.createCommandEncoder();
            encDist.copyBufferToBuffer(this.bfsDistanceBuffer, 0, staging, 0, nodeArraySize);
            this.device.queue.submit([encDist.finish()]);
            await staging.mapAsync(GPUMapMode.READ);
            const distData = new Int32Array(staging.getMappedRange().slice(0));
            staging.unmap();
            staging.destroy();

            let totalDist = 0, reachable = 0, directNeighbors = 0;
            for (let v = 0; v < N; v++) {
                if (v === S) continue;
                const d = distData[v];
                if (d !== -1) {
                    totalDist += d;
                    reachable++;
                    if (d === 1) directNeighbors++;
                }
            }

            connectivity[S] = directNeighbors;
            if (reachable > 0) {
                const md = totalDist / reachable;
                meanDepth[S] = md;
                integration[S] = md > 0 ? 1.0 / md : 0;
            }
        }

        let sumInt = 0, sumMD = 0, sumConn = 0;
        for (let i = 0; i < N; i++) {
            sumInt  += integration[i];
            sumMD   += meanDepth[i];
            sumConn += connectivity[i];
        }

        let avgIntegration = N > 0 ? sumInt / N : 0;
        let avgMeanDepth   = N > 0 ? sumMD  / N : 0;
        const avgConnectivity = N > 0 ? sumConn / N : 0;
        if (N < 2) { avgIntegration = 0; avgMeanDepth = 0; }
        const connectivityPct = (avgConnectivity / Math.max(1, N - 1)) * 100;

        const graphMetrics = { 
            avgIntegration, avgMeanDepth, avgConnectivity, connectivityPct,
            perNode: { connectivity, meanDepth, integration } 
        };


        // ─────────────────────────────────────────────────────────────────
        // STEP 7 — Readback GPU results
        // ─────────────────────────────────────────────────────────────────
        const metricsArray   = await this._readbackMetrics(N);
        const packedAdjacency = await this._readbackAdjacency(N);

        // ─────────────────────────────────────────────────────────────────
        // STEP 8 — Unpack bit-packed adjacency matrix
        // ─────────────────────────────────────────────────────────────────
        const adjacencyMatrix = new Uint32Array(totalPairs);
        let visiblePairs = 0;

        for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
                const flatIdx = i * N + j;
                const u32Idx  = Math.floor(flatIdx / 32);
                const bitIdx  = flatIdx % 32;
                if ((packedAdjacency[u32Idx] & (1 << bitIdx)) !== 0) {
                    adjacencyMatrix[flatIdx] = 1;
                    visiblePairs++;
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // STEP 9 — Aggregate GPU BFS Logic completed earlier
        // ─────────────────────────────────────────────────────────────────

        // ─────────────────────────────────────────────────────────────────
        // STEP 10 — Aggregate and return results
        // ─────────────────────────────────────────────────────────────────

        // Copy computed headings back to observerPoints for the Visualizer cone helper.
        // We do NOT read back the headingBuffer (GPU-only) — instead we re-use the heading
        // that Pass 0 computed implicitly, exposed via a lightweight CPU re-run of the
        // 16-ray sweep on the already-uploaded data.  Since headings are purely cosmetic
        // (frustum widget), a one-time lightweight CPU estimate is acceptable here.
        //
        // NOTE: If you later add a GPU readback of headings, replace the block below.
        const voxelStateData = await this._readBufferToCPU(voxelBuffer);
        const headingMaxDist = this.config.maxRayDistMeters / resolution;
        for (let i = 0; i < N; i++) {
            const pt  = observerPoints[i];
            const dir = this._findLongestLineOfSightCPU(pt, nx, ny, nz, voxelStateData, headingMaxDist);
            pt.heading = { x: dir.dx, y: 0.0, z: dir.dz };
        }

        const result = this._aggregateResults(metricsArray, N, graphMetrics, resolution);

        result.status         = 'complete';
        result.observerPoints = observerPoints;
        result.gridConfig     = this.gridConfig;
        result.perNode        = graphMetrics.perNode;
        result.adjacencyMatrix = adjacencyMatrix;

        const graphDensity = totalPairs > 0 ? (visiblePairs / totalPairs * 100).toFixed(1) : 0;

        console.log('[SpaceSyntaxSolver] Analysis Complete (3D Spherical Isovist + VGA).');
        console.log(`  Observers       : ${N}`);
        console.log(`  Avg 3D Volume   : ${result.avgVolume.toFixed(2)} voxel³ (${result.avgVolumeM3.toFixed(3)} m³)`);
        console.log(`  Avg Surface Area: ${result.avgSurfaceArea.toFixed(2)} voxel² (${result.avgSurfaceAreaM2.toFixed(3)} m²)`);
        console.log(`  Avg Compactness : ${result.avgCompactness.toFixed(4)}`);
        console.log(`  Spatial Chaos   : ${result.spatialChaos.toFixed(4)}`);
        console.log(`  Adjacency       : ${visiblePairs}/${totalPairs} pairs (${graphDensity}% density)`);
        console.log(`  Avg Integration : ${result.globalIntegration.toFixed(4)}`);
        console.log(`  Avg Mean Depth  : ${result.meanDepth.toFixed(4)}`);
        console.log(`  Avg Connectivity: ${result.rawConnectivity.toFixed(1)} (${result.avgConnectivity.toFixed(1)}%)`);

        // ─────────────────────────────────────────────────────────────────
        // DATA SCIENCE EXPORT — Per-Node Metrics Table
        //
        // Outputs a clean row-per-node grid to the browser console.
        // Copy via: right-click → "Copy table" in DevTools, paste into CSV.
        //
        // Columns:
        //   agentIndex   — VGA node index (0-based), maps 1:1 to agent spawn order
        //   integration  — Normalised Space Syntax integration (1 / mean depth)
        //   connectivity — Raw direct-neighbour count from BFS (depth=1 edges)
        //   compactness  — 3D isovist compactness [0,1] (1 = perfectly spherical)
        //   aci          — Spatial Chaos Index = 1 − compactness [0,1]
        // ─────────────────────────────────────────────────────────────────
        const exportTable = [];
        const perNodeIntegration   = graphMetrics.perNode.integration;   // Float64Array[N]
        const perNodeConnectivity  = graphMetrics.perNode.connectivity;  // Float64Array[N] (raw count)

        for (let i = 0; i < N; i++) {
            // Compactness is at slot [i*4+2] of the GPU readback metrics array
            const nodeCompactness = metricsArray[i * 4 + 2];
            const nodeACI         = 1.0 - nodeCompactness; // Spatial Chaos Index

            exportTable.push({
                agentIndex:   i,
                integration:  Number(perNodeIntegration[i].toFixed(4)),
                connectivity: Number(perNodeConnectivity[i].toFixed(4)),
                compactness:  Number(nodeCompactness.toFixed(4)),
                aci:          Number(nodeACI.toFixed(4)),
            });
        }

        console.log('📊 [Data Science Export] Per-Node Metrics for CSV:');
        console.table(exportTable);

        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — GPU Buffer Helpers
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Read an entire GPU buffer back to CPU as a Float32Array.
     * Used only for the lightweight cosmetic heading re-computation (cone widget).
     */
    async _readBufferToCPU(gpuBuffer) {
        const staging = this.device.createBuffer({
            size:  gpuBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(gpuBuffer, 0, staging, 0, gpuBuffer.size);
        this.device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return data;
    }

    /**
     * Read back the per-observer isovist metrics (Float32 × 4 per observer).
     * Layout: [volume, surface_area, compactness, 0]
     */
    async _readbackMetrics(N) {
        const byteSize = Math.max(16, N * 4 * 4);
        const staging  = this.device.createBuffer({
            size:  byteSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.outputMetricsBuffer, 0, staging, 0, byteSize);
        this.device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return data;
    }

    /**
     * Read back the bit-packed adjacency matrix (Uint32Array).
     */
    async _readbackAdjacency(N) {
        const totalPairs = N * N;
        const u32Count   = Math.ceil(totalPairs / 32);
        const byteSize   = Math.max(4, u32Count * 4);
        const staging    = this.device.createBuffer({
            size:  byteSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.adjacencyBuffer, 0, staging, 0, byteSize);
        this.device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return data;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — Aggregate Results
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Aggregate per-observer 3D isovist metrics + BFS graph metrics.
     *
     * @param {Float32Array} metrics      — [volume, surface_area, compactness, 0] × N
     * @param {number}       N            — Observer count
     * @param {Object}       graphMetrics — BFS results
     * @param {number}       resolution   — metres per voxel
     * @returns {Object} Result object with 3D metrics + legacy aliases
     */
    _aggregateResults(metrics, N, graphMetrics, resolution) {
        // Physical unit scale factors
        const res2 = resolution * resolution;       // m² per voxel²
        const res3 = resolution * resolution * res2; // m³ per voxel³

        let sumVolume = 0, sumSurface = 0, sumCompactness = 0;

        const volumes      = new Float64Array(N);
        const compactnesses = new Float64Array(N);

        for (let i = 0; i < N; i++) {
            const vol    = metrics[i * 4 + 0]; // voxel³
            const surf   = metrics[i * 4 + 1]; // voxel²
            const compact = metrics[i * 4 + 2]; // dimensionless

            volumes[i]       = vol;
            compactnesses[i] = compact;

            sumVolume     += vol;
            sumSurface    += surf;
            sumCompactness += compact;
        }

        const avgVolume      = sumVolume      / N;
        const avgSurfaceArea = sumSurface     / N;
        const avgCompactness = sumCompactness / N;

        // Physical unit conversions (for UI display — NOT used in downstream thresholds)
        const avgVolumeM3      = avgVolume      * res3;
        const avgSurfaceAreaM2 = avgSurfaceArea * res2;

        // Standard deviation of compactness → visual complexity proxy
        let sumSqDiff = 0;
        for (let i = 0; i < N; i++) {
            const d = compactnesses[i] - avgCompactness;
            sumSqDiff += d * d;
        }
        const stddevCompactness = Math.sqrt(sumSqDiff / N);

        // Intelligibility = Pearson R² of connectivity vs integration
        let intelligibility = 0;
        if (graphMetrics?.perNode) {
            const connArr = graphMetrics.perNode.connectivity;
            const intArr  = graphMetrics.perNode.integration;
            const r = this._pearsonCorrelation(connArr, intArr, N);
            intelligibility = r * r;
        }

        // Spatial Chaos = 1 − average 3D compactness
        const spatialChaos = 1.0 - avgCompactness;

        return {
            // ── Primary 3D isovist metrics (internal voxel units) ──────────
            avgVolume,             // voxel³ — used by NeuroaestheticEvaluator thresholds
            avgSurfaceArea,        // voxel²
            avgCompactness,        // [0,1]
            observerCount: N,

            // ── Physical unit mirrors (for UI display only) ─────────────────
            avgVolumeM3,           // m³
            avgSurfaceAreaM2,      // m²

            // ── Spatial quality metrics ─────────────────────────────────────
            spatialChaos,
            visualComplexity: stddevCompactness,

            // ── Space Syntax graph metrics (from CPU BFS / future GPU BFS) ──
            globalIntegration: Math.max(0.001, graphMetrics.avgIntegration),
            intelligibility,
            meanDepth:         graphMetrics.avgMeanDepth,
            avgConnectivity:   graphMetrics.connectivityPct,  // %
            rawConnectivity:   graphMetrics.avgConnectivity,  // absolute count

            // ── Legacy aliases (interface stability with NeuroaestheticEvaluator) ──
            avgArea:       avgVolume,       // old 2D area → mapped to 3D volume
            avgPerimeter:  avgSurfaceArea,  // old 2D perim → mapped to surface area
            isovistArea:   avgVolume,
            visibilityIndex: avgCompactness, // now 3D compactness, same field name
            sampleCount:   N,
            synergy:       0,
        };
    }

    /**
     * Pearson product-moment correlation coefficient.
     * Returns value in [-1, 1].
     */
    _pearsonCorrelation(xs, ys, n) {
        if (n < 2) return 0;
        let sx = 0, sy = 0, sxy = 0, sx2 = 0, sy2 = 0;
        for (let i = 0; i < n; i++) {
            sx  += xs[i]; sy  += ys[i];
            sxy += xs[i] * ys[i];
            sx2 += xs[i] * xs[i]; sy2 += ys[i] * ys[i];
        }
        const num = n * sxy - sx * sy;
        const den = Math.sqrt((n * sx2 - sx * sx) * (n * sy2 - sy * sy));
        return den === 0 ? 0 : num / den;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — Stub results
    // ═══════════════════════════════════════════════════════════════════════

    _pendingResult() {
        return {
            status: 'pending', message: 'Place observers to calculate',
            avgVolume: 0, avgSurfaceArea: 0, avgVolumeM3: 0, avgSurfaceAreaM2: 0,
            avgCompactness: 0, spatialChaos: 0, visualComplexity: 0,
            observerCount: 0, globalIntegration: 0, intelligibility: 0,
            meanDepth: 0, avgConnectivity: 0, rawConnectivity: 0,
            avgArea: 0, avgPerimeter: 0, isovistArea: 0, visibilityIndex: 0,
            sampleCount: 0, synergy: 0, adjacencyMatrix: new Uint32Array(0),
        };
    }

    _emptyResult() {
        return { ...this._pendingResult(), status: 'empty' };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — Coordinate Conversion
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Convert world-space coordinates to voxel-space indices.
     */
    _worldToVoxel(worldPt) {
        const resolution = this.gridConfig.resolution;
        const bounds = this.gridConfig.bounds || {};

        let minX = 0, minY = 0, minZ = 0;
        if (bounds.min !== undefined && bounds.min.x !== undefined) {
            minX = bounds.min.x; minY = bounds.min.y; minZ = bounds.min.z;
        } else if (bounds.minX !== undefined) {
            minX = bounds.minX; minY = bounds.minY; minZ = bounds.minZ;
        }

        const vx = Math.floor((worldPt.x - minX) / resolution);
        const vy = Math.floor((worldPt.y - minY) / resolution);
        const vz = Math.floor((worldPt.z - minZ) / resolution);

        const { nx, ny, nz } = this.gridConfig.dimensions;
        return {
            x: Math.max(0, Math.min(nx - 1, vx)),
            y: Math.max(0, Math.min(ny - 1, vy)),
            z: Math.max(0, Math.min(nz - 1, vz))
        };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — CPU BFS Graph Traversal (Pass C functional fallback)
    // ═══════════════════════════════════════════════════════════════════════

    // ─────────────────────────────────────────────────────────────────
    // CPU BFS has been removed (migrated to GPU Pass C completely)
    // ─────────────────────────────────────────────────────────────────

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — Cosmetic CPU heading (for Visualizer cone widget only)
    //
    //  This is a lightweight CPU re-run of the Pass 0 algorithm, used only
    //  to populate pt.heading for the 3D frustum wire helper.
    //  It does NOT affect any analytical calculations; those use the GPU buffer.
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * 16-ray XZ sweep to find longest LOS heading (CPU cosmetic only).
     * @private
     */
    _findLongestLineOfSightCPU(pt, nx, ny, nz, data, maxDistVoxels) {
        const numRays  = 16;
        const stepSize = 0.5;
        const sliceSize = nx * ny;
        const maxSteps = Math.ceil(maxDistVoxels / stepSize);

        let maxDist = -1;
        let bestDx = 1.0, bestDz = 0.0;

        for (let i = 0; i < numRays; i++) {
            const angle = (i * Math.PI * 2.0) / numRays;
            const dx = Math.cos(angle);
            const dz = Math.sin(angle);

            let hitDist = maxDistVoxels;
            for (let step = 1; step <= maxSteps; step++) {
                const dist = step * stepSize;
                const ix = Math.floor(pt.x + dx * dist + 0.5);
                const iy = Math.floor(pt.y          + 0.5);
                const iz = Math.floor(pt.z + dz * dist + 0.5);
                if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
                    hitDist = dist; break;
                }
                if (Math.round(data[(ix + iy * nx + iz * sliceSize) * 8]) === 1) {
                    hitDist = dist; break;
                }
            }
            if (hitDist > maxDist) { maxDist = hitDist; bestDx = dx; bestDz = dz; }
        }

        const len = Math.sqrt(bestDx * bestDx + bestDz * bestDz);
        return len < 0.0001
            ? { dx: 1.0, dz: 0.0 }
            : { dx: bestDx / len, dz: bestDz / len };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — Shader Loaders
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Load the shared AVGA raycaster WGSL from disk.
     */
    async _loadShaderCode() {
        const response = await fetch('./wgsl/avga_raycaster.wgsl');
        if (!response.ok) throw new Error('Failed to load avga_raycaster.wgsl: ' + response.status);
        return response.text();
    }

    /**
     * Generate the GPU BFS frontier-expansion shader (Pass C mock).
     *
     * The shader is fully compilable WGSL. It expands one BFS frontier level
     * per dispatch. The JS host loop that drives it (dispatch × depth) is
     * stubbed in the analyze() method above with a detailed TODO comment.
     *
     * Uniform layout for BFS (16 bytes):
     *   u32 observer_count
     *   u32 current_depth     ← set by JS before each dispatch
     *   u32 source_node       ← BFS source (set per outer JS loop iteration)
     *   u32 _pad
     */
    _generateBFSShader() {
        return /* wgsl */ `
struct BfsUniforms {
    observer_count : u32,
    current_depth  : u32,
    source_node    : u32,
    _pad           : u32,
}

@group(0) @binding(0) var<uniform>            bfs_uniforms    : BfsUniforms;
@group(0) @binding(1) var<storage, read>      bfs_adjacency   : array<u32>;
@group(0) @binding(2) var<storage, read_write> frontier       : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> next_frontier  : array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> distances      : array<i32>;
@group(0) @binding(5) var<storage, read_write> bfs_integration: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn compute_bfs_frontier(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v      = gid.x;
    let N      = bfs_uniforms.observer_count;
    let depth  = bfs_uniforms.current_depth;

    if (v >= N) { return; }

    if (distances[v] != -1) { return; }

    let u32_count = (N + 31u) / 32u;

    for (var wi : u32 = 0u; wi < u32_count; wi++) {
        let word = atomicLoad(&frontier[wi]);
        if (word == 0u) { continue; }

        for (var bit : u32 = 0u; bit < 32u; bit++) {
            let u = wi * 32u + bit;
            if (u >= N) { break; }

            if ((word >> bit) == 0u) { break; }
            if ((word & (1u << bit)) == 0u) { continue; }

            let pair    = u * N + v;
            let adj_idx = pair / 32u;
            let adj_bit = pair % 32u;

            if ((bfs_adjacency[adj_idx] & (1u << adj_bit)) != 0u) {
                distances[v] = i32(depth + 1u);

                let nf_idx = v / 32u;
                let nf_bit = v % 32u;
                atomicOr(&next_frontier[nf_idx], 1u << nf_bit);
                return;
            }
        }
    }
}
        `;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Cleanup
    // ═══════════════════════════════════════════════════════════════════════

    destroy() {
        this.observerBuffer?.destroy();
        this.outputMetricsBuffer?.destroy();
        this.adjacencyBuffer?.destroy();
        this.headingBuffer?.destroy();
        this.uniformBuffer?.destroy();
        this._stubBuffer4?.destroy();
        this._stubBuffer5?.destroy();
        this.bfsIntegrationBuffer?.destroy();
        this.bfsFrontierBuffer?.destroy();
        this.bfsNextFrontierBuffer?.destroy();
        this.bfsDistanceBuffer?.destroy();
        this.bfsUniformBuffer?.destroy();

        this.observerBuffer       = null;
        this.outputMetricsBuffer  = null;
        this.adjacencyBuffer      = null;
        this.headingBuffer        = null;
        this.uniformBuffer        = null;
        this._stubBuffer4         = null;
        this._stubBuffer5         = null;
        this.bfsIntegrationBuffer = null;
        this.bfsFrontierBuffer    = null;
        this.bfsNextFrontierBuffer= null;
        this.bfsDistanceBuffer    = null;
        this.bfsUniformBuffer     = null;

        console.log('[SpaceSyntaxSolver] Destroyed.');
    }
}
