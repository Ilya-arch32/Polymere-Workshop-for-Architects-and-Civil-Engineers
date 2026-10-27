/**
 * Polymère — ACISolver
 *
 * GPU-солвер для вычисления Architectural Chaos Index (ACI) — геометрического
 * когнитивного налога по теории Prospect–Refuge.
 *
 * Использует точку входа `compute_aci_geometry` в wgsl/avga_raycaster.wgsl.
 * Для каждого агента выполняет ray-marching по вокселькной сетке и возвращает:
 *
 *   • prospect          Σ глубина лучей переднего полушария    (метры)
 *   • dBack             физическое расстояние до стены сзади   (метры)
 *   • compactness       степенное среднее 2D изовиста          ([0, 1])
 *   • refugeProtected   dBack < refugeMaxMeters                (bool)
 *
 * Примечание: метрики визуального пробоя через стекло (affordanceGap) удалены —
 * визуальная проницаемость без физического прохода является чертой Prospect,
 * а не когнитивной ошибкой. Слот [2] шейдерного вывода зарезервирован (0.0).
 *
 * Архитектурные решения:
 * - Разделяет шейдерный модуль avga_raycaster.wgsl со SpaceSyntaxSolver,
 *   не добавляя новых binding-слотов — live Space Syntax pipeline не затрагивается.
 * - Буферы персистентны и масштабируются по числу агентов (grow-on-demand),
 *   без повторного выделения на каждый вызов — пригодно для тика 5 Hz в BehaviorSimulator.
 * - Не имеет ES-импортов: самодостаточен для конкатенации через build-bundle.ps1.
 */
export class ACISolver {
    device;
    gridConfig;

    // GPU resources (created in initialize)
    pipeline;              // compute_aci_geometry
    bindGroupLayout;
    proxPipeline;          // compute_aci_proxemics (анизотропная проксемика, 16 лучей)
    proxBindGroupLayout;
    uniformBuffer;

    // Anisotropic proxemics: rays per agent (MUST match NUM_PROX_DIRS in WGSL).
    proxDirs = 16;

    // Per-population buffers (created / grown in _ensureCapacity)
    agentBuffer   = null;  // binding 2 (observers): vec4<f32> position per agent
    forwardBuffer = null;  // binding 3 (headings) : vec4<f32> forward  per agent
    aciOutBuffer  = null;  // binding 4 (metrics)  : 4×f32 result        per agent
    proxOutBuffer = null;  // binding 6 (proxemics): proxDirs×f32 radius(θ) per agent
    stagingBuffer = null;  // MAP_READ readback target (geometry)
    proxStaging   = null;  // MAP_READ readback target (proxemics)
    capacity      = 0;
    _mapInFlight  = false; // guard: never overlap mapAsync on the staging buffers

    config = {
        maxRayDistMeters:   30.0, // isovist ray reach
        refugeMaxMeters:     1.0, // dBack below this ⇒ back is protected
        rayStepVoxels:       0.5, // half-voxel march precision
    };

    constructor(device, gridConfig, config = {}) {
        this.device     = device;
        this.gridConfig = gridConfig;
        this.config     = { ...this.config, ...config };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — initialize()   (call once before computeGeometry)
    // ═══════════════════════════════════════════════════════════════════════
    async initialize() {
        const shaderCode = await this._loadShaderCode();

        this.device.pushErrorScope('validation');
        const module = this.device.createShaderModule({
            label: 'ACI Shared Compute Shader (avga_raycaster)',
            code: shaderCode,
        });
        const err = await this.device.popErrorScope();
        if (err) {
            console.error('[ACISolver] WGSL compilation error:', err.message);
            throw new Error('ACI WGSL compilation failed: ' + err.message);
        }

        // 5-slot layout: compute_aci_geometry touches uniforms(0), voxels(1),
        // observers(2), headings(3), output_metrics(4) — but NOT adjacency(5),
        // so slot 5 is intentionally absent (no stub needed).
        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'ACI Geometry Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // voxels
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // agent pos
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // agent fwd (declared r/w)
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // ACI out (r/w)
            ],
        });

        this.pipeline = this.device.createComputePipeline({
            label: 'ACI Geometry Pipeline',
            layout: this.device.createPipelineLayout({
                label: 'ACI Geometry Pipeline Layout',
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            compute: { module, entryPoint: 'compute_aci_geometry' },
        });

        // ── Sprint 4: proxemics pipeline (compute_aci_proxemics) ────────────
        //  Touches uniforms(0), voxels(1), observers(2) + writes proxemics(6).
        //  It does NOT use headings(3)/output_metrics(4)/adjacency(5), so those
        //  slots are omitted from this layout (unused declared bindings are legal).
        this.proxBindGroupLayout = this.device.createBindGroupLayout({
            label: 'ACI Proxemics Bind Group Layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform'           } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // voxels
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // agent pos
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage'           } }, // proxemics out
            ],
        });

        this.proxPipeline = this.device.createComputePipeline({
            label: 'ACI Proxemics Pipeline',
            layout: this.device.createPipelineLayout({
                label: 'ACI Proxemics Pipeline Layout',
                bindGroupLayouts: [this.proxBindGroupLayout],
            }),
            compute: { module, entryPoint: 'compute_aci_proxemics' },
        });

        // 48-byte uniform block — identical layout to SpaceSyntaxSolver so the
        // shared shader's `Uniforms` struct reads the same fields.
        this.uniformBuffer = this.device.createBuffer({
            label: 'ACI Uniform Buffer',
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        console.log('[ACISolver] Initialized (compute_aci_geometry + compute_aci_proxemics).');
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _ensureCapacity(n)   grow persistent buffers to hold n agents
    // ═══════════════════════════════════════════════════════════════════════
    _ensureCapacity(n) {
        if (n <= this.capacity && this.agentBuffer) return;

        this.agentBuffer?.destroy();
        this.forwardBuffer?.destroy();
        this.aciOutBuffer?.destroy();
        this.proxOutBuffer?.destroy();
        this.stagingBuffer?.destroy();
        this.proxStaging?.destroy();

        const cap  = Math.max(n, 1);
        const vec4 = cap * 4 * 4;              // 16 bytes/agent
        const out  = cap * 4 * 4;              // 4 f32/agent (geometry)
        const prox = cap * this.proxDirs * 4;  // proxDirs f32/agent (proxemics)

        this.agentBuffer = this.device.createBuffer({
            label: 'ACI Agent Position Buffer',
            size: vec4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.forwardBuffer = this.device.createBuffer({
            label: 'ACI Agent Forward Buffer',
            size: vec4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.aciOutBuffer = this.device.createBuffer({
            label: 'ACI Output Buffer',
            size: out, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.proxOutBuffer = this.device.createBuffer({
            label: 'ACI Proxemics Output Buffer',
            size: prox, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        this.stagingBuffer = this.device.createBuffer({
            label: 'ACI Readback Staging',
            size: out, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        this.proxStaging = this.device.createBuffer({
            label: 'ACI Proxemics Readback Staging',
            size: prox, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        this.capacity = cap;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — computeGeometry(voxelBuffer, agents)
    //
    //  @param {GPUBuffer} voxelBuffer  — the grid buffer (stride 8 f32).
    //  @param {Array<{x,y,z,fx,fy,fz}>} agents — WORLD-space position + forward.
    //  @returns {Promise<Array<Object>>} per-agent ACI, or null if a prior
    //           readback is still in flight (throttle-safe for Sprint 3).
    // ═══════════════════════════════════════════════════════════════════════
    async computeGeometry(voxelBuffer, agents) {
        if (!this.pipeline) throw new Error('[ACISolver] initialize() not called.');
        if (!agents || agents.length === 0) return [];
        if (this._mapInFlight) return null; // previous tick's readback not done

        const N = agents.length;
        this._ensureCapacity(N);

        const { nx, ny, nz } = this.gridConfig.dimensions;
        const resolution = this.gridConfig.resolution;

        // Pack agent position (voxel coords) + forward (world-space unit dir).
        const pos = new Float32Array(N * 4);
        const fwd = new Float32Array(N * 4);
        for (let i = 0; i < N; i++) {
            const a = agents[i];
            const v = this._worldToVoxel(a);
            pos[i * 4 + 0] = v.x; pos[i * 4 + 1] = v.y; pos[i * 4 + 2] = v.z; pos[i * 4 + 3] = 0;
            fwd[i * 4 + 0] = a.fx || 0; fwd[i * 4 + 1] = a.fy || 0; fwd[i * 4 + 2] = a.fz || 0; fwd[i * 4 + 3] = 0;
        }
        this.device.queue.writeBuffer(this.agentBuffer,   0, pos);
        this.device.queue.writeBuffer(this.forwardBuffer, 0, fwd);
        this.device.queue.writeBuffer(this.aciOutBuffer,  0, new Float32Array(N * 4));            // zero stale
        this.device.queue.writeBuffer(this.proxOutBuffer, 0, new Float32Array(N * this.proxDirs)); // zero stale

        // Uniforms (same 48-byte layout the shared shader expects).
        const maxRayDistVoxels = this.config.maxRayDistMeters / resolution;
        const ub = new ArrayBuffer(48);
        const dv = new DataView(ub);
        dv.setUint32( 0, nx, true);
        dv.setUint32( 4, ny, true);
        dv.setUint32( 8, nz, true);
        dv.setUint32(12, N,  true);                          // observer_count = agent count
        dv.setFloat32(16, resolution,               true);
        dv.setFloat32(20, maxRayDistVoxels,         true);
        dv.setFloat32(24, this.config.rayStepVoxels, true);
        dv.setFloat32(28, 120.0, true);                      // fov_horizontal (unused by this pass)
        dv.setFloat32(32,  90.0, true);                      // fov_vertical   (unused by this pass)
        this.device.queue.writeBuffer(this.uniformBuffer, 0, ub);

        const bindGroup = this.device.createBindGroup({
            label: 'ACI Geometry Bind Group',
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: voxelBuffer        } },
                { binding: 2, resource: { buffer: this.agentBuffer   } },
                { binding: 3, resource: { buffer: this.forwardBuffer } },
                { binding: 4, resource: { buffer: this.aciOutBuffer  } },
            ],
        });

        const proxBindGroup = this.device.createBindGroup({
            label: 'ACI Proxemics Bind Group',
            layout: this.proxBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: voxelBuffer        } },
                { binding: 2, resource: { buffer: this.agentBuffer   } },
                { binding: 6, resource: { buffer: this.proxOutBuffer } },
            ],
        });

        const byteLen     = N * 4 * 4;
        const proxByteLen = N * this.proxDirs * 4;
        const enc = this.device.createCommandEncoder({ label: 'ACI Encoder' });

        // Pass 1 — geometric tax (prospect / refuge / compactness).
        const geoPass = enc.beginComputePass({ label: 'ACI Pass 1 — Geometric Tax' });
        geoPass.setPipeline(this.pipeline);
        geoPass.setBindGroup(0, bindGroup);
        geoPass.dispatchWorkgroups(Math.ceil(N / 64), 1, 1);
        geoPass.end();

        // Pass 2 — anisotropic proxemics. Separate pass in the SAME encoder;
        // WebGPU guarantees a memory barrier between passes, and both share the
        // agent-position/uniform buffers written above.
        const proxPass = enc.beginComputePass({ label: 'ACI Pass 2 — Proxemics' });
        proxPass.setPipeline(this.proxPipeline);
        proxPass.setBindGroup(0, proxBindGroup);
        proxPass.dispatchWorkgroups(Math.ceil(N / 64), 1, 1);
        proxPass.end();

        enc.copyBufferToBuffer(this.aciOutBuffer,  0, this.stagingBuffer, 0, byteLen);
        enc.copyBufferToBuffer(this.proxOutBuffer, 0, this.proxStaging,   0, proxByteLen);
        this.device.queue.submit([enc.finish()]);

        // Async readback of BOTH outputs (guarded so overlapping ticks can't
        // double-map). Map concurrently; both resolve before we copy + unmap.
        this._mapInFlight = true;
        let raw, proxRaw;
        try {
            await Promise.all([
                this.stagingBuffer.mapAsync(GPUMapMode.READ, 0, byteLen),
                this.proxStaging.mapAsync(GPUMapMode.READ, 0, proxByteLen),
            ]);
            raw     = new Float32Array(this.stagingBuffer.getMappedRange(0, byteLen).slice(0));
            proxRaw = new Float32Array(this.proxStaging.getMappedRange(0, proxByteLen).slice(0));
            this.stagingBuffer.unmap();
            this.proxStaging.unmap();
        } finally {
            this._mapInFlight = false;
        }

        // Decode per-agent results. Geometry is in voxel units (→ metres here);
        // proxemics is ALREADY in metres (the GPU applied the Hall curve).
        // Slot [2] is reserved (0.0) — the affordance gap was removed.
        const D = this.proxDirs;
        const results = new Array(N);
        for (let i = 0; i < N; i++) {
            const prospect    = raw[i * 4 + 0] * resolution;
            const dBack       = raw[i * 4 + 1] * resolution;
            const compactness = raw[i * 4 + 3];
            // Anisotropic personal-space radius(θ): index d ↔ angle d·(2π/D) in
            // the XZ plane (d=0 → +X, CCW toward +Z), matching atan2(dz, dx).
            const proxemics = proxRaw.slice(i * D, i * D + D);
            results[i] = {
                prospect,
                dBack,
                compactness,
                refugeProtected: dBack < this.config.refugeMaxMeters,
                proxemics,          // Float32Array(proxDirs), metres
                proxDirs:  D,       // ray count, for CPU θ→bucket mapping
            };
        }
        return results;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — world → voxel coordinate conversion (clamped in-bounds)
    //  Mirrors SpaceSyntaxSolver._worldToVoxel so agent coords line up exactly
    //  with the grid the Space Syntax observers use.
    // ═══════════════════════════════════════════════════════════════════════
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
            z: Math.max(0, Math.min(nz - 1, vz)),
        };
    }

    async _loadShaderCode() {
        // Cache-bust to match the loader convention — avoids the browser HTTP cache
        // serving a stale shader after a re-upload (a silent, confusing failure).
        const response = await fetch('./wgsl/avga_raycaster.wgsl?v=' + Date.now());
        if (!response.ok) throw new Error('Failed to load avga_raycaster.wgsl: ' + response.status);
        return response.text();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  Cleanup
    // ═══════════════════════════════════════════════════════════════════════
    destroy() {
        this.agentBuffer?.destroy();
        this.forwardBuffer?.destroy();
        this.aciOutBuffer?.destroy();
        this.proxOutBuffer?.destroy();
        this.stagingBuffer?.destroy();
        this.proxStaging?.destroy();
        this.uniformBuffer?.destroy();
        this.agentBuffer = this.forwardBuffer = this.aciOutBuffer = this.proxOutBuffer = null;
        this.stagingBuffer = this.proxStaging = this.uniformBuffer = null;
        this.capacity = 0;
        console.log('[ACISolver] Destroyed.');
    }
}

// Optional global handle for the bundle build (parallels other AHI modules).
if (typeof window !== 'undefined') {
    window.ACISolver = ACISolver;
}
