/**
 * AHI 2.0 Ultimate - Neuroaesthetic Evaluator
 *
 * Scientifically-grounded aesthetic metrics based on cognitive neuroscience
 * Replaces subjective "beauty" with measurable neurobiological responses
 */
import { PHYSICS_CONSTANTS } from './VoxelTypes.js';
export class NeuroaestheticEvaluator {
    device;
    fractalAnalyzer;
    spaceSyntaxSolver;
    spectralTracer;
    // Scientific thresholds from literature
    OPTIMAL_FRACTAL_D = PHYSICS_CONSTANTS.FRACTAL_OPTIMAL_D;
    FRACTAL_TOLERANCE = PHYSICS_CONSTANTS.FRACTAL_TOLERANCE;
    OPTIMAL_ENTROPY_MIN = PHYSICS_CONSTANTS.ENTROPY_OPTIMAL_MIN;
    OPTIMAL_ENTROPY_MAX = PHYSICS_CONSTANTS.ENTROPY_OPTIMAL_MAX;
    MIN_INTELLIGIBILITY = 0.7;
    MIN_EML = 200;
    MIN_CS = 0.3;
    // Optional acoustic solver reference
    acousticSolver = null;
    // Backend URL for Python analysis service
    backendUrl = '';
    // WebGPU Compute resources for entropy calculation
    entropyPipeline = null;
    entropyBindGroupLayout = null;
    histogramBuffer = null;
    // Cached analysis results
    cachedEntropyResult = null;
    cachedColorHarmony = null;
    CACHE_TTL_MS = 1000; // 1 second cache
    constructor(device, fractalAnalyzer, spaceSyntaxSolver, spectralTracer, acousticSolver) {
        this.device = device;
        this.fractalAnalyzer = fractalAnalyzer;
        this.spaceSyntaxSolver = spaceSyntaxSolver;
        this.spectralTracer = spectralTracer;
        this.acousticSolver = acousticSolver || null;
        this.gridConfig = spaceSyntaxSolver?.gridConfig || fractalAnalyzer?.gridConfig || null;
        // Initialize WebGPU entropy compute pipeline
        this.initializeEntropyPipeline();
    }
    /**
     * Set grid configuration for proper voxel calculations
     */
    setGridConfig(gridConfig) {
        this.gridConfig = gridConfig;
    }
    /**
     * Set backend URL for Python analysis service
     */
    setBackendUrl(url) {
        this.backendUrl = url;
    }
    /**
     * Initialize WebGPU compute pipeline for entropy calculation
     * This provides a fallback when Python backend is unavailable
     */
    async initializeEntropyPipeline() {
        try {
            // VISUAL COMPLEXITY ENTROPY: Measures perceivable scene richness
            // Focus on BOUNDARY voxels + material variety for true visual entropy
            const shaderModule = this.device.createShaderModule({
                label: 'Visual Complexity Entropy Shader',
                code: `
                    // Visual Complexity Entropy - Neuroaesthetics-aligned measurement
                    // Focuses on BOUNDARY voxels (perceivable surfaces) + material variety
                    struct Uniforms {
                        grid_size: vec3<u32>,
                        total_voxels: u32,
                    }
                    
                    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
                    @group(0) @binding(1) var<storage, read> voxels: array<f32>;
                    @group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;
                    @group(0) @binding(3) var<storage, read_write> result: array<f32>;
                    
                    // Voxel layout: [state, material, density, temp, vx, vy, vz, pad] = 8 floats
                    const VOXEL_STRIDE: u32 = 8u;
                    
                    // VISUAL COMPLEXITY ENTROPY
                    // Only boundary voxels (SOLID with FLUID neighbor) contribute
                    // Histogram bins: material(0-15) * 32 + surface_topology(0-31) = 512 bins
                    @compute @workgroup_size(64)
                    fn buildHistogram(@builtin(global_invocation_id) gid: vec3<u32>) {
                        let idx = gid.x;
                        if (idx >= uniforms.total_voxels) { return; }
                        
                        // Get 3D position
                        let x = idx % uniforms.grid_size.x;
                        let y = (idx / uniforms.grid_size.x) % uniforms.grid_size.y;
                        let z = idx / (uniforms.grid_size.x * uniforms.grid_size.y);
                        
                        let raw_state = u32(voxels[idx * VOXEL_STRIDE]);
                        
                        // Only process SOLID voxels (potential boundary surfaces)
                        let is_solid = (raw_state & 1u) != 0u;
                        if (!is_solid) { return; }
                        
                        // Get material ID (index 1 in voxel struct)
                        let material_id = u32(voxels[idx * VOXEL_STRIDE + 1u]);
                        
                        // 26-connected neighborhood analysis for surface topology
                        var fluid_neighbors: u32 = 0u;
                        var surface_normal_x: i32 = 0;
                        var surface_normal_y: i32 = 0;
                        var surface_normal_z: i32 = 0;
                        var neighbor_material_variety: u32 = 0u;
                        var seen_materials: u32 = 0u; // Bitmask of unique materials seen
                        
                        // Check all 26 neighbors
                        for (var dx: i32 = -1; dx <= 1; dx++) {
                            for (var dy: i32 = -1; dy <= 1; dy++) {
                                for (var dz: i32 = -1; dz <= 1; dz++) {
                                    if (dx == 0 && dy == 0 && dz == 0) { continue; }
                                    
                                    let nx = i32(x) + dx;
                                    let ny = i32(y) + dy;
                                    let nz = i32(z) + dz;
                                    
                                    if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
                                        ny >= 0 && ny < i32(uniforms.grid_size.y) &&
                                        nz >= 0 && nz < i32(uniforms.grid_size.z)) {
                                        
                                        let n_idx = u32(nx) + u32(ny) * uniforms.grid_size.x + 
                                                   u32(nz) * uniforms.grid_size.x * uniforms.grid_size.y;
                                        let n_state = u32(voxels[n_idx * VOXEL_STRIDE]);
                                        let n_material = u32(voxels[n_idx * VOXEL_STRIDE + 1u]);
                                        
                                        // Count FLUID neighbors (air/outdoor = visible surface)
                                        if ((n_state & 2u) != 0u) {
                                            fluid_neighbors += 1u;
                                            // Accumulate surface normal direction
                                            surface_normal_x -= dx;
                                            surface_normal_y -= dy;
                                            surface_normal_z -= dz;
                                        }
                                        
                                        // Track neighbor material variety (for SOLID neighbors)
                                        if ((n_state & 1u) != 0u && n_material != material_id) {
                                            let mat_bit = 1u << (n_material & 15u);
                                            if ((seen_materials & mat_bit) == 0u) {
                                                seen_materials = seen_materials | mat_bit;
                                                neighbor_material_variety += 1u;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Only boundary voxels (have at least one FLUID neighbor) contribute
                        if (fluid_neighbors == 0u) { return; }
                        
                        // Classify surface topology by dominant normal direction
                        // 0=floor/ceiling, 1=wall-X, 2=wall-Z, 3=corner, 4=edge
                        var topology: u32 = 0u;
                        let abs_nx = abs(surface_normal_x);
                        let abs_ny = abs(surface_normal_y);
                        let abs_nz = abs(surface_normal_z);
                        let max_component = max(abs_nx, max(abs_ny, abs_nz));
                        
                        if (max_component == abs_ny && abs_ny > abs_nx + abs_nz) {
                            topology = 0u; // Horizontal surface (floor/ceiling)
                        } else if (abs_nx > abs_nz) {
                            topology = 1u; // X-facing wall
                        } else if (abs_nz > abs_nx) {
                            topology = 2u; // Z-facing wall
                        } else if (fluid_neighbors >= 8u) {
                            topology = 3u; // Corner (many exposed faces)
                        } else {
                            topology = 4u; // Edge
                        }
                        
                        // Add exposure level: 0-7 based on fluid neighbor count
                        let exposure = clamp(fluid_neighbors / 4u, 0u, 7u);
                        
                        // Final topology encoding: topology(0-4) * 8 + exposure(0-7) = 0-39
                        let surface_code = clamp(topology * 8u + exposure, 0u, 39u);
                        
                        // Add material variety bonus (0-8)
                        let variety_code = clamp(neighbor_material_variety, 0u, 7u);
                        
                        // Histogram bin: material(0-15) * 32 + surface_code(0-31)
                        // Use 512 bins for rich pattern capture
                        let material_bin = clamp(material_id, 0u, 15u);
                        let combined_surface = clamp(surface_code + variety_code, 0u, 31u);
                        let bin = material_bin * 32u + combined_surface;
                        
                        atomicAdd(&histogram[bin], 1u);
                        
                        // Also track total boundary count in bin 511
                        atomicAdd(&histogram[511u], 1u);
                    }
                    
                    // Calculate Shannon entropy from boundary voxel histogram
                    // H = -Σ p_i * log2(p_i)
                    @compute @workgroup_size(1)
                    fn calculateEntropy(@builtin(global_invocation_id) gid: vec3<u32>) {
                        // Get total boundary voxels from bin 511
                        let boundary_total = f32(atomicLoad(&histogram[511u]));
                        
                        if (boundary_total < 100.0) {
                            // Not enough boundary voxels for meaningful entropy
                            result[0] = 0.0;
                            result[1] = 0.0;
                            return;
                        }
                        
                        var entropy: f32 = 0.0;
                        var non_zero_bins: u32 = 0u;
                        
                        // Calculate entropy over bins 0-510 (511 is the counter)
                        for (var i: u32 = 0u; i < 511u; i++) {
                            let count = f32(atomicLoad(&histogram[i]));
                            if (count > 0.0) {
                                let p = count / boundary_total;
                                entropy -= p * log2(p);
                                non_zero_bins += 1u;
                            }
                        }
                        
                        // Store raw entropy (max possible = log2(511) ≈ 9 bits)
                        result[0] = entropy;
                        
                        // Store boundary density (edge density proxy)
                        // Ratio of boundary voxels to total voxels
                        result[1] = boundary_total / f32(uniforms.total_voxels);
                    }
                `
            });
            this.entropyBindGroupLayout = this.device.createBindGroupLayout({
                label: 'Entropy Bind Group Layout',
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }
                ]
            });
            this.entropyPipeline = this.device.createComputePipeline({
                label: 'Entropy Histogram Pipeline',
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [this.entropyBindGroupLayout]
                }),
                compute: {
                    module: shaderModule,
                    entryPoint: 'buildHistogram'
                }
            });

            // Second pipeline for entropy calculation from histogram
            this.entropyCalcPipeline = this.device.createComputePipeline({
                label: 'Entropy Calc Pipeline',
                layout: this.device.createPipelineLayout({
                    bindGroupLayouts: [this.entropyBindGroupLayout]
                }),
                compute: {
                    module: shaderModule,
                    entryPoint: 'calculateEntropy'
                }
            });

            // Allocate histogram buffer (512 bins: material(16) * topology(32))
            this.histogramBuffer = this.device.createBuffer({
                label: 'Histogram Buffer',
                size: 512 * 4, // 512 u32 bins (last bin = boundary count)
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            console.log('[NeuroaestheticEvaluator] Entropy pipeline initialized');
        }
        catch (error) {
            console.warn('[NeuroaestheticEvaluator] Failed to initialize entropy pipeline:', error);
        }
    }
    /**
     * Set acoustic solver for RT60 integration
     */
    setAcousticSolver(solver) {
        this.acousticSolver = solver;
    }
    /**
     * Evaluate neuroaesthetic response to environment
     */
    async evaluate(voxelBuffer, onProgress = null) {
        // 1. Biophilic Fluency (Fractal Analysis)
        // NO FALLBACK - require valid FractalAnalyzer
        if (!this.fractalAnalyzer) {
            throw new Error('[NeuroaestheticEvaluator] FractalAnalyzer not available. Initialize with valid FractalAnalyzer.');
        }
        const fractalMetrics = await this.fractalAnalyzer.analyze(voxelBuffer);
        const fractalD = fractalMetrics.fractalDimension;
        // Categorize based on Taylor et al. research
        let fractalCategory;
        if (fractalD < 1.3) {
            fractalCategory = 'low';
        }
        else if (fractalD >= 1.3 && fractalD <= 1.5) {
            fractalCategory = 'optimal';
        }
        else {
            fractalCategory = 'high';
        }
        // Stress reduction peaks at D=1.4 (up to 60% reduction)
        const stressReduction = this.calculateStressReduction(fractalD);
        if (onProgress) onProgress(15);

        // 2. Spatial Cognition (Space Syntax)
        let spaceMetrics = null;
        if (this.spaceSyntaxSolver) {
            try {
                spaceMetrics = await this.spaceSyntaxSolver.analyze(voxelBuffer);
            } catch (e) {
                console.warn('[NeuroaestheticEvaluator] SpaceSyntaxSolver failed:', e.message);
            }
        } else {
            // NO FALLBACK - throw error if SpaceSyntaxSolver not available
            throw new Error('[NeuroaestheticEvaluator] SpaceSyntaxSolver not available. Initialize with valid SpaceSyntaxSolver.');
        }
        if (!spaceMetrics || spaceMetrics.globalIntegration === 0) {
            // NO FALLBACK - throw error for invalid results
            throw new Error('[NeuroaestheticEvaluator] SpaceSyntax returned invalid results (zeros). Fix SpaceSyntax analysis.');
        }
        // Intelligibility: correlation between local and global integration
        const spatialIntelligibility = spaceMetrics.intelligibility;
        // Prospect-Refuge theory (Appleton): balance of view and shelter
        const prospectRefuge = this.calculateProspectRefuge(spaceMetrics);
        // Mystery & Complexity (Kaplan): partial occlusion increases exploration
        const mysteryComplexity = spaceMetrics.visualComplexity;
        if (onProgress) onProgress(40);

        // 3. Visual Processing Load
        const visualEntropy = await this.calculateVisualEntropy(voxelBuffer);
        const colorHarmony = await this.calculateColorHarmony(voxelBuffer);
        const edgeDensity = fractalMetrics.lacunarity; // Proxy for contour richness
        if (onProgress) onProgress(70);

        // 4. Circadian Entrainment (from spectral analysis)
        // These metrics are optional - only set if spectralTracer is available
        let melanopicLux = null;
        let circadianStimulus = null;
        let spectralQuality = null;
        if (this.spectralTracer) {
            const lightingSample = { x: 5, y: 5, z: 1.5 }; // Eye level
            const lightingMetrics = await this.spectralTracer.computeMetrics(lightingSample);
            melanopicLux = lightingMetrics.melanopicLux;
            circadianStimulus = lightingMetrics.circadianStimulus;
            spectralQuality = lightingMetrics.colorRenderingIndex || null;
        }
        // 5. Acoustic Comfort (integrated with AcousticSolver)
        // These metrics are optional - only set if acousticSolver is available
        let reverberationBalance = null;
        let speechClarity = null;
        let acousticPrivacy = null;
        if (this.acousticSolver) {
            const acousticMetrics = await this.acousticSolver.calculateMetrics();
            reverberationBalance = this.calculateRT60Score(acousticMetrics.RT60);
            speechClarity = this.calculateC50Score(acousticMetrics.C50);
            acousticPrivacy = null; // Would need sound insulation calculation
        }
        if (onProgress) onProgress(95);

        // Calculate overall Harmony Score (0-100)
        // Combines fractal, entropy, color harmony, and spatial metrics
        const fractalScore = fractalCategory === 'optimal' ? 1.0 :
            (fractalCategory === 'low' ? 0.5 : 0.3);
        const entropyScore = Math.min(1, visualEntropy / 6); // Optimal ~6 bits
        const spatialScore = Math.min(1, spatialIntelligibility);
        const harmonyScore = Math.round(
            (fractalScore * 0.3 + entropyScore * 0.2 + colorHarmony * 0.25 + spatialScore * 0.25) * 100
        );

        // Возвращаем только сырые метрики - итоговые баллы рассчитываются на бэкенде
        return {
            // Biophilic Fluency
            fractalDimension: fractalD,
            fractalCategory,
            stressReduction,
            // Spatial Cognition
            spatialIntelligibility,
            prospectRefuge,
            mysteryComplexity,
            // Visual Processing
            visualEntropy,
            colorHarmony,
            edgeDensity,
            // Circadian
            melanopicLux,
            circadianStimulus,
            spectralQuality,
            // Acoustic
            reverberationBalance,
            speechClarity,
            acousticPrivacy,
            // Overall Harmony Score (0-100)
            harmonyScore
        };
    }
    /**
     * Calculate stress reduction from fractal dimension
     * Based on Taylor et al. (2006) and Hagerhall et al. (2008)
     */
    calculateStressReduction(D) {
        // Peak stress reduction at D=1.4
        // Gaussian curve centered at optimal
        const deviation = Math.abs(D - this.OPTIMAL_FRACTAL_D);
        const sigma = 0.15; // Width of optimal range
        // Up to 60% cortisol reduction at optimal D
        return 0.6 * Math.exp(-(deviation * deviation) / (2 * sigma * sigma));
    }
    /**
     * Calculate prospect-refuge balance
     * Based on Appleton's evolutionary theory
     */
    calculateProspectRefuge(spaceMetrics) {
        // Ideal: high prospect (view) with some refuge (shelter)
        // Too open = exposed, too enclosed = trapped
        const openness = spaceMetrics.meanDepth / 10; // Normalized
        const enclosure = 1 - openness;
        // Optimal at 70% open, 30% enclosed
        const idealRatio = 0.7;
        const balance = 1 - Math.abs(openness - idealRatio);
        return balance;
    }
    /**
     * Calculate visual entropy using WebGPU compute shader or Python backend
     * TRL 7: Real algorithm instead of Math.random() placeholder
     *
     * Uses Shannon entropy: H = -Σ p_i * log2(p_i)
     * Optimal range: 4-6 bits for comfortable visual complexity
     */
    async calculateVisualEntropy(voxelBuffer) {
        // Check cache first
        const now = Date.now();
        if (this.cachedEntropyResult &&
            (now - this.cachedEntropyResult.timestamp) < this.CACHE_TTL_MS) {
            return this.cachedEntropyResult.entropy;
        }
        let entropyBits;
        // Strategy 1: Try WebGPU compute shader (fastest, no network)
        if (this.entropyPipeline && this.histogramBuffer) {
            try {
                entropyBits = await this.calculateEntropyGPU(voxelBuffer);
                console.log(`[NeuroaestheticEvaluator] GPU entropy: ${entropyBits.toFixed(2)} bits`);
            }
            catch (error) {
                console.warn('[NeuroaestheticEvaluator] GPU entropy failed, trying backend');
                entropyBits = await this.calculateEntropyBackend(voxelBuffer);
            }
        }
        else {
            // Strategy 2: Python backend API
            entropyBits = await this.calculateEntropyBackend(voxelBuffer);
        }
        // Cache result
        this.cachedEntropyResult = { entropy: entropyBits, timestamp: now };

        // Return raw entropy in bits (not normalized score)
        return entropyBits;
    }
    /**
     * Calculate entropy using WebGPU compute shader
     * Implements Shannon entropy on voxel density distribution
     */
    async calculateEntropyGPU(voxelBuffer) {
        if (!this.entropyPipeline || !this.histogramBuffer || !this.entropyBindGroupLayout) {
            throw new Error('Entropy pipeline not initialized');
        }

        // Voxel format: 8 floats = 32 bytes per voxel
        const VOXEL_BYTES = 8 * 4; // 32 bytes
        const totalVoxels = Math.floor(voxelBuffer.size / VOXEL_BYTES);

        // Get grid dimensions from gridConfig or estimate
        const nx = this.gridConfig?.dimensions?.nx || Math.ceil(Math.cbrt(totalVoxels));
        const ny = this.gridConfig?.dimensions?.ny || nx;
        const nz = this.gridConfig?.dimensions?.nz || nx;

        console.log(`[NeuroaestheticEvaluator] Entropy GPU DEBUG:`);
        console.log(`  - voxelBuffer.size: ${voxelBuffer.size} bytes`);
        console.log(`  - totalVoxels: ${totalVoxels}`);
        console.log(`  - grid: ${nx}x${ny}x${nz}`);

        // Create uniform buffer with correct grid size
        const uniformData = new Uint32Array([nx, ny, nz, totalVoxels]);
        const uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Create result buffer
        const resultBuffer = this.device.createBuffer({
            size: 8, // 2 floats: entropy, edge_density
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Clear histogram (512 bins for material × topology)
        const zeroData = new Uint32Array(512).fill(0);
        this.device.queue.writeBuffer(this.histogramBuffer, 0, zeroData);

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.entropyBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: voxelBuffer } },
                { binding: 2, resource: { buffer: this.histogramBuffer } },
                { binding: 3, resource: { buffer: resultBuffer } }
            ]
        });

        // Execute compute - Step 1: Build histogram
        const commandEncoder = this.device.createCommandEncoder();
        const pass1 = commandEncoder.beginComputePass();
        pass1.setPipeline(this.entropyPipeline); // buildHistogram
        pass1.setBindGroup(0, bindGroup);
        pass1.dispatchWorkgroups(Math.ceil(totalVoxels / 64));
        pass1.end();

        // Step 2: Calculate entropy from histogram
        const pass2 = commandEncoder.beginComputePass();
        pass2.setPipeline(this.entropyCalcPipeline); // calculateEntropy
        pass2.setBindGroup(0, bindGroup);
        pass2.dispatchWorkgroups(1); // Single workgroup for reduction
        pass2.end();

        // Read back result
        const stagingBuffer = this.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        // Also read histogram for debugging (512 bins)
        const histStagingBuffer = this.device.createBuffer({
            size: 512 * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, 8);
        commandEncoder.copyBufferToBuffer(this.histogramBuffer, 0, histStagingBuffer, 0, 512 * 4);
        this.device.queue.submit([commandEncoder.finish()]);

        // Read histogram first to debug
        await histStagingBuffer.mapAsync(GPUMapMode.READ);
        const histData = new Uint32Array(histStagingBuffer.getMappedRange());
        let nonZeroBins = 0;
        let maxBin = 0;
        let maxBinIdx = 0;

        // Count by material class (bins 0-511, where bin 511 = total boundary count)
        const materialCounts = new Array(16).fill(0);
        const boundaryTotal = histData[511];

        for (let i = 0; i < 511; i++) {
            if (histData[i] > 0) {
                nonZeroBins++;
                if (histData[i] > maxBin) {
                    maxBin = histData[i];
                    maxBinIdx = i;
                }
                // Categorize by material (bin / 32)
                const materialId = Math.floor(i / 32);
                materialCounts[materialId] += histData[i];
            }
        }
        console.log(`[NeuroaestheticEvaluator] Histogram: ${nonZeroBins} non-zero bins, max bin[${maxBinIdx}]=${maxBin}`);
        console.log(`[NeuroaestheticEvaluator] Boundary voxels: ${boundaryTotal}, materials: [${materialCounts.filter(c => c > 0).join(', ')}]`);
        histStagingBuffer.unmap();
        histStagingBuffer.destroy();

        // Now read entropy result
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Float32Array(stagingBuffer.getMappedRange());
        const entropy = resultData[0];
        const edgeDensity = resultData[1];

        console.log(`  - entropy result: ${entropy.toFixed(4)} bits`);
        console.log(`  - edge density: ${edgeDensity.toFixed(4)}`);

        stagingBuffer.unmap();

        // Cleanup
        uniformBuffer.destroy();
        resultBuffer.destroy();
        stagingBuffer.destroy();
        return entropy;
    }
    /**
     * Calculate entropy via Python backend API
     * Fallback when GPU compute is unavailable
     */
    async calculateEntropyBackend(voxelBuffer) {
        try {
            // For full implementation, would render voxels to image and send to backend
            // For now, use a simplified voxel-based entropy calculation
            // Read voxel data
            const stagingBuffer = this.device.createBuffer({
                size: Math.min(voxelBuffer.size, 65536), // Limit for performance
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(voxelBuffer, 0, stagingBuffer, 0, stagingBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const voxelData = new Float32Array(stagingBuffer.getMappedRange());
            // Calculate Shannon entropy locally (same algorithm as visual_complexity.py)
            const histogram = new Map();
            const stride = 16; // floats per voxel
            const totalSamples = Math.floor(voxelData.length / stride);
            for (let i = 0; i < totalSamples; i++) {
                // Quantize density/material to 256 bins
                const value = Math.floor(Math.abs(voxelData[i * stride]) * 255) % 256;
                histogram.set(value, (histogram.get(value) || 0) + 1);
            }
            // Shannon entropy: H = -Σ p_i * log2(p_i)
            let entropy = 0;
            for (const count of histogram.values()) {
                if (count > 0) {
                    const p = count / totalSamples;
                    entropy -= p * Math.log2(p);
                }
            }
            stagingBuffer.unmap();
            stagingBuffer.destroy();
            console.log(`[NeuroaestheticEvaluator] CPU entropy: ${entropy.toFixed(2)} bits`);
            return entropy;
        }
        catch (error) {
            // NO FALLBACK - propagate error
            throw new Error(`[NeuroaestheticEvaluator] Entropy calculation failed: ${error.message}`);
        }
    }
    /**
     * Calculate color harmony based on material distribution
     * TRL 7: Real algorithm instead of Math.random() placeholder
     *
     * Analyzes color wheel relationships (complementary, analogous, triadic)
     * Returns score 0-1 (1 = perfect harmony)
     */
    async calculateColorHarmony(voxelBuffer) {
        // Check cache
        const now = Date.now();
        if (this.cachedColorHarmony &&
            (now - this.cachedColorHarmony.timestamp) < this.CACHE_TTL_MS) {
            return this.cachedColorHarmony.harmony;
        }

        // Calculate harmony from actual voxel materials
        const harmony = await this.calculateColorHarmonyFromVoxels(voxelBuffer);
        this.cachedColorHarmony = { harmony, timestamp: now };
        return harmony;
    }
    /**
     * Calculate color harmony from actual voxel material distribution
     * Reads material IDs from voxelBuffer and applies HSV color wheel analysis
     */
    async calculateColorHarmonyFromVoxels(voxelBuffer) {
        // MaterialID to HSV color mapping (based on VoxelTypes.js MaterialID enum)
        // Format: [hue (0-360), saturation (0-1), value (0-1)]
        const materialHSV = {
            0: [0, 0, 0.95],       // AIR - nearly white (invisible surfaces)
            1: [30, 0.05, 0.65],   // CONCRETE - neutral gray
            2: [30, 0.45, 0.55],   // WOOD - warm brown
            3: [200, 0.15, 0.85],  // GLASS - light blue tint
            4: [55, 0.08, 0.88],   // INSULATION - pale yellow
            5: [210, 0.10, 0.70],  // STEEL - cool metallic
            6: [15, 0.55, 0.50],   // BRICK - reddish brown
            7: [45, 0.03, 0.92],   // GYPSUM - off-white
            99: [0, 0.8, 0.8],      // HEAT_SOURCE - warm red
        };

        // Use strided sampling across ENTIRE buffer (not just first 65K)
        const VOXEL_STRIDE = 8; // floats per voxel
        const VOXEL_BYTES = VOXEL_STRIDE * 4;
        const totalVoxels = Math.floor(voxelBuffer.size / VOXEL_BYTES);

        // Sample up to 65536 voxels, spread evenly across the entire buffer
        const MAX_SAMPLES = 65536;
        const sampleStep = Math.max(1, Math.floor(totalVoxels / MAX_SAMPLES));
        const actualSamples = Math.min(MAX_SAMPLES, totalVoxels);

        console.log(`[NeuroaestheticEvaluator] Color harmony sampling: ${actualSamples} samples with step ${sampleStep} from ${totalVoxels} voxels`);

        try {
            // Read the entire buffer for strided access
            const stagingBuffer = this.device.createBuffer({
                size: voxelBuffer.size,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            const commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(voxelBuffer, 0, stagingBuffer, 0, voxelBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);

            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const voxelData = new Float32Array(stagingBuffer.getMappedRange());

            // Count material occurrences in SOLID voxels using strided sampling
            const materialCounts = new Map();
            let totalSolidVoxels = 0;
            let sampledCount = 0;
            const debugMaterialSamples = []; // Debug: sample first few solid voxels

            for (let voxelIdx = 0; voxelIdx < totalVoxels && sampledCount < actualSamples; voxelIdx += sampleStep) {
                const state = Math.floor(voxelData[voxelIdx * VOXEL_STRIDE]);
                const materialId = Math.floor(voxelData[voxelIdx * VOXEL_STRIDE + 1]);
                sampledCount++;

                // Only count SOLID voxels (state & 1 != 0)
                if ((state & 1) !== 0) {
                    totalSolidVoxels++;
                    materialCounts.set(materialId, (materialCounts.get(materialId) || 0) + 1);

                    // Debug: sample first 5 solid voxels
                    if (debugMaterialSamples.length < 5) {
                        debugMaterialSamples.push({
                            voxelIdx,
                            state,
                            materialId,
                            rawValues: Array.from(voxelData.slice(voxelIdx * VOXEL_STRIDE, voxelIdx * VOXEL_STRIDE + 4))
                        });
                    }
                }
            }

            // Debug: Log material IDs found
            console.log(`[NeuroaestheticEvaluator] Sampled ${sampledCount} voxels, found ${totalSolidVoxels} SOLID`);
            console.log(`[NeuroaestheticEvaluator] Material IDs found:`, Array.from(materialCounts.entries()).map(([id, count]) => `${id}(${count})`).join(', '));
            if (debugMaterialSamples.length > 0) {
                console.log(`[NeuroaestheticEvaluator] Sample SOLID voxels:`, debugMaterialSamples);
            }

            stagingBuffer.unmap();
            stagingBuffer.destroy();

            if (totalSolidVoxels < 100) {
                console.warn('[NeuroaestheticEvaluator] Too few solid voxels for color harmony');
                return 0.5; // Neutral score for insufficient data
            }

            // Get dominant materials (>5% occurrence)
            const dominantMaterials = [];
            for (const [materialId, count] of materialCounts) {
                const percentage = count / totalSolidVoxels;
                if (percentage > 0.05 && materialHSV[materialId]) {
                    dominantMaterials.push({
                        materialId,
                        percentage,
                        hsv: materialHSV[materialId]
                    });
                }
            }

            console.log(`[NeuroaestheticEvaluator] Dominant materials:`,
                dominantMaterials.map(m => `${m.materialId}(${(m.percentage * 100).toFixed(1)}%)`).join(', '));

            if (dominantMaterials.length < 2) {
                // Monochromatic scheme - not bad, not great
                console.log('[NeuroaestheticEvaluator] Monochromatic color scheme');
                return 0.7;
            }

            // Extract HSV values weighted by prevalence
            const hues = dominantMaterials.map(m => m.hsv[0]);
            const saturations = dominantMaterials.map(m => m.hsv[1]);
            const values = dominantMaterials.map(m => m.hsv[2]);

            // Evaluate harmony schemes
            const harmonyScore = this.evaluateColorHarmonySchemes(hues, saturations, values);
            console.log(`[NeuroaestheticEvaluator] Color harmony score: ${harmonyScore.toFixed(2)}`);

            return harmonyScore;
        } catch (error) {
            console.error('[NeuroaestheticEvaluator] Color harmony calculation failed:', error);
            throw new Error(`Color harmony failed: ${error.message}`);
        }
    }

    /**
     * Evaluate color harmony schemes (analogous, complementary, triadic, etc.)
     */
    evaluateColorHarmonySchemes(hues, saturations, values) {
        const harmonySchemes = {
            'analogous': 30,         // Adjacent colors
            'complementary': 180,    // Opposite colors
            'triadic': 120,          // 3 equidistant colors
            'split_complementary': 150,
            'tetradic': 90
        };

        let bestHarmonyScore = 0;

        for (const [schemeName, expectedAngle] of Object.entries(harmonySchemes)) {
            let schemeScore = 0;
            let comparisons = 0;

            for (let i = 0; i < hues.length; i++) {
                for (let j = i + 1; j < hues.length; j++) {
                    let angleDiff = Math.abs(hues[i] - hues[j]);
                    angleDiff = Math.min(angleDiff, 360 - angleDiff); // Wrap around

                    if (schemeName === 'analogous') {
                        // Adjacent colors score higher when closer
                        if (angleDiff <= expectedAngle) {
                            schemeScore += 1 - (angleDiff / expectedAngle);
                        }
                    } else {
                        // Other schemes need specific angles
                        const deviation = Math.abs(angleDiff - expectedAngle);
                        if (deviation < 30) { // 30° tolerance
                            schemeScore += 1 - (deviation / 30);
                        }
                    }
                    comparisons++;
                }
            }

            if (comparisons > 0) {
                const normalizedScore = schemeScore / comparisons;
                bestHarmonyScore = Math.max(bestHarmonyScore, normalizedScore);
            }
        }

        // Factor in saturation and value consistency
        const satStd = this.standardDeviation(saturations);
        const valStd = this.standardDeviation(values);
        const satConsistency = Math.max(0, 1 - satStd);
        const valConsistency = Math.max(0, 1 - valStd);

        // Combined harmony score (weighted)
        const finalScore = bestHarmonyScore * 0.6 + satConsistency * 0.2 + valConsistency * 0.2;
        return Math.max(0, Math.min(1, finalScore));
    }
    /**
     * Calculate standard deviation
     */
    standardDeviation(values) {
        if (values.length === 0)
            return 0;
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
        return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / values.length);
    }
    /**
     * Целевые значения для оптимизации (только сырые метрики)
     */
    getOptimizationTargets() {
        return {
            fractalDimension: { min: 1.3, target: 1.4, max: 1.5 },
            spatialIntelligibility: { min: 0.7, target: 0.85, max: 1.0 },
            visualEntropy: { min: 4.0, target: 5.0, max: 6.0 },
            melanopicLux: { min: 200, target: 300, max: 500 },
            circadianStimulus: { min: 0.3, target: 0.4, max: 0.5 },
            reverberationBalance: { min: 0.5, target: 0.7, max: 1.0 },
            speechClarity: { min: 0.5, target: 0.8, max: 1.0 }
        };
    }
    /**
     * Преобразование метрик в читаемые инсайты
     */
    generateInsights(metrics) {
        const insights = [];
        // Fractal insights
        if (metrics.fractalCategory === 'optimal') {
            insights.push(`✓ Fractal complexity (D=${metrics.fractalDimension.toFixed(2)}) optimal for stress reduction`);
        }
        else {
            insights.push(`⚠ Fractal complexity ${metrics.fractalCategory} (D=${metrics.fractalDimension.toFixed(2)}), adjust toward 1.4`);
        }
        // Spatial insights
        if (metrics.spatialIntelligibility > this.MIN_INTELLIGIBILITY) {
            insights.push(`✓ Space is intelligible (R²=${metrics.spatialIntelligibility.toFixed(2)}) for easy wayfinding`);
        }
        else {
            insights.push(`⚠ Spatial configuration confusing (R²=${metrics.spatialIntelligibility.toFixed(2)}), simplify layout`);
        }
        // Circadian insights
        if (metrics.melanopicLux > this.MIN_EML) {
            insights.push(`✓ Daylight sufficient (${metrics.melanopicLux.toFixed(0)} EML) for circadian health`);
        }
        else {
            insights.push(`⚠ Insufficient daylight (${metrics.melanopicLux.toFixed(0)} EML), increase glazing`);
        }
        // Acoustic insights
        if (metrics.reverberationBalance > 0.6) {
            insights.push(`✓ Reverberation in comfort range`);
        }
        if (metrics.speechClarity > 0.7) {
            insights.push(`✓ Good speech clarity for communication`);
        }
        return insights;
    }
    /**
     * Нормализация RT60 в диапазон 0-1
     * Используется в evaluate() для reverberationBalance
     */
    calculateRT60Score(rt60) {
        // Оптимальный диапазон: 0.4-0.8s для офиса, 0.8-1.2s для лекций
        const optimalRT60 = 0.6;
        const tolerance = 0.3;
        return Math.max(0, 1 - Math.abs(rt60 - optimalRT60) / tolerance);
    }
    /**
     * Нормализация C50 в диапазон 0-1
     * Используется в evaluate() для speechClarity
     */
    calculateC50Score(c50) {
        // C50 > 0 dB = хорошая разборчивость речи
        if (c50 >= 3)
            return 1.0;
        if (c50 >= 0)
            return 0.8;
        if (c50 >= -3)
            return 0.5;
        return 0.2;
    }
}
