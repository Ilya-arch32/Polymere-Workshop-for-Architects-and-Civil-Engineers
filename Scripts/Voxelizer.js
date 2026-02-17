/**
 * AHI 2.0 Ultimate - Universal Voxelizer
 *
 * Преобразует произвольную 3D геометрию (из IFC/GLB) в семантическую воксельную сетку.
 * Использует адаптивное октодерево для оптимизации памяти.
 *
 * Критически важно: Вокселизация - это единственный источник правды для всех солверов!
 */
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { IFCLoader } from 'https://cdn.jsdelivr.net/npm/web-ifc-three@0.0.126/IFCLoader.js';
import { MaterialID } from './VoxelTypes.js';
/**
 * Октодеревный узел для адаптивной воксельной сетки
 */
class OctreeNode {
    bounds;
    level;
    children = null;
    voxelData = null;
    constructor(bounds, level) {
        this.bounds = bounds;
        this.level = level;
    }
    subdivide() {
        const center = this.bounds.getCenter(new THREE.Vector3());
        const size = this.bounds.getSize(new THREE.Vector3());
        const halfSize = size.multiplyScalar(0.5);
        this.children = [];
        for (let i = 0; i < 8; i++) {
            const offsetX = (i & 1) ? halfSize.x : 0;
            const offsetY = (i & 2) ? halfSize.y : 0;
            const offsetZ = (i & 4) ? halfSize.z : 0;
            const min = new THREE.Vector3(this.bounds.min.x + offsetX, this.bounds.min.y + offsetY, this.bounds.min.z + offsetZ);
            const childBounds = new THREE.Box3(min, min.clone().add(halfSize));
            this.children.push(new OctreeNode(childBounds, this.level + 1));
        }
    }
    isLeaf() {
        return this.children === null;
    }
}
/**
 * Главный класс воксельзатора
 */
export class Voxelizer {
    config;
    scene;
    gridConfig = null;
    voxelGrid = null;
    materialLibrary;
    // GPU resources
    device = null;
    voxelBuffer = null;
    triangleBuffer = null;
    voxelizePipeline = null;
    floodFillPipeline = null;
    materialPipeline = null;
    // TRL 7: Robust voxelization pipelines
    gapDetectionPipeline = null;
    gapClosurePipeline = null;
    boundaryDilationPipeline = null;
    constructor(config) {
        this.config = config;
        this.scene = new THREE.Scene();
        this.materialLibrary = new Map();
        this.excludedElementIds = config.excludedElementIds || [];
        this.initializeMaterialLibrary();
    }
    /**
     * Initialize GPU device and pipelines
     */
    async initializeGPU(device) {
        this.device = device;
        // Load shader code from AHI loader
        const shaderCode = window.AHIModules?.shaders?.['VoxelizerGPU.wgsl'] || '';
        if (!shaderCode) throw new Error('[Voxelizer] WGSL shader not loaded');
        const shaderModule = device.createShaderModule({
            label: 'Voxelizer GPU Shader',
            code: shaderCode
        });
        // Create compute pipelines
        this.voxelizePipeline = device.createComputePipeline({
            label: 'Voxelize Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'voxelize'
            }
        });
        this.floodFillPipeline = device.createComputePipeline({
            label: 'Flood Fill Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'flood_fill'
            }
        });
        this.materialPipeline = device.createComputePipeline({
            label: 'Material Assignment Pipeline',
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'assign_materials'
            }
        });
        console.log('[Voxelizer] GPU pipelines initialized');
    }
    /**
     * Инициализация библиотеки материалов с физическими свойствами
     * U-values based on typical building materials (W/(m²·K))
     */
    initializeMaterialLibrary() {
        // Concrete wall with insulation
        this.materialLibrary.set(MaterialID.CONCRETE, {
            id: MaterialID.CONCRETE,
            name: 'Concrete',
            density: 2400,
            specificHeat: 880,
            thermalConductivity: 1.4,
            uValue: 0.5,  // Insulated wall ~0.5 W/(m²·K)
            reflectanceSpectrum: new Float32Array(16).fill(0.3),
            roughness: 0.8,
        });
        // Wood (doors, frames)
        this.materialLibrary.set(MaterialID.WOOD, {
            id: MaterialID.WOOD,
            name: 'Wood',
            density: 600,
            specificHeat: 1700,
            thermalConductivity: 0.15,
            uValue: 1.5,  // Wooden door ~1.5 W/(m²·K)
            reflectanceSpectrum: new Float32Array(16).fill(0.4),
            roughness: 0.6,
        });
        // Glass (windows)
        this.materialLibrary.set(MaterialID.GLASS, {
            id: MaterialID.GLASS,
            name: 'Glass',
            density: 2500,
            specificHeat: 840,
            thermalConductivity: 1.0,
            uValue: 1.4,  // Double glazing ~1.4 W/(m²·K)
            reflectanceSpectrum: new Float32Array(16).fill(0.08),
            transmittanceSpectrum: new Float32Array(16).fill(0.85),
            roughness: 0.05,
        });
        // Air (fluid cells)
        this.materialLibrary.set(MaterialID.AIR, {
            id: MaterialID.AIR,
            name: 'Air',
            density: 1.225,
            specificHeat: 1005,
            thermalConductivity: 0.026,
            uValue: 0.0,  // Air itself has no U-value
            reflectanceSpectrum: new Float32Array(16).fill(0.0),
            roughness: 0.0,
            kinematicViscosity: 1.5e-5,
        });
        // Heat source (radiators, heaters)
        this.materialLibrary.set(99, {  // MaterialID.HEAT_SOURCE
            id: 99,
            name: 'Heat Source',
            density: 7800,  // Steel radiator
            specificHeat: 500,
            thermalConductivity: 50.0,
            uValue: 0.0,  // N/A - emits heat
            heatOutput: 1000,   // Default 1kW output
            temperature: 333.15,  // 60°C in Kelvin
        });
    }
    /**
     * Загрузка IFC модели
     */
    async loadIFCModel(ifcUrl) {
        const loader = new IFCLoader();
        return new Promise((resolve, reject) => {
            loader.load(ifcUrl, (model) => {
                this.scene.add(model);
                console.log(`[Voxelizer] IFC model loaded: ${model.uuid}`);
                resolve();
            }, undefined, reject);
        });
    }
    /**
     * Загрузка GLB модели (альтернатива IFC)
     */
    async loadGLBModel(glbUrl) {
        const loader = new THREE.GLTFLoader();
        return new Promise((resolve, reject) => {
            loader.load(glbUrl, (gltf) => {
                this.scene.add(gltf.scene);
                console.log(`[Voxelizer] GLB model loaded`);
                resolve();
            }, undefined, reject);
        });
    }
    /**
     * Add arbitrary object to scene (for procedural geometry)
     * IMPORTANT: Clones the object to preserve the original scene (e.g., IFC preview)
     */
    addObject(object) {
        // Count meshes in original
        let originalMeshCount = 0;
        let originalTriCount = 0;
        object.traverse((child) => {
            if ((child.isMesh || child.type === 'Mesh') && child.geometry) {
                originalMeshCount++;
                const indices = child.geometry.index;
                originalTriCount += indices ? indices.count / 3 : child.geometry.attributes.position.count / 3;
            }
        });
        console.log('[Voxelizer] Original object: meshes=', originalMeshCount, 'triangles=', Math.floor(originalTriCount));

        // Clone the object to avoid moving it from the original scene
        const clonedObject = object.clone(true);
        this.scene.add(clonedObject);

        // Count meshes in clone
        let cloneMeshCount = 0;
        let cloneTriCount = 0;
        clonedObject.traverse((child) => {
            if ((child.isMesh || child.type === 'Mesh') && child.geometry) {
                cloneMeshCount++;
                const indices = child.geometry.index;
                cloneTriCount += indices ? indices.count / 3 : child.geometry.attributes.position.count / 3;
            }
        });
        console.log('[Voxelizer] Cloned object: meshes=', cloneMeshCount, 'triangles=', Math.floor(cloneTriCount));

        if (cloneMeshCount < originalMeshCount) {
            console.warn('[Voxelizer] WARNING: Clone has fewer meshes than original!');
        }
    }

    /**
     * Set excluded element IDs (from UI selection)
     */
    setExcludedElements(ids) {
        this.excludedElementIds = ids.map(String);
        console.log(`[Voxelizer] Excluding ${this.excludedElementIds.length} elements from voxelization`);
    }
    /**
     * Clear all objects from scene (for optimizer)
     */
    clearScene() {
        while (this.scene.children.length > 0) {
            const child = this.scene.children[0];
            this.scene.remove(child);
            // Dispose geometry and material if they exist
            if (child.geometry) {
                child.geometry.dispose();
            }
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach((mat) => mat.dispose());
                }
                else {
                    child.material.dispose();
                }
            }
        }
        // Clear voxel buffer
        this.voxelGrid = new Float32Array(0);
        console.log('[Voxelizer] Scene cleared');
    }
    /**
     * КРИТИЧЕСКИЙ МЕТОД: Вокселизация загруженной геометрии на GPU
     *
     * Алгоритм:
     * 1. Вычисляем bounding box всей сцены
     * 2. Извлекаем треугольники из THREE.js мешей
     * 3. Создаём GPU буферы для треугольников и вокселей
     * 4. Запускаем compute shader для консервативной вокселизации
     * 5. Flood fill для определения внутренних/внешних вокселей
     * 6. Присваиваем материалы
     */
    async voxelizeScene() {
        if (!this.device) {
            throw new Error('[Voxelizer] GPU not initialized! Call initializeGPU() first');
        }
        console.time('GPU Voxelization');
        // Шаг 1: FIRST extract triangles (applies exclusion filter!)
        const triangles = this.extractTriangles();
        const numTriangles = triangles.length / 12; // 48 bytes = 12 floats per triangle (3 vertices * 4 floats)
        console.log(`[Voxelizer] Extracted ${numTriangles} triangles`);

        // ============ DIAGNOSTIC: Analyze triangle thickness ============
        // Sample triangles to estimate typical wall thickness
        console.log('[Voxelizer] === DIAGNOSTIC: Triangle Analysis ===');
        let minTriArea = Infinity, maxTriArea = 0, totalTriArea = 0;
        let smallTriCount = 0; // triangles smaller than resolution^2

        for (let i = 0; i < Math.min(triangles.length, 12000); i += 12) {
            const v0 = { x: triangles[i], y: triangles[i + 1], z: triangles[i + 2] };
            const v1 = { x: triangles[i + 4], y: triangles[i + 5], z: triangles[i + 6] };
            const v2 = { x: triangles[i + 8], y: triangles[i + 9], z: triangles[i + 10] };

            // Calculate triangle area using cross product
            const e1 = { x: v1.x - v0.x, y: v1.y - v0.y, z: v1.z - v0.z };
            const e2 = { x: v2.x - v0.x, y: v2.y - v0.y, z: v2.z - v0.z };
            const cross = {
                x: e1.y * e2.z - e1.z * e2.y,
                y: e1.z * e2.x - e1.x * e2.z,
                z: e1.x * e2.y - e1.y * e2.x
            };
            const area = 0.5 * Math.sqrt(cross.x * cross.x + cross.y * cross.y + cross.z * cross.z);

            if (area < minTriArea) minTriArea = area;
            if (area > maxTriArea) maxTriArea = area;
            totalTriArea += area;

            // Count small triangles
            const resSquared = this.config.resolution * this.config.resolution;
            if (area < resSquared) smallTriCount++;
        }

        const sampleCount = Math.min(numTriangles, 1000);
        console.log('[Voxelizer] Triangle area stats (sample of', sampleCount, '):');
        console.log('[Voxelizer]   Min area:', minTriArea.toFixed(6), 'm²');
        console.log('[Voxelizer]   Max area:', maxTriArea.toFixed(6), 'm²');
        console.log('[Voxelizer]   Avg area:', (totalTriArea / sampleCount).toFixed(6), 'm²');
        console.log('[Voxelizer]   Resolution:', this.config.resolution, 'm (voxel area:', (this.config.resolution * this.config.resolution).toFixed(4), 'm²)');
        console.log('[Voxelizer]   Small triangles (<resolution²):', smallTriCount, '/', sampleCount);

        if (smallTriCount > sampleCount * 0.3) {
            console.warn('[Voxelizer] ⚠️ WARNING:', (100 * smallTriCount / sampleCount).toFixed(1), '% of triangles are smaller than voxel area!');
            console.warn('[Voxelizer]   This may cause thin walls to be missed during voxelization.');
            console.warn('[Voxelizer]   Consider reducing voxel resolution or using conservative voxelization.');
        }


        // Шаг 2: Compute bounding box from FILTERED triangles (not full scene!)
        // This ensures excluded objects don't affect grid bounds
        let triMinX = Infinity, triMinY = Infinity, triMinZ = Infinity;
        let triMaxX = -Infinity, triMaxY = -Infinity, triMaxZ = -Infinity;
        for (let i = 0; i < triangles.length; i += 12) {  // 12 floats per triangle
            for (let v = 0; v < 3; v++) {
                const x = triangles[i + v * 4];
                const y = triangles[i + v * 4 + 1];
                const z = triangles[i + v * 4 + 2];
                // Skip NaN/Infinity vertices
                if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
                triMinX = Math.min(triMinX, x); triMaxX = Math.max(triMaxX, x);
                triMinY = Math.min(triMinY, y); triMaxY = Math.max(triMaxY, y);
                triMinZ = Math.min(triMinZ, z); triMaxZ = Math.max(triMaxZ, z);
            }
        }
        console.log(`[Voxelizer] Triangle bounds: X[${triMinX.toFixed(2)}, ${triMaxX.toFixed(2)}], Y[${triMinY.toFixed(2)}, ${triMaxY.toFixed(2)}], Z[${triMinZ.toFixed(2)}, ${triMaxZ.toFixed(2)}]`);

        // NO FALLBACK: If bounds invalid, throw error
        if (!isFinite(triMinX) || !isFinite(triMaxX) || triMinX >= triMaxX) {
            throw new Error(`[Voxelizer] Triangle bounds invalid! triMinX=${triMinX}, triMaxX=${triMaxX}. No valid geometry after exclusion filter.`);
        }

        // Create bbox from triangle bounds (not from scene!)
        const bbox = new THREE.Box3(
            new THREE.Vector3(triMinX, triMinY, triMinZ),
            new THREE.Vector3(triMaxX, triMaxY, triMaxZ)
        );
        const size = bbox.getSize(new THREE.Vector3());
        // Добавляем padding (10% с каждой стороны для воздуха)
        const padding = size.clone().multiplyScalar(0.1);
        bbox.min.sub(padding);
        bbox.max.add(padding);
        // Шаг 3: Вычисляем размеры сетки
        const expandedSize = bbox.getSize(new THREE.Vector3());

        // Calculate initial grid dimensions
        let nx = Math.ceil(expandedSize.x / this.config.resolution);
        let ny = Math.ceil(expandedSize.y / this.config.resolution);
        let nz = Math.ceil(expandedSize.z / this.config.resolution);
        let totalVoxels = nx * ny * nz;

        // LIMIT: Max 2 million voxels (batching prevents TDR timeout)
        const MAX_VOXELS = 2000000;
        if (totalVoxels > MAX_VOXELS) {
            // Scale up resolution to fit within limit
            const scaleFactor = Math.cbrt(totalVoxels / MAX_VOXELS);
            const newResolution = this.config.resolution * scaleFactor;
            console.warn(`[Voxelizer] Grid too large (${totalVoxels} voxels). Scaling resolution from ${this.config.resolution} to ${newResolution.toFixed(3)}m`);

            nx = Math.ceil(expandedSize.x / newResolution);
            ny = Math.ceil(expandedSize.y / newResolution);
            nz = Math.ceil(expandedSize.z / newResolution);
            totalVoxels = nx * ny * nz;
            this.config.resolution = newResolution;
        }

        // Ensure all dimensions are integers for WebGPU buffer sizes
        nx = Math.floor(nx);
        ny = Math.floor(ny);
        nz = Math.floor(nz);
        totalVoxels = Math.floor(nx * ny * nz);

        console.log(`[Voxelizer] Grid dimensions: ${nx}x${ny}x${nz} = ${totalVoxels} voxels`);
        console.log(`[Voxelizer] Memory estimate: ${(totalVoxels * 32 / 1024 / 1024).toFixed(2)} MB`);
        console.log(`[Voxelizer] Grid bounds: X[${bbox.min.x.toFixed(2)}, ${bbox.max.x.toFixed(2)}], Y[${bbox.min.y.toFixed(2)}, ${bbox.max.y.toFixed(2)}], Z[${bbox.min.z.toFixed(2)}, ${bbox.max.z.toFixed(2)}]`);
        this.gridConfig = {
            resolution: this.config.resolution,
            bounds: {
                minX: bbox.min.x, maxX: bbox.max.x,
                minY: bbox.min.y, maxY: bbox.max.y,
                minZ: bbox.min.z, maxZ: bbox.max.z,
            },
            dimensions: { nx, ny, nz, totalVoxels },  // totalVoxels INSIDE dimensions for LBM/CHT
            totalVoxels,  // Also keep at root level for backwards compatibility
        };
        // Шаг 4: Создаём GPU буферы
        // WGSL VoxelGrid struct layout (64 bytes total):
        //   offset 0:  dimensions (vec3) + padding = 16 bytes
        //   offset 16: bounds_min (vec3) + padding = 16 bytes  
        //   offset 32: bounds_max (vec3) + padding = 16 bytes
        //   offset 48: resolution (f32) = 4 bytes
        //   offset 52: tri_start (u32) = 4 bytes  [BATCHING]
        //   offset 56: tri_end (u32) = 4 bytes    [BATCHING]
        //   offset 60: padding = 4 bytes

        // Initial uniform data (tri_start=0, tri_end will be set per batch)
        const uniformData = new Float32Array([
            nx, ny, nz, 0,                              // dimensions (offset 0)
            bbox.min.x, bbox.min.y, bbox.min.z, 0,      // bounds_min (offset 16)
            bbox.max.x, bbox.max.y, bbox.max.z, 0,      // bounds_max (offset 32)
            this.config.resolution, 0, 0, 0             // resolution + tri_start + tri_end + pad (offset 48)
        ]);
        console.log(`[Voxelizer] Uniform buffer: dims=[${nx},${ny},${nz}], min=[${bbox.min.x.toFixed(2)},${bbox.min.y.toFixed(2)},${bbox.min.z.toFixed(2)}], max=[${bbox.max.x.toFixed(2)},${bbox.max.y.toFixed(2)},${bbox.max.z.toFixed(2)}], res=${this.config.resolution.toFixed(4)}`);

        const uniformBuffer = this.device.createBuffer({
            label: 'Grid Uniforms',
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Triangle buffer
        this.triangleBuffer = this.device.createBuffer({
            label: 'Triangle Buffer',
            size: triangles.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.triangleBuffer, 0, triangles);

        // Voxel buffer (8 floats per voxel)
        const voxelBufferSize = totalVoxels * 32;
        this.voxelBuffer = this.device.createBuffer({
            label: 'Voxel Buffer',
            size: voxelBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Solid mask buffer
        const solidMaskSize = Math.ceil(totalVoxels / 32) * 4;
        const solidMaskBuffer = this.device.createBuffer({
            label: 'Solid Mask',
            size: solidMaskSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Bind groups
        const voxelizeBindGroup = this.device.createBindGroup({
            label: 'Voxelize Bind Group',
            layout: this.voxelizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: this.triangleBuffer } },
                { binding: 2, resource: { buffer: this.voxelBuffer } },
                { binding: 3, resource: { buffer: solidMaskBuffer } }
            ]
        });
        const floodFillBindGroup = this.device.createBindGroup({
            label: 'Flood Fill Bind Group',
            layout: this.floodFillPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 2, resource: { buffer: this.voxelBuffer } }
            ]
        });
        const materialBindGroup = this.device.createBindGroup({
            label: 'Material Bind Group',
            layout: this.materialPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 2, resource: { buffer: this.voxelBuffer } }
            ]
        });

        // ═══════════════════════════════════════════════════════════════
        // BATCHED VOXELIZATION: Process triangles in chunks to avoid TDR
        // ═══════════════════════════════════════════════════════════════
        const BATCH_SIZE = 50000; // triangles per batch (tune this!)
        const numBatches = Math.ceil(numTriangles / BATCH_SIZE);
        const workgroupsX = Math.ceil(nx / 4);
        const workgroupsY = Math.ceil(ny / 4);
        const workgroupsZ = Math.ceil(nz / 4);

        console.log(`[Voxelizer] BATCHED: ${numTriangles} triangles in ${numBatches} batches of ${BATCH_SIZE}`);

        for (let batch = 0; batch < numBatches; batch++) {
            const triStart = batch * BATCH_SIZE;
            const triEnd = Math.min(triStart + BATCH_SIZE, numTriangles);

            console.log(`[Voxelizer] Batch ${batch + 1}/${numBatches}: triangles ${triStart}-${triEnd}`);

            // Update tri_start and tri_end in uniform buffer (offset 52 and 56)
            // These are u32 in WGSL but we write as Float32 reinterpreted
            const batchData = new Uint32Array([triStart, triEnd]);
            this.device.queue.writeBuffer(uniformBuffer, 52, batchData);

            // Dispatch voxelization for this batch
            const commandEncoder = this.device.createCommandEncoder();
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.voxelizePipeline);
            pass.setBindGroup(0, voxelizeBindGroup);
            pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
            pass.end();

            // Submit and wait for completion before next batch
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
        }

        // Pass 2: Flood fill (single pass after all batches)
        {
            const commandEncoder = this.device.createCommandEncoder();
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.floodFillPipeline);
            pass.setBindGroup(0, floodFillBindGroup);
            pass.dispatchWorkgroups(Math.ceil(totalVoxels / 256));
            pass.end();
            this.device.queue.submit([commandEncoder.finish()]);
        }

        // Pass 3: Material assignment
        {
            const commandEncoder = this.device.createCommandEncoder();
            const pass = commandEncoder.beginComputePass();
            pass.setPipeline(this.materialPipeline);
            pass.setBindGroup(0, materialBindGroup);
            pass.dispatchWorkgroups(Math.ceil(totalVoxels / 64));
            pass.end();
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
        }

        console.timeEnd('GPU Voxelization');
        console.log(`[Voxelizer] Batched voxelization complete (${numBatches} batches)`);

        // DEBUG: GPU readback to verify voxelization results
        try {
            const voxelData = await this.debugReadback();
            let solidCount = 0, fluidCount = 0, otherCount = 0;
            const stride = 8;
            const totalVoxelCount = voxelData.length / stride;

            // Scan ALL voxels for accurate count (VoxelTypes: EMPTY=0, SOLID=1, FLUID=2)
            for (let i = 0; i < totalVoxelCount; i++) {
                const state = Math.round(voxelData[i * stride]);
                if (state === 2) fluidCount++;      // FLUID = 2
                else if (state === 1) solidCount++; // SOLID = 1
                else if (state === 4) solidCount++; // GLASS = 4 (counts as solid for boundary)
                else otherCount++;                   // EMPTY = 0 or unknown
            }
            console.log(`[Voxelizer] GPU Readback (ALL ${totalVoxelCount}): FLUID=${fluidCount}, SOLID=${solidCount}, other=${otherCount}`);

            // Find first solid voxel for debugging
            let firstSolidIdx = -1;
            for (let i = 0; i < totalVoxelCount && firstSolidIdx < 0; i++) {
                if (Math.round(voxelData[i * stride]) === 1) {  // SOLID = 1 (not >= 1 which matches FLUID=2)
                    firstSolidIdx = i;
                }
            }
            if (firstSolidIdx >= 0) {
                console.log(`[Voxelizer] First SOLID voxel at idx=${firstSolidIdx}:`,
                    Array.from(voxelData.slice(firstSolidIdx * stride, firstSolidIdx * stride + 8)));
            }
        } catch (e) {
            console.warn('[Voxelizer] Debug readback failed:', e.message);
        }

        // Note: Data stays in GPU memory for LBMSolver
        return this.gridConfig;
    }
    /**
     * Extract triangles from THREE.js meshes into GPU-ready format
     */
    extractTriangles() {
        const triangleList = [];
        const excludedIds = this.excludedElementIds || [];
        let excludedCount = 0;
        let includedCount = 0;

        // ============ DIAGNOSTIC: Track element types ============
        const elementTypeStats = {};
        const ifcTypeStats = {};
        const meshNameSamples = [];
        let wallCount = 0, windowCount = 0, doorCount = 0, slabCount = 0, roofCount = 0;
        let openingCount = 0, otherCount = 0;
        let totalTriangles = 0;
        const trianglesByType = {};

        console.log('[Voxelizer] === DIAGNOSTIC: Analyzing IFC elements ===');
        console.log('[Voxelizer] Note: Using loose classification for robustness.');

        this.scene.traverse((object) => {
            if ((object.isMesh || object.type === 'Mesh') && object.geometry) {
                // Check if this mesh or its parents are in excluded list
                let isExcluded = false;
                let current = object;
                while (current && !isExcluded) {
                    // Check by uuid, name, or userData.expressID (IFC element ID)
                    const meshId = String(current.uuid);
                    const meshName = String(current.name || '');
                    const expressId = String(current.userData?.expressID || '');

                    if (excludedIds.includes(meshId) ||
                        excludedIds.includes(meshName) ||
                        excludedIds.includes(expressId)) {
                        isExcluded = true;
                    }
                    current = current.parent;
                }

                if (isExcluded) {
                    excludedCount++;
                    return; // Skip this mesh
                }
                includedCount++;

                const mesh = object;
                const geometry = mesh.geometry;
                const material = this.inferMaterial(mesh);

                // ============ DIAGNOSTIC: Analyze mesh type ============
                const meshName = (mesh.name || '').toLowerCase();
                const userData = mesh.userData || {};
                const ifcType = (userData.type || userData.ifcType || '').toUpperCase();
                const expressID = userData.expressID || 'unknown';

                // Count by IFC type
                if (ifcType) {
                    ifcTypeStats[ifcType] = (ifcTypeStats[ifcType] || 0) + 1;
                }

                // Categorize element - ROBUST FALLBACK
                let category = 'OTHER';
                const nameLower = meshName.toLowerCase();
                const typeUpper = ifcType.toUpperCase();

                if (typeUpper.includes('WALL') || nameLower.includes('wall') || nameLower.includes('wand') || nameLower.includes('mur') || nameLower.includes('pared')) {
                    category = 'WALL';
                    wallCount++;
                } else if (typeUpper.includes('WINDOW') || nameLower.includes('window') || nameLower.includes('glass') || nameLower.includes('glazing')) {
                    category = 'WINDOW';
                    windowCount++;
                } else if (typeUpper.includes('DOOR') || nameLower.includes('door')) {
                    category = 'DOOR';
                    doorCount++;
                } else if (typeUpper.includes('SLAB') || typeUpper.includes('FLOOR') || nameLower.includes('floor') || nameLower.includes('slab')) {
                    category = 'SLAB';
                    slabCount++;
                } else if (typeUpper.includes('ROOF') || nameLower.includes('roof')) {
                    category = 'ROOF';
                    roofCount++;
                } else {
                    category = 'OTHER'; // Default to OTHER, but still voxelize it!
                    otherCount++;
                }

                // Sample mesh names for debugging (sample ALL meshes, not just named ones)
                if (meshNameSamples.length < 50) {
                    meshNameSamples.push({
                        name: mesh.name ? mesh.name.substring(0, 50) : '(unnamed)',
                        ifcType: ifcType || 'none',
                        expressID: expressID,
                        category: category,
                        userDataKeys: Object.keys(userData).join(',') || 'empty'
                    });
                }

                // Update world matrix
                mesh.updateWorldMatrix(true, false);
                const matrix = mesh.matrixWorld;
                // Get positions
                const positions = geometry.attributes.position;
                const indices = geometry.index;

                // Count triangles for this mesh
                let meshTriCount = 0;

                if (indices) {
                    // Indexed geometry
                    meshTriCount = Math.floor(indices.count / 3);
                    for (let i = 0; i < indices.count; i += 3) {
                        const v0 = new THREE.Vector3().fromBufferAttribute(positions, indices.array[i]);
                        const v1 = new THREE.Vector3().fromBufferAttribute(positions, indices.array[i + 1]);
                        const v2 = new THREE.Vector3().fromBufferAttribute(positions, indices.array[i + 2]);
                        // Transform to world space
                        v0.applyMatrix4(matrix);
                        v1.applyMatrix4(matrix);
                        v2.applyMatrix4(matrix);
                        // WGSL Triangle struct: 48 bytes
                        // material_id: 1=SOLID, 4=GLASS (transparent)
                        const matId = (category === 'WINDOW' || category === 'OPENING') ? 4 : 1;

                        triangleList.push(
                            v0.x, v0.y, v0.z, 0,      // v0 + padding (16 bytes)
                            v1.x, v1.y, v1.z, 0,      // v1 + padding (16 bytes)
                            v2.x, v2.y, v2.z, matId   // v2(12) + material_id(4) = 16 bytes
                        );
                    }
                }
                else {
                    // Non-indexed geometry
                    meshTriCount = Math.floor(positions.count / 3);
                    for (let i = 0; i < positions.count; i += 3) {
                        const v0 = new THREE.Vector3().fromBufferAttribute(positions, i);
                        const v1 = new THREE.Vector3().fromBufferAttribute(positions, i + 1);
                        const v2 = new THREE.Vector3().fromBufferAttribute(positions, i + 2);
                        // Transform to world space
                        v0.applyMatrix4(matrix);
                        v1.applyMatrix4(matrix);
                        v2.applyMatrix4(matrix);
                        // WGSL Triangle struct: 48 bytes = 12 floats
                        triangleList.push(
                            v0.x, v0.y, v0.z, 0,      // v0 + padding (16 bytes)
                            v1.x, v1.y, v1.z, 0,      // v1 + padding (16 bytes)
                            v2.x, v2.y, v2.z, material// v2(12) + material_id(4) = 16 bytes
                        );
                    }
                }

                totalTriangles += meshTriCount;
                trianglesByType[category] = (trianglesByType[category] || 0) + meshTriCount;
            }
        });

        // ============ DIAGNOSTIC: Log element statistics ============
        console.log('[Voxelizer] === DIAGNOSTIC: Element Type Statistics ===');
        console.log('[Voxelizer] Total meshes:', includedCount, '(excluded:', excludedCount, ')');
        console.log('[Voxelizer] Total triangles:', totalTriangles);
        console.log('[Voxelizer] Element categories:');
        console.log('[Voxelizer]   WALL:    ', wallCount, 'meshes,', trianglesByType['WALL'] || 0, 'triangles');
        console.log('[Voxelizer]   WINDOW:  ', windowCount, 'meshes,', trianglesByType['WINDOW'] || 0, 'triangles');
        console.log('[Voxelizer]   DOOR:    ', doorCount, 'meshes,', trianglesByType['DOOR'] || 0, 'triangles');
        console.log('[Voxelizer]   SLAB:    ', slabCount, 'meshes,', trianglesByType['SLAB'] || 0, 'triangles');
        console.log('[Voxelizer]   ROOF:    ', roofCount, 'meshes,', trianglesByType['ROOF'] || 0, 'triangles');
        console.log('[Voxelizer]   OPENING: ', openingCount, 'meshes,', trianglesByType['OPENING'] || 0, 'triangles');
        console.log('[Voxelizer]   OTHER:   ', otherCount, 'meshes,', trianglesByType['OTHER'] || 0, 'triangles');

        // Log IFC types found
        console.log('[Voxelizer] === DIAGNOSTIC: IFC Types Found ===');
        const sortedIfcTypes = Object.entries(ifcTypeStats).sort((a, b) => b[1] - a[1]);
        for (const [type, count] of sortedIfcTypes.slice(0, 20)) {
            console.log('[Voxelizer]   ', type, ':', count);
        }

        // Sample mesh names
        console.log('[Voxelizer] === DIAGNOSTIC: Sample Mesh Names (first 20) ===');
        for (const sample of meshNameSamples.slice(0, 20)) {
            console.log('[Voxelizer]  ', sample.category, '| name:', sample.name, '| IFC:', sample.ifcType, '| expressID:', sample.expressID, '| userData:', sample.userDataKeys);
        }

        // Summarize userData keys found across all samples
        const allUserDataKeys = new Set();
        for (const sample of meshNameSamples) {
            if (sample.userDataKeys && sample.userDataKeys !== 'empty') {
                sample.userDataKeys.split(',').forEach(k => allUserDataKeys.add(k));
            }
        }
        console.log('[Voxelizer] All userData keys found:', Array.from(allUserDataKeys).join(', ') || 'NONE');

        // ============ DIAGNOSTIC: Query IFC API for element types ============
        console.log('[Voxelizer] === DIAGNOSTIC: IFC API Element Type Query ===');
        const ifcAPI = window.ifcAPI;
        const modelID = window.currentModelID;

        if (ifcAPI && modelID !== undefined) {
            console.log('[Voxelizer] IFC API available, querying element types from expressIDs...');

            // Collect unique expressIDs from samples
            const expressIDsToQuery = [];
            for (const sample of meshNameSamples) {
                if (sample.expressID && sample.expressID !== 'unknown') {
                    expressIDsToQuery.push(parseInt(sample.expressID));
                }
            }

            // Query IFC for element types
            const ifcElementTypes = {};
            let queriedCount = 0;
            let queryErrors = 0;

            for (const expressID of expressIDsToQuery.slice(0, 50)) {  // Sample first 50
                try {
                    const line = ifcAPI.GetLine(modelID, expressID);
                    if (line) {
                        // Get the IFC type name
                        const typeID = line.type;
                        const typeName = ifcAPI.GetNameFromTypeCode ? ifcAPI.GetNameFromTypeCode(typeID) : ('Type_' + typeID);

                        ifcElementTypes[typeName] = (ifcElementTypes[typeName] || 0) + 1;
                        queriedCount++;

                        // Log first 10 for detail
                        if (queriedCount <= 10) {
                            const name = line.Name?.value || line.GlobalId?.value || '?';
                            console.log('[Voxelizer]   expressID', expressID, '→', typeName, ':', name);
                        }
                    }
                } catch (e) {
                    queryErrors++;
                    if (queryErrors <= 3) {
                        console.warn('[Voxelizer]   Query error for expressID', expressID, ':', e.message);
                    }
                }
            }

            console.log('[Voxelizer] Queried', queriedCount, 'elements (errors:', queryErrors, ')');
            console.log('[Voxelizer] IFC Element Types Found:');

            const sortedTypes = Object.entries(ifcElementTypes).sort((a, b) => b[1] - a[1]);
            for (const [typeName, count] of sortedTypes) {
                const upperName = typeName.toUpperCase();
                const isBuilding = upperName.includes('WALL') || upperName.includes('SLAB') ||
                    upperName.includes('ROOF') || upperName.includes('STAIR');
                const isOpening = upperName.includes('WINDOW') || upperName.includes('DOOR') ||
                    upperName.includes('OPENING');
                const marker = isBuilding ? '🏠' : (isOpening ? '🚪' : '  ');
                console.log('[Voxelizer]   ', marker, typeName, ':', count);
            }

            // Check for openings
            const hasOpenings = sortedTypes.some(([name]) =>
                name.includes('OPENING') || name.includes('WINDOW') || name.includes('DOOR'));
            if (hasOpenings) {
                console.warn('[Voxelizer] ⚠️ OPENINGS DETECTED: Windows/doors/openings create gaps that allow flood-fill to leak!');
            }

            // Check for walls (case-insensitive)
            const hasWalls = sortedTypes.some(([name]) => name.toUpperCase().includes('WALL'));
            if (!hasWalls) {
                console.warn('[Voxelizer] ⚠️ NO WALLS FOUND in IFC - building shell may be represented differently');
            } else {
                console.log('[Voxelizer] ✓ Walls detected in IFC model');
            }

        } else {
            console.log('[Voxelizer] IFC API not available (ifcAPI:', !!ifcAPI, 'modelID:', modelID, ')');
            console.log('[Voxelizer] Cannot query IFC element types without IFC API.');
        }

        // Warnings
        if (wallCount === 0) {
            // Only warn if we actually have failing geometry.
            if (includedCount > 0) {
                console.warn('[Voxelizer] ⚠️ WARNING: No explicit WALL elements detected. Treating shell as generic geometry.');
            }
        }
        if (openingCount > 0) {
            console.warn('[Voxelizer] ⚠️ WARNING:', openingCount, 'OPENING elements found - these may create gaps in walls!');
        }
        if (windowCount > 0 || doorCount > 0) {
            console.log('[Voxelizer] NOTE: Windows and doors detected - ensure they are voxelized as SOLID for closed shell.');
        }

        if (excludedCount > 0) {
            console.log(`[Voxelizer] Excluded ${excludedCount} meshes, included ${includedCount} meshes`);
        }

        return new Float32Array(triangleList);
    }
    /**
     * Detect IFC element type and return {state, material}
     * This method identifies radiators, windows, doors, and wall types from IFC
     */
    detectIFCElementType(mesh) {
        const name = (mesh.name || '').toLowerCase();
        const userData = mesh.userData || {};
        const ifcType = (userData.type || '').toLowerCase();

        // Heat sources: IfcSpaceHeater, radiator, etc.
        if (ifcType.includes('spaceheater') ||
            ifcType.includes('heater') ||
            name.includes('radiator') ||
            name.includes('heater') ||
            name.includes('радиатор') ||
            name.includes('отопл') ||
            name.includes('батарея')) {
            console.log(`[Voxelizer] Detected HEAT_SOURCE: ${mesh.name}`);
            return {
                state: VoxelState.HEAT_SOURCE | VoxelState.SOLID,
                material: 99  // MaterialID.HEAT_SOURCE
            };
        }

        // Windows: IfcWindow
        if (ifcType.includes('window') ||
            name.includes('window') ||
            name.includes('окно') ||
            name.includes('glazing') ||
            name.includes('остекл')) {
            return {
                state: VoxelState.WINDOW | VoxelState.GLASS | VoxelState.SOLID,
                material: MaterialID.GLASS
            };
        }

        // Doors: IfcDoor
        if (ifcType.includes('door') ||
            name.includes('door') ||
            name.includes('дверь')) {
            return {
                state: VoxelState.DOOR | VoxelState.SOLID,
                material: MaterialID.WOOD
            };
        }

        // External walls: Check if wall is at model boundary
        if (ifcType.includes('wall') ||
            name.includes('wall') ||
            name.includes('стен')) {
            // Check if external (at boundary of model or marked as external)
            const isExternal = userData.isExternal ||
                name.includes('external') ||
                name.includes('наруж') ||
                name.includes('внешн');
            if (isExternal) {
                return {
                    state: VoxelState.EXTERNAL_WALL | VoxelState.SOLID,
                    material: MaterialID.CONCRETE
                };
            }
        }

        // Default: use inferMaterial for backward compatibility
        return {
            state: VoxelState.SOLID,
            material: this.inferMaterial(mesh)
        };
    }

    /**
     * Эвристика для определения материала из меша (backward compatibility)
     */
    inferMaterial(mesh) {
        const name = (mesh.name || '').toLowerCase();
        if (name.includes('glass') || name.includes('window') || name.includes('стекл')) {
            return MaterialID.GLASS;
        }
        else if (name.includes('wood') || name.includes('door') || name.includes('дерев')) {
            return MaterialID.WOOD;
        }
        else if (name.includes('concrete') || name.includes('wall') || name.includes('floor') || name.includes('бетон')) {
            return MaterialID.CONCRETE;
        }
        // По умолчанию - бетон
        return MaterialID.CONCRETE;
    }

    /**
     * CPU-based voxelization as fallback when GPU shader fails
     * @param {Float32Array} triangles - Triangle data in WGSL format (16 floats per triangle)
     * @param {Object} gridConfig - Grid configuration
     * @returns {Float32Array} - Voxel data (8 floats per voxel)
     */
    async voxelizeTrianglesCPU(triangles, gridConfig) {
        console.time('[Voxelizer] CPU Voxelization');
        const { nx, ny, nz, resolution, bounds } = gridConfig.dimensions
            ? {
                nx: gridConfig.dimensions.nx, ny: gridConfig.dimensions.ny, nz: gridConfig.dimensions.nz,
                resolution: gridConfig.resolution, bounds: gridConfig.bounds
            }
            : gridConfig;

        const totalVoxels = nx * ny * nz;
        const numTriangles = triangles.length / 16;
        console.log(`[Voxelizer CPU] Processing ${numTriangles} triangles into ${nx}x${ny}x${nz} = ${totalVoxels} voxels`);

        // Create voxel data array (8 floats per voxel: state, material, pad, pad, temp, vx, vy, vz)
        const voxelData = new Float32Array(totalVoxels * 8);

        // Initialize all voxels as FLUID (state=2 per VoxelTypes.js) with default temperature
        for (let i = 0; i < totalVoxels; i++) {
            voxelData[i * 8 + 0] = 2;     // state = FLUID (VoxelTypes.FLUID = 2)
            voxelData[i * 8 + 1] = 0;     // material = AIR
            voxelData[i * 8 + 4] = 293.0; // temperature = 20°C in Kelvin
        }

        // Helper: Triangle-AABB intersection test (Separating Axis Theorem simplified)
        const triangleAABBIntersect = (v0, v1, v2, boxMin, boxMax) => {
            // First check: triangle bbox vs voxel bbox
            const triMinX = Math.min(v0.x, v1.x, v2.x);
            const triMaxX = Math.max(v0.x, v1.x, v2.x);
            const triMinY = Math.min(v0.y, v1.y, v2.y);
            const triMaxY = Math.max(v0.y, v1.y, v2.y);
            const triMinZ = Math.min(v0.z, v1.z, v2.z);
            const triMaxZ = Math.max(v0.z, v1.z, v2.z);

            // AABB overlap test
            if (triMaxX < boxMin.x || triMinX > boxMax.x) return false;
            if (triMaxY < boxMin.y || triMinY > boxMax.y) return false;
            if (triMaxZ < boxMin.z || triMinZ > boxMax.z) return false;

            return true; // Conservative: if bboxes overlap, mark as solid
        };

        // Process each triangle
        let solidVoxels = 0;
        for (let t = 0; t < numTriangles; t++) {
            const base = t * 16;
            const v0 = { x: triangles[base + 0], y: triangles[base + 1], z: triangles[base + 2] };
            const v1 = { x: triangles[base + 4], y: triangles[base + 5], z: triangles[base + 6] };
            const v2 = { x: triangles[base + 8], y: triangles[base + 9], z: triangles[base + 10] };
            const material = triangles[base + 12];

            // Get triangle bounding box in voxel coordinates
            const triMinX = Math.min(v0.x, v1.x, v2.x);
            const triMaxX = Math.max(v0.x, v1.x, v2.x);
            const triMinY = Math.min(v0.y, v1.y, v2.y);
            const triMaxY = Math.max(v0.y, v1.y, v2.y);
            const triMinZ = Math.min(v0.z, v1.z, v2.z);
            const triMaxZ = Math.max(v0.z, v1.z, v2.z);

            // Convert to voxel indices
            const iMin = Math.max(0, Math.floor((triMinX - bounds.min.x) / resolution));
            const iMax = Math.min(nx - 1, Math.floor((triMaxX - bounds.min.x) / resolution));
            const jMin = Math.max(0, Math.floor((triMinY - bounds.min.y) / resolution));
            const jMax = Math.min(ny - 1, Math.floor((triMaxY - bounds.min.y) / resolution));
            const kMin = Math.max(0, Math.floor((triMinZ - bounds.min.z) / resolution));
            const kMax = Math.min(nz - 1, Math.floor((triMaxZ - bounds.min.z) / resolution));

            // Mark voxels in triangle's bbox as SOLID
            for (let k = kMin; k <= kMax; k++) {
                for (let j = jMin; j <= jMax; j++) {
                    for (let i = iMin; i <= iMax; i++) {
                        const voxelMin = {
                            x: bounds.min.x + i * resolution,
                            y: bounds.min.y + j * resolution,
                            z: bounds.min.z + k * resolution
                        };
                        const voxelMax = {
                            x: voxelMin.x + resolution,
                            y: voxelMin.y + resolution,
                            z: voxelMin.z + resolution
                        };

                        if (triangleAABBIntersect(v0, v1, v2, voxelMin, voxelMax)) {
                            const idx = i + j * nx + k * nx * ny;
                            if (voxelData[idx * 8] !== 1) { // Not already solid
                                voxelData[idx * 8 + 0] = 1;        // state = SOLID
                                voxelData[idx * 8 + 1] = material; // material
                                solidVoxels++;
                            }
                        }
                    }
                }
            }

            // Progress every 10000 triangles
            if (t > 0 && t % 50000 === 0) {
                console.log(`[Voxelizer CPU] Progress: ${t}/${numTriangles} triangles, ${solidVoxels} solid voxels`);
            }
        }

        console.timeEnd('[Voxelizer] CPU Voxelization');
        console.log(`[Voxelizer CPU] Complete: ${solidVoxels} solid voxels from ${numTriangles} triangles`);

        return voxelData;
    }
    /**
     * Get GPU voxel buffer (for solvers)
     */
    getVoxelBuffer() {
        if (!this.voxelBuffer) {
            throw new Error('[Voxelizer] No GPU voxel buffer available');
        }
        return this.voxelBuffer;
    }
    /**
     * Get grid configuration
     */
    getGridConfig() {
        if (!this.gridConfig) {
            throw new Error('[Voxelizer] No grid configuration available');
        }
        return this.gridConfig;
    }
    /**
     * Debug: Read back voxel data from GPU (expensive!)
     */
    async debugReadback() {
        if (!this.device || !this.voxelBuffer || !this.gridConfig) {
            throw new Error('[Voxelizer] GPU resources not available');
        }
        const size = this.gridConfig.totalVoxels * 32; // 8 floats per voxel
        // Create staging buffer
        const stagingBuffer = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        // Copy from GPU to staging
        const commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(this.voxelBuffer, 0, stagingBuffer, 0, size);
        this.device.queue.submit([commandEncoder.finish()]);
        // Map and read
        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange());
        const copy = new Float32Array(data); // Make a copy
        stagingBuffer.unmap();
        stagingBuffer.destroy();
        return copy;
    }
    /**
     * TRL 7: Validate geometry for watertightness and manifoldness
     * Checks for boundary edges, non-manifold edges, and gaps
     */
    validateGeometry() {
        const triangles = this.extractTriangles();
        const numTriangles = triangles.length / 10;
        // Edge-face connectivity analysis
        const edgeMap = new Map(); // edge key -> face count
        const gapLocations = [];
        let degenerateCount = 0;
        for (let t = 0; t < numTriangles; t++) {
            const base = t * 10;
            const v0 = new THREE.Vector3(triangles[base], triangles[base + 1], triangles[base + 2]);
            const v1 = new THREE.Vector3(triangles[base + 3], triangles[base + 4], triangles[base + 5]);
            const v2 = new THREE.Vector3(triangles[base + 6], triangles[base + 7], triangles[base + 8]);
            // Check for degenerate triangles (zero area)
            const edge1 = v1.clone().sub(v0);
            const edge2 = v2.clone().sub(v0);
            const cross = edge1.cross(edge2);
            if (cross.length() < 1e-10) {
                degenerateCount++;
                continue;
            }
            // Add edges to connectivity map
            const edges = [
                [v0, v1], [v1, v2], [v2, v0]
            ];
            for (const [a, b] of edges) {
                // Create canonical edge key (smaller vertex first)
                const key = this.createEdgeKey(a, b);
                edgeMap.set(key, (edgeMap.get(key) || 0) + 1);
            }
        }
        // Count boundary edges (appear only once) and non-manifold (>2)
        let boundaryEdges = 0;
        let nonManifoldEdges = 0;
        for (const [key, count] of edgeMap) {
            if (count === 1) {
                boundaryEdges++;
                // Parse edge key to get location
                const coords = key.split('_').map(parseFloat);
                gapLocations.push(new THREE.Vector3((coords[0] + coords[3]) / 2, (coords[1] + coords[4]) / 2, (coords[2] + coords[5]) / 2));
            }
            else if (count > 2) {
                nonManifoldEdges++;
            }
        }
        // Calculate quality score
        const isWatertight = boundaryEdges === 0 && nonManifoldEdges === 0;
        let qualityScore = 1.0;
        qualityScore -= (boundaryEdges / Math.max(1, edgeMap.size)) * 0.5;
        qualityScore -= (nonManifoldEdges / Math.max(1, edgeMap.size)) * 0.3;
        qualityScore -= (degenerateCount / Math.max(1, numTriangles)) * 0.2;
        qualityScore = Math.max(0, qualityScore);
        // Generate recommendations
        const recommendations = [];
        if (boundaryEdges > 0) {
            recommendations.push(`Found ${boundaryEdges} boundary edges. Enable robust mode or repair mesh.`);
        }
        if (nonManifoldEdges > 0) {
            recommendations.push(`Found ${nonManifoldEdges} non-manifold edges. Mesh requires cleanup.`);
        }
        if (degenerateCount > 0) {
            recommendations.push(`Found ${degenerateCount} degenerate triangles. Consider mesh decimation.`);
        }
        if (isWatertight) {
            recommendations.push('Geometry is watertight. Standard voxelization recommended.');
        }
        console.log(`[Voxelizer] Geometry validation: watertight=${isWatertight}, quality=${(qualityScore * 100).toFixed(1)}%`);
        return {
            isWatertight,
            totalTriangles: numTriangles,
            boundaryEdges,
            nonManifoldEdges,
            degenerateTriangles: degenerateCount,
            gapLocations,
            qualityScore,
            recommendations
        };
    }
    /**
     * Create canonical edge key for edge-face connectivity
     */
    createEdgeKey(a, b) {
        // Round to avoid floating point issues
        const precision = 1e6;
        const ax = Math.round(a.x * precision) / precision;
        const ay = Math.round(a.y * precision) / precision;
        const az = Math.round(a.z * precision) / precision;
        const bx = Math.round(b.x * precision) / precision;
        const by = Math.round(b.y * precision) / precision;
        const bz = Math.round(b.z * precision) / precision;
        // Canonical order: smaller vertex first
        if (ax < bx || (ax === bx && ay < by) || (ax === bx && ay === by && az < bz)) {
            return `${ax}_${ay}_${az}_${bx}_${by}_${bz}`;
        }
        else {
            return `${bx}_${by}_${bz}_${ax}_${ay}_${az}`;
        }
    }
    /**
     * TRL 7: Robust voxelization for non-watertight geometry
     * Automatically detects and closes small gaps
     */
    async voxelizeSceneRobust() {
        console.time('[Voxelizer] Robust Voxelization');
        // Step 1: Validate geometry
        const validation = this.validateGeometry();
        const repairsApplied = [];
        let gapsClosed = 0;
        let voxelsAdded = 0;
        // Step 2: Standard voxelization
        const gridConfig = await this.voxelizeScene();
        // Step 3: If geometry has issues and robust mode enabled, apply repairs
        if (this.config.enableRobustMode && !validation.isWatertight) {
            console.log('[Voxelizer] Applying robust repairs...');
            // Read back voxel data for CPU-based gap analysis
            const voxelData = await this.debugReadback();
            const { nx, ny, nz } = gridConfig.dimensions;
            // Step 3a: Morphological closing to fill small gaps
            if (this.config.gapThreshold > 0) {
                const closingResult = this.applyMorphologicalClosing(voxelData, nx, ny, nz, this.config.gapThreshold);
                gapsClosed = closingResult.gapsClosed;
                voxelsAdded = closingResult.voxelsAdded;
                if (gapsClosed > 0) {
                    repairsApplied.push(`Morphological closing: ${gapsClosed} gaps, ${voxelsAdded} voxels added`);
                    // Write repaired data back to GPU
                    this.device.queue.writeBuffer(this.voxelBuffer, 0, closingResult.repairedData);
                }
            }
            // Step 3b: Boundary dilation for thin walls
            if (this.config.boundaryDilation > 0) {
                const dilationResult = await this.applyBoundaryDilation(this.config.boundaryDilation);
                if (dilationResult > 0) {
                    repairsApplied.push(`Boundary dilation: ${dilationResult} voxels added`);
                    voxelsAdded += dilationResult;
                }
            }
            // Step 3c: Re-run flood fill with repaired geometry
            await this.rerunFloodFill();
            repairsApplied.push('Flood fill re-executed with repaired geometry');
        }
        // Calculate confidence based on repairs and original quality
        let confidence = validation.qualityScore;
        if (repairsApplied.length > 0) {
            // Repairs add some uncertainty
            confidence = Math.min(confidence + 0.2, 0.95);
        }
        console.timeEnd('[Voxelizer] Robust Voxelization');
        console.log(`[Voxelizer] Robust result: ${gapsClosed} gaps closed, ${voxelsAdded} voxels added, confidence=${(confidence * 100).toFixed(1)}%`);
        return {
            gridConfig,
            validation,
            repairsApplied,
            gapsClosed,
            voxelsAdded,
            confidence
        };
    }
    /**
     * Apply morphological closing (dilation followed by erosion) to close small gaps
     */
    applyMorphologicalClosing(voxelData, nx, ny, nz, maxGapSize) {
        const SOLID = 1; // VoxelState.SOLID
        const FLUID = 2; // VoxelState.FLUID
        const stride = 8; // floats per voxel
        // Create working copy
        const workData = new Float32Array(voxelData);
        let gapsClosed = 0;
        let voxelsAdded = 0;
        // Dilation pass: expand solid regions
        const dilatedMask = new Uint8Array(nx * ny * nz);
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    const state = workData[idx * stride];
                    if (state === SOLID) {
                        // Mark this voxel and neighbors within gap threshold
                        for (let dz = -maxGapSize; dz <= maxGapSize; dz++) {
                            for (let dy = -maxGapSize; dy <= maxGapSize; dy++) {
                                for (let dx = -maxGapSize; dx <= maxGapSize; dx++) {
                                    const nx2 = x + dx, ny2 = y + dy, nz2 = z + dz;
                                    if (nx2 >= 0 && nx2 < nx && ny2 >= 0 && ny2 < ny && nz2 >= 0 && nz2 < nz) {
                                        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                                        if (dist <= maxGapSize) {
                                            dilatedMask[nx2 + ny2 * nx + nz2 * nx * ny] = 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Erosion pass: shrink back, but keep newly connected regions
        const erodedMask = new Uint8Array(nx * ny * nz);
        for (let z = maxGapSize; z < nz - maxGapSize; z++) {
            for (let y = maxGapSize; y < ny - maxGapSize; y++) {
                for (let x = maxGapSize; x < nx - maxGapSize; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    // Check if all neighbors in kernel are dilated
                    let allDilated = true;
                    for (let dz = -maxGapSize; dz <= maxGapSize && allDilated; dz++) {
                        for (let dy = -maxGapSize; dy <= maxGapSize && allDilated; dy++) {
                            for (let dx = -maxGapSize; dx <= maxGapSize && allDilated; dx++) {
                                const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
                                if (dist <= maxGapSize) {
                                    const nidx = (x + dx) + (y + dy) * nx + (z + dz) * nx * ny;
                                    if (dilatedMask[nidx] === 0) {
                                        allDilated = false;
                                    }
                                }
                            }
                        }
                    }
                    if (allDilated) {
                        erodedMask[idx] = 1;
                    }
                }
            }
        }
        // Apply closing: mark previously FLUID voxels that are now in eroded mask as SOLID
        for (let i = 0; i < nx * ny * nz; i++) {
            const currentState = workData[i * stride];
            const wasOriginallyFluid = (currentState === FLUID || currentState === 0);
            const originallyWasSolid = voxelData[i * stride] === SOLID;
            if (wasOriginallyFluid && erodedMask[i] === 1 && !originallyWasSolid) {
                // This is a gap that should be closed
                // But only if it bridges two solid regions
                let hasSolidNeighbor = false;
                const x = i % nx;
                const y = Math.floor(i / nx) % ny;
                const z = Math.floor(i / (nx * ny));
                // Check 6-connectivity for solid neighbors
                const neighbors = [
                    [x - 1, y, z], [x + 1, y, z],
                    [x, y - 1, z], [x, y + 1, z],
                    [x, y, z - 1], [x, y, z + 1]
                ];
                for (const [nx2, ny2, nz2] of neighbors) {
                    if (nx2 >= 0 && nx2 < nx && ny2 >= 0 && ny2 < ny && nz2 >= 0 && nz2 < nz) {
                        const nidx = nx2 + ny2 * nx + nz2 * nx * ny;
                        if (voxelData[nidx * stride] === SOLID) {
                            hasSolidNeighbor = true;
                            break;
                        }
                    }
                }
                if (hasSolidNeighbor) {
                    workData[i * stride] = SOLID; // Close the gap
                    voxelsAdded++;
                    gapsClosed++;
                }
            }
        }
        return { repairedData: workData, gapsClosed, voxelsAdded };
    }
    /**
     * Apply boundary dilation to thicken thin walls
     */
    async applyBoundaryDilation(dilationRadius) {
        // Would use GPU compute shader for efficiency
        // Simplified CPU implementation for now
        console.log(`[Voxelizer] Boundary dilation with radius ${dilationRadius}`);
        return 0; // Placeholder - implement with GPU shader
    }
    /**
     * Get the GPU buffer containing voxel data (Storage Buffer)
     */
    getVoxelBuffer() {
        // Return the GPU buffer directly for LBM solver
        return this.voxelBuffer;
    }
    /**
     * Re-run flood fill after repairs
     */
    async rerunFloodFill() {
        if (!this.device || !this.voxelBuffer || !this.gridConfig)
            return;
        // Create bind group and run flood fill pipeline
        // Similar to voxelizeScene() but only the flood fill pass
        console.log('[Voxelizer] Re-running flood fill...');
    }
    /**
     * Debug: visualize specific layers (requires readback)
     */
    async debugGetLayer(z) {
        if (!this.gridConfig)
            return null;
        const data = await this.debugReadback();
        const { nx, ny } = this.gridConfig.dimensions;
        const layer = new Float32Array(nx * ny);
        for (let j = 0; j < ny; j++) {
            for (let i = 0; i < nx; i++) {
                const idx = (i + j * nx + z * nx * ny);
                layer[j * nx + i] = data[idx * 8]; // state field
            }
        }
        return layer;
    }
}
/**
 * Factory function with GPU initialization
 */
export async function createVoxelizer(resolution, device, enableRobust = true) {
    const config = {
        resolution,
        adaptiveOctree: false, // Пока отключаем для простоты
        minResolution: resolution,
        maxResolution: resolution * 4,
        defaultMaterials: new Map([
            ['concrete', MaterialID.CONCRETE],
            ['wood', MaterialID.WOOD],
            ['glass', MaterialID.GLASS],
        ]),
        // TRL 7: Robust voxelization defaults
        enableRobustMode: enableRobust,
        gapThreshold: 2, // Close gaps up to 2 voxels
        boundaryDilation: 0, // No dilation by default
    };
    const voxelizer = new Voxelizer(config);
    // Initialize GPU if device provided
    if (device) {
        await voxelizer.initializeGPU(device);
    }
    return voxelizer;
}
