
/**
 * SimulationVisualizer.js
 * Handles 3D visualization of simulation results (CFD, Thermal, Lighting) in the IFC 3D Viewer.
 */

export class SimulationVisualizer {
    constructor(scene, camera) {
        this.scene = scene;
        this.camera = camera;
        this.activeVisualizations = {
            airflow: null,
            thermal: null,
            lighting: null,
            acoustics: null
        };

        // Color ramps
        this.thermalRamp = this._createThermalRamp();
        this.airflowRamp = this._createAirflowRamp();
    }

    _createThermalRamp() {
        return [
            { t: 0.0, c: [0, 0, 1] },      // Blue (Cold)
            { t: 0.25, c: [0, 1, 1] },     // Cyan
            { t: 0.5, c: [0, 1, 0] },      // Green
            { t: 0.75, c: [1, 1, 0] },     // Yellow
            { t: 1.0, c: [1, 0, 0] }       // Red (Hot)
        ];
    }

    _createAirflowRamp() {
        return [
            { t: 0.0, c: [0, 0, 0.5] },    // Dark Blue (Stagnant)
            { t: 0.2, c: [0, 0.5, 1] },    // Blue
            { t: 0.5, c: [0.8, 0.8, 0.8] },// White-ish (Comfortable breeze)
            { t: 0.8, c: [1, 0.5, 0] },    // Orange
            { t: 1.0, c: [1, 0, 0] }       // Red (High velocity)
        ];
    }

    _getColor(value, min, max, ramp) {
        if (value < min) value = min;
        if (value > max) value = max;
        const t = (value - min) / (max - min);

        for (let i = 0; i < ramp.length - 1; i++) {
            if (t >= ramp[i].t && t <= ramp[i + 1].t) {
                const localT = (t - ramp[i].t) / (ramp[i + 1].t - ramp[i].t);
                const c1 = ramp[i].c;
                const c2 = ramp[i + 1].c;
                return [
                    c1[0] + (c2[0] - c1[0]) * localT,
                    c1[1] + (c2[1] - c1[1]) * localT,
                    c1[2] + (c2[2] - c1[2]) * localT
                ];
            }
        }
        return ramp[ramp.length - 1].c;
    }

    /**
     * Clear all active visualizations
     */
    clearAll() {
        Object.keys(this.activeVisualizations).forEach(key => {
            this.clear(key);
        });
    }

    /**
     * Clear a specific visualization type
     */
    clear(type) {
        if (this.activeVisualizations[type]) {
            this.scene.remove(this.activeVisualizations[type]);
            // Dispose geometry and materials
            if (this.activeVisualizations[type].geometry) this.activeVisualizations[type].geometry.dispose();
            if (this.activeVisualizations[type].material) this.activeVisualizations[type].material.dispose();
            this.activeVisualizations[type] = null;
        }
    }

    /**
     * Visualize Airflow (Velocity Field)
     * @param {Float32Array} velocityData - [vx, vy, vz, ...] flattened array
     * @param {Object} gridConfig - { dimensions: {nx, ny, nz}, voxelSize }
     */
    visualizeAirflow(velocityData, gridConfig) {
        this.clear('airflow');
        console.log('Visualizing Airflow...');

        const { nx, ny, nz } = gridConfig.dimensions;
        const vs = gridConfig.voxelSize;
        const bbox = gridConfig.bbox; // { min: {x,y,z}, max: {x,y,z} }

        const vertices = [];
        const colors = [];
        const sizes = [];

        // Sample every Nth voxel to avoid clutter
        const step = 2;

        for (let x = 0; x < nx; x += step) {
            for (let y = 0; y < ny; y += step) {
                for (let z = 0; z < nz; z += step) {
                    const idx = (z * nx * ny) + (y * nx) + x;
                    const vIdx = idx * 3; // 3 components per voxel

                    const vx = velocityData[vIdx];
                    const vy = velocityData[vIdx + 1];
                    const vz = velocityData[vIdx + 2];

                    if (isNaN(vx)) continue;

                    const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);

                    // Threshold to hide essentially stagnant air
                    if (speed < 0.05) continue;

                    // World Position calculation
                    // Center of voxel
                    const wx = bbox.min.x + (x + 0.5) * vs;
                    const wy = bbox.min.y + (y + 0.5) * vs;
                    const wz = bbox.min.z + (z + 0.5) * vs;

                    vertices.push(wx, wy, wz);

                    // Color based on speed (0 to 3 m/s)
                    const rgb = this._getColor(speed, 0, 3.0, this.airflowRamp);
                    colors.push(rgb[0], rgb[1], rgb[2]);

                    // Size based on speed
                    sizes.push(Math.min(0.5, speed * 0.2));
                }
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        // Note: size attenuation requires custom shader or PointsMaterial size, 
        // effectively mostly uniform size for simple implementation

        // Using points as "particles"
        const material = new THREE.PointsMaterial({
            size: vs * 0.8,
            vertexColors: true,
            transparent: true,
            opacity: 0.7,
            sizeAttenuation: true
        });

        const points = new THREE.Points(geometry, material);
        points.name = 'AHI_Airflow_Viz';

        this.scene.add(points);
        this.activeVisualizations.airflow = points;
        console.log(`Added ${vertices.length / 3} airflow particles`);
    }

    /**
     * Visualize Thermal Map
     * @param {Float32Array} tempData - Temperature in Kelvin per voxel
     * @param {Object} gridConfig
     */
    visualizeThermal(tempData, gridConfig) {
        this.clear('thermal');
        console.log('Visualizing Thermal Map...');

        const { nx, ny, nz } = gridConfig.dimensions;
        const vs = gridConfig.voxelSize;
        const bbox = gridConfig.bbox;

        const vertices = [];
        const colors = [];

        // Determine range for coloring (auto-scale slightly)
        // Assume simplified range for HVAC: 18C to 30C (291K to 303K)
        const minTemp = 291.15;
        const maxTemp = 303.15;

        // Sampling step
        const step = 2;

        for (let idx = 0; idx < tempData.length; idx++) {
            // Reconstruct coords
            // Optimization: Iterate by coords to support stepping
            // But actually tempData is flat. Let's iterate coords.
        }

        for (let z = 0; z < nz; z += step) {
            for (let y = 0; y < ny; y += step) {
                for (let x = 0; x < nx; x += step) {
                    const idx = (z * nx * ny) + (y * nx) + x;
                    const tempK = tempData[idx];

                    // Filter invalid/empty
                    if (tempK <= 0 || isNaN(tempK)) continue;

                    // Filter "ambient" or boundary if needed, but here we show everything with valid temp
                    // Check range to avoid noise
                    if (tempK < 250 || tempK > 350) continue;

                    const wx = bbox.min.x + (x + 0.5) * vs;
                    const wy = bbox.min.y + (y + 0.5) * vs;
                    const wz = bbox.min.z + (z + 0.5) * vs;

                    vertices.push(wx, wy, wz);

                    // Normalize color
                    const rgb = this._getColor(tempK, minTemp, maxTemp, this.thermalRamp);
                    colors.push(rgb[0], rgb[1], rgb[2]);
                }
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: vs * 1.5, // Overlap slightly to look like a volume
            vertexColors: true,
            transparent: true,
            opacity: 0.3, // Semi-transparent "fog"
            sizeAttenuation: true,
            depthWrite: false, // Better blending
            blending: THREE.AdditiveBlending
        });

        const cloud = new THREE.Points(geometry, material);
        cloud.name = 'AHI_Thermal_Viz';

        this.scene.add(cloud);
        this.activeVisualizations.thermal = cloud;
        console.log(`Added ${vertices.length / 3} thermal voxels`);
    }

    /**
     * Visualize Solar/Lighting
     * @param {Object} lightingData - Could be ray paths or voxel illuminance
     * For now, simulating "Sun Rays" direction
     */
    visualizeLighting(sunPos, gridConfig) {
        this.clear('lighting');
        console.log('Visualizing Lighting...');

        // Create lines representing sun rays coming into the bounding box
        const { azimuth, elevation } = sunPos;
        // Convert to radians
        const azRad = (azimuth - 180) * Math.PI / 180; // Standardize direction
        const elRad = elevation * Math.PI / 180;

        // Sun vector (pointing TO sun)
        const sunDir = new THREE.Vector3(
            Math.sin(azRad) * Math.cos(elRad),
            Math.sin(elRad),
            Math.cos(azRad) * Math.cos(elRad)
        ).normalize();

        // Rays coming FROM sun = -sunDir
        const rayDir = sunDir.clone().negate();

        // Create random rays entering the bounding box from top
        const bbox = gridConfig.bbox;
        const width = bbox.max.x - bbox.min.x;
        const depth = bbox.max.z - bbox.min.z;
        const height = bbox.max.y - bbox.min.y;

        const vertices = [];

        // Generate 100 rays
        for (let i = 0; i < 100; i++) {
            // Random start point above the building
            const sx = bbox.min.x + Math.random() * width;
            const sz = bbox.min.z + Math.random() * depth;
            const sy = bbox.max.y + 2.0; // Start 2m above max height

            // End point (ray length 20m)
            const rayLen = height + 5.0;
            const ex = sx + rayDir.x * rayLen;
            const ey = sy + rayDir.y * rayLen;
            const ez = sz + rayDir.z * rayLen;

            vertices.push(sx, sy, sz);
            vertices.push(ex, ey, ez);
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));

        const material = new THREE.LineBasicMaterial({
            color: 0xffdd00, // Gold/Yellow
            transparent: true,
            opacity: 0.4,
            linewidth: 1
        });

        const lines = new THREE.LineSegments(geometry, material);
        lines.name = 'AHI_Lighting_Viz';

        this.scene.add(lines);
        this.activeVisualizations.lighting = lines;
        console.log('Added Simulated Sun Rays');
    }

    // Visualize Acoustics - Animated expanding wavefronts from source
    // @param {Float32Array} rirData - Room Impulse Response data
    // @param {Object} sourcePos - { x, y, z } source position in voxel coordinates
    // @param {Object} gridConfig - Grid configuration
    visualizeAcoustics(rirData, sourcePos, gridConfig) {
        // Clear previous
        if (this.activeVisualizations.acoustics) {
            this.scene.remove(this.activeVisualizations.acoustics);
            this.activeVisualizations.acoustics = null;
        }

        if (!sourcePos || !gridConfig) {
            console.warn('[SimulationVisualizer] Missing sourcePos or gridConfig for acoustics');
            return;
        }

        const { nx, ny, nz } = gridConfig.dimensions;
        const voxelSize = gridConfig.voxelSize || gridConfig.resolution || 0.5;

        // Convert voxel position to world coordinates
        const centerX = (sourcePos.x - nx / 2) * voxelSize;
        const centerY = (sourcePos.z - nz / 2) * voxelSize; // Z becomes Y in Three.js
        const centerZ = (sourcePos.y - ny / 2) * voxelSize;

        // Create group for all wavefront spheres
        const group = new THREE.Group();
        group.name = 'AHI_Acoustics_Viz';

        // Create multiple concentric wavefront rings
        const numWaves = 5;
        const maxRadius = Math.min(nx, ny, nz) * voxelSize * 0.4;

        for (let i = 0; i < numWaves; i++) {
            const radius = maxRadius * (i + 1) / numWaves;
            const geometry = new THREE.RingGeometry(radius * 0.95, radius, 64);
            
            // Rotate ring to lie in XZ plane (horizontal)
            geometry.rotateX(-Math.PI / 2);
            
            const material = new THREE.MeshBasicMaterial({
                color: 0x00aaff,
                transparent: true,
                opacity: 0.3 - (i * 0.05),
                side: THREE.DoubleSide
            });

            const ring = new THREE.Mesh(geometry, material);
            ring.position.set(centerX, centerY, centerZ);
            group.add(ring);
        }

        // Add source point indicator
        const sourceGeom = new THREE.SphereGeometry(voxelSize * 0.5, 16, 16);
        const sourceMat = new THREE.MeshBasicMaterial({
            color: 0xff3300,
            transparent: true,
            opacity: 0.8
        });
        const sourceMarker = new THREE.Mesh(sourceGeom, sourceMat);
        sourceMarker.position.set(centerX, centerY, centerZ);
        group.add(sourceMarker);

        // Animation function for expanding wavefronts
        let animationFrame = 0;
        const animateWaves = () => {
            if (!this.activeVisualizations.acoustics) return;

            animationFrame++;
            const t = (animationFrame % 100) / 100; // 0-1 over 100 frames

            group.children.forEach((child, idx) => {
                if (child.geometry instanceof THREE.RingGeometry) {
                    const baseRadius = maxRadius * (idx + 1) / numWaves;
                    const pulseOffset = (t + idx * 0.2) % 1;
                    const scale = 0.8 + pulseOffset * 0.4;
                    child.scale.set(scale, scale, 1);
                    child.material.opacity = 0.3 * (1 - pulseOffset);
                }
            });

            requestAnimationFrame(animateWaves);
        };

        this.scene.add(group);
        this.activeVisualizations.acoustics = group;
        
        // Start animation
        animateWaves();
        
        console.log(`[SimulationVisualizer] Added Acoustics visualization at (${centerX.toFixed(1)}, ${centerY.toFixed(1)}, ${centerZ.toFixed(1)})`);
    }
}
