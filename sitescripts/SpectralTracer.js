/**
 * AHI 2.0 - Professional Daylight Analysis Module
 *
 * Computes interior illuminance using industry-standard analytical methods:
 * - CIE Standard Sky Models (Clear, Intermediate, Overcast)
 * - Split-Flux Method for Daylight Factor (BRE Digest 310)
 * - Lumen Method with inter-reflections
 * - EPW weather data integration
 *
 * Outputs:
 * - Daylight Factor (DF) per CIE/ISO standards
 * - Interior Illuminance (lux)
 * - Uniformity Ratio (Emin/Eavg)
 * - Equivalent Melanopic Lux (EML) per CIE S 026
 * - Circadian Stimulus (CS) per Rea et al.
 *
 * References:
 * - CIE 110-1994: Spatial Distribution of Daylight - CIE Standard Overcast Sky
 * - CIE S 011/E:2003: Spatial Distribution of Daylight - CIE Standard General Sky
 * - BRE Digest 310: Estimating daylight in buildings
 * - CIBSE Lighting Guide LG10: Daylighting
 */
import { BVHBuilder } from './BVHBuilder.js';

export class SpectralTracer {
    device;
    gridConfig;
    config;

    // GPU Resources (placeholders for future ray tracing)
    spectralBuffer;
    accumulationBuffer;
    bvhBuffer;
    materialBuffer;

    // CIE Standard Sky parameters
    static CIE_SKY_TYPES = {
        CLEAR: 1,           // CIE Clear Sky (Type 12)
        INTERMEDIATE: 2,    // CIE Intermediate Sky (Type 8)
        OVERCAST: 3         // CIE Standard Overcast Sky (Type 1)
    };

    // Photopic luminous efficiency curve V(λ) - CIE 1924
    static PHOTOPIC_CURVE = new Float32Array([
        0.000, 0.001, 0.004, 0.012,  // 380-455nm
        0.060, 0.139, 0.323, 0.710,  // 480-555nm
        0.954, 0.995, 0.870, 0.631,  // 580-655nm
        0.381, 0.175, 0.061, 0.017   // 680-755nm
    ]);

    // Melanopic action spectrum - CIE S 026:2018 (ipRGC sensitivity)
    static MELANOPIC_CURVE = new Float32Array([
        0.001, 0.002, 0.010, 0.050,  // 380-455nm
        0.377, 1.000, 0.548, 0.165,  // 480-555nm (peak at 480nm)
        0.051, 0.018, 0.008, 0.004,  // 580-655nm
        0.002, 0.001, 0.001, 0.000   // 680-755nm
    ]);

    // Default surface reflectances (ρ) per CIBSE/IES standards
    static DEFAULT_REFLECTANCES = {
        ceiling: 0.70,   // White/light ceiling
        walls: 0.50,     // Light colored walls
        floor: 0.20,     // Dark floor
        furniture: 0.30, // Average furniture
        glazing_tau: 0.65 // Double glazing transmittance
    };

    constructor(device, gridConfig, config = {}) {
        this.device = device;
        this.gridConfig = gridConfig;

        // Initialize spectral bands (380nm to 780nm, 25nm steps)
        this.wavelengths = new Float32Array(16);
        for (let i = 0; i < 16; i++) {
            this.wavelengths[i] = 380 + i * 25;
        }

        this.config = {
            wavelengthBands: 16,
            // Surface reflectances
            reflectances: { ...SpectralTracer.DEFAULT_REFLECTANCES, ...config.reflectances },
            // Glazing properties
            glazingTransmittance: config.glazingTransmittance ?? 0.65,
            glazingTint: config.glazingTint ?? 'clear', // clear, green, bronze, blue
            // Calculation options
            includeInterreflections: config.includeInterreflections ?? true,
            skyType: config.skyType ?? 'auto', // auto, clear, intermediate, overcast
            ...config
        };

        console.log('[DaylightAnalysis] Initialized with config:', {
            glazingTau: this.config.glazingTransmittance,
            reflectances: this.config.reflectances
        });
    }

    /**
     * Initialize daylight analysis resources
     * Note: BVH/GPU resources disabled - using analytical model only
     */
    async initialize(voxelBuffer, materialLibrary) {
        // Extract reflectances from materials if provided
        if (materialLibrary) {
            this.extractMaterialReflectances(materialLibrary);
        }

        // GPU ray tracing disabled - skip BVH building
        // This saves memory and prevents GPU device crashes
        console.log('[DaylightAnalysis] Initialized (analytical model - GPU ray tracing disabled)');
    }

    /**
     * Extract visual reflectance from material library
     */
    extractMaterialReflectances(materials) {
        const materialArray = Array.isArray(materials) ? materials :
            (materials?.elements || Object.values(materials || {}));

        if (!materialArray || materialArray.length === 0) return;

        // Calculate average reflectance from materials
        let totalReflectance = 0;
        let count = 0;

        for (const mat of materialArray) {
            // Try to get reflectance from various properties
            const refl = mat.reflectance ?? mat.albedo ??
                (mat.color ? this.colorToReflectance(mat.color) : null);
            if (refl !== null) {
                totalReflectance += refl;
                count++;
            }
        }

        if (count > 0) {
            const avgReflectance = totalReflectance / count;
            this.config.reflectances.walls = avgReflectance;
            console.log('[DaylightAnalysis] Material average reflectance:', avgReflectance.toFixed(2));
        }
    }

    /**
     * Convert RGB color to approximate visual reflectance
     */
    colorToReflectance(color) {
        if (!color) return null;
        // Assume color is {r, g, b} in 0-1 range or 0-255
        const r = color.r > 1 ? color.r / 255 : color.r;
        const g = color.g > 1 ? color.g / 255 : color.g;
        const b = color.b > 1 ? color.b / 255 : color.b;
        // Luminance-weighted reflectance
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    /**
     * Main analysis entry point - GPU Path Tracing
     * @param {Object} sunPosition - Solar position data
     * @param {Object} options - Analysis options
     * @returns {Object} Complete daylight analysis results
     */
    async trace(sunPosition, options = {}) {
        // === ANALYTICAL MODEL ONLY ===
        // GPU ray tracing disabled to prevent device crashes on consumer hardware.
        // Uses BRE Split-Flux method (industry standard) instead.
        // Re-enable GPU path tracing when deployed on server with dedicated GPU.
        console.log('[DaylightAnalysis] Using analytical model (GPU ray tracing disabled)');

        const elevation = sunPosition?.sunDirection?.elevation ?? 0;
        const azimuth = sunPosition?.sunDirection?.azimuth ?? 0;
        const epwWeather = options?.epwWeather || sunPosition?.epwWeather;
        const detectedRooms = options?.detectedRooms || [];
        const windowToFloorRatio = options?.windowToFloorRatio ?? 0.15;

        // === NIGHT TIME CHECK ===
        if (elevation <= 0) {
            return this.createNightResult();
        }

        // === DETERMINE SKY TYPE ===
        const skyType = this.determineSkyType(epwWeather, elevation);

        // === CALCULATE EXTERIOR ILLUMINANCE ===
        const exterior = this.calculateExteriorIlluminance(sunPosition, epwWeather, elevation, skyType);

        // === ANALYZE ROOM GEOMETRY ===
        const roomAnalysis = this.analyzeRooms(detectedRooms, windowToFloorRatio);

        // === CALCULATE DAYLIGHT FACTOR (BRE Split-Flux Method) ===
        const daylightFactor = this.calculateDaylightFactorSplitFlux(
            roomAnalysis, windowToFloorRatio, skyType, elevation
        );

        // === CALCULATE INTERIOR ILLUMINANCE ===
        const interior = this.calculateInteriorIlluminance(
            exterior.globalHorizontal, daylightFactor, roomAnalysis
        );

        // === UNIFORMITY ===
        const uniformity = this.calculateUniformity(roomAnalysis, skyType, exterior.diffuseFraction);

        // === CIRCADIAN METRICS ===
        const circadian = this.calculateCircadianMetrics(
            interior.average, elevation, exterior.colorTemperature
        );

        // === BUILD RESULT ===
        console.log(`[DaylightAnalysis] Results: ${interior.average.toFixed(0)}lux, DF=${daylightFactor.total.toFixed(1)}%`);

        return {
            // Primary metrics
            avgIlluminance: Math.round(interior.average),
            minIlluminance: Math.round(interior.minimum),
            maxIlluminance: Math.round(interior.maximum),
            daylightFactor: parseFloat(daylightFactor.total.toFixed(1)),

            // Component breakdown
            skyComponent: parseFloat(daylightFactor.sky.toFixed(2)),
            externalComponent: parseFloat(daylightFactor.external.toFixed(2)),
            internalComponent: parseFloat(daylightFactor.internal.toFixed(2)),

            // Distribution
            uniformity: parseFloat(uniformity.toFixed(2)),
            diversityRatio: interior.maximum > 0 ? parseFloat((interior.maximum / Math.max(1, interior.minimum)).toFixed(1)) : 1.0,

            // Circadian/Biological
            melanopicLux: circadian.melanopicLux,
            melanopicRatio: parseFloat(circadian.melanopicRatio.toFixed(2)),
            circadianStimulus: parseFloat(circadian.cs.toFixed(2)),
            equivalentMelanopicLux: circadian.eml,

            // Sky conditions
            skyType: skyType,
            exteriorIlluminance: Math.round(exterior.globalHorizontal),
            diffuseFraction: parseFloat(exterior.diffuseFraction.toFixed(2)),
            colorTemperature: exterior.colorTemperature,

            // Metadata
            dataSource: exterior.source,
            analysisMethod: 'analytical-split-flux', // Changed from gpu-path-tracing
            roomIndex: parseFloat(roomAnalysis.averageRoomIndex.toFixed(2)),

            // Compliance
            compliance: this.checkCompliance(interior.average, daylightFactor.total, uniformity)
        };
    }

    /**
     * Read radiance buffer from GPU
     */
    async readRadianceBuffer() {
        const bufferSize = this.spectralBuffer.size;
        const stagingBuffer = this.device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.spectralBuffer, 0, stagingBuffer, 0, bufferSize);
        this.device.queue.submit([encoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(stagingBuffer.getMappedRange());
        const radianceData = new Float32Array(data); // Copy
        stagingBuffer.unmap();

        return radianceData;
    }

    /**
     * Process path tracing results into metrics
     */
    processPathTracingResults(radianceData, exterior, skyType, detectedRooms) {
        const { nx, ny, nz } = this.gridConfig.dimensions;
        const totalVoxels = nx * ny * nz;

        // Convert radiance to illuminance (lux)
        // Radiance is in RGB W/m²/sr, illuminance is in lux
        const LUMEN_CONSTANT = 683; // lm/W at 555nm

        let sum_illuminance = 0;
        let min_illuminance = Infinity;
        let max_illuminance = 0;
        let count = 0;

        for (let i = 0; i < totalVoxels; i++) {
            const r = radianceData[i * 4 + 0];
            const g = radianceData[i * 4 + 1];
            const b = radianceData[i * 4 + 2];

            // Luminance (simple Y from RGB)
            const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            const illuminance = luminance * LUMEN_CONSTANT * Math.PI; // Convert to lux

            if (illuminance > 0.1) { // Filter out near-zero values
                sum_illuminance += illuminance;
                min_illuminance = Math.min(min_illuminance, illuminance);
                max_illuminance = Math.max(max_illuminance, illuminance);
                count++;
            }
        }

        const avg_illuminance = count > 0 ? sum_illuminance / count : 0;

        // Calculate daylight factor: DF = (interior_lux / exterior_lux) * 100%
        const daylight_factor = exterior.globalHorizontal > 0
            ? (avg_illuminance / exterior.globalHorizontal) * 100
            : 0;

        // Uniformity
        const uniformity = min_illuminance > 0 ? min_illuminance / avg_illuminance : 0;

        // Circadian metrics
        const circadian = this.calculateCircadianMetrics(
            avg_illuminance,
            exterior.sunDirection?.elevation ?? 0,
            exterior.colorTemperature
        );

        console.log(`[DaylightRT] Results: ${avg_illuminance.toFixed(0)}lux, DF=${daylight_factor.toFixed(1)}%`);

        return {
            // Primary metrics
            avgIlluminance: Math.round(avg_illuminance),
            minIlluminance: Math.round(min_illuminance === Infinity ? 0 : min_illuminance),
            maxIlluminance: Math.round(max_illuminance),
            daylightFactor: parseFloat(daylight_factor.toFixed(1)),

            // Distribution
            uniformity: parseFloat(uniformity.toFixed(2)),
            diversityRatio: max_illuminance > 0 ? parseFloat((max_illuminance / Math.max(1, min_illuminance)).toFixed(1)) : 1.0,

            // Circadian/Biological
            melanopicLux: circadian.melanopicLux,
            melanopicRatio: parseFloat(circadian.melanopicRatio.toFixed(2)),
            circadianStimulus: parseFloat(circadian.cs.toFixed(2)),
            equivalentMelanopicLux: circadian.eml,

            // Sky conditions
            skyType: skyType, // Pass the string directly
            exteriorIlluminance: Math.round(exterior.globalHorizontal),
            diffuseFraction: parseFloat(exterior.diffuseFraction.toFixed(2)),
            colorTemperature: exterior.colorTemperature,

            // Metadata
            dataSource: exterior.source,
            analysisMethod: 'gpu-path-tracing',
            samples: 128,

            // Compliance
            compliance: this.checkCompliance(avg_illuminance, daylight_factor, uniformity)
        };
    }

    /**
     * Determine CIE sky type from weather data
     */
    determineSkyType(epwWeather, elevation) {
        if (this.config.skyType !== 'auto') {
            return this.config.skyType;
        }

        if (!epwWeather) return 'intermediate';

        // Use cloud cover if available (0-10 tenths)
        const cloudCover = epwWeather.totalSkyCover ?? 5;

        // Also check diffuse fraction if radiation data available
        let diffuseFraction = 0.5;
        if (epwWeather.directNormalRad !== undefined && epwWeather.diffuseHorizRad !== undefined) {
            const ghi = epwWeather.directNormalRad * Math.sin(elevation * Math.PI / 180) + epwWeather.diffuseHorizRad;
            diffuseFraction = ghi > 0 ? epwWeather.diffuseHorizRad / ghi : 0.5;
        }

        // Decision matrix
        if (cloudCover <= 2 && diffuseFraction < 0.3) {
            return 'clear';
        } else if (cloudCover >= 8 || diffuseFraction > 0.7) {
            return 'overcast';
        } else {
            return 'intermediate';
        }
    }

    /**
     * Calculate exterior horizontal illuminance using EPW or models
     */
    calculateExteriorIlluminance(sunPosition, epwWeather, elevation, skyType) {
        const gammaRad = elevation * Math.PI / 180;
        let directHorizontal, diffuseHorizontal, globalHorizontal;
        let source = 'model';
        let colorTemperature = 6500; // Default daylight CCT

        // Priority 1: EPW illuminance data (lux)
        if (epwWeather?.directNormalIllum && epwWeather?.diffuseHorizIllum) {
            directHorizontal = epwWeather.directNormalIllum * Math.sin(gammaRad);
            diffuseHorizontal = epwWeather.diffuseHorizIllum;
            globalHorizontal = directHorizontal + diffuseHorizontal;
            source = 'epw-illuminance';
        }
        // Priority 2: EPW irradiance data (W/m²)
        else if (epwWeather?.directNormalRad !== undefined && epwWeather?.diffuseHorizRad !== undefined) {
            // Perez luminous efficacy model (simplified)
            const efficacy = this.calculateLuminousEfficacy(elevation, skyType, epwWeather);
            directHorizontal = epwWeather.directNormalRad * Math.sin(gammaRad) * efficacy.direct;
            diffuseHorizontal = epwWeather.diffuseHorizRad * efficacy.diffuse;
            globalHorizontal = directHorizontal + diffuseHorizontal;
            source = 'epw-irradiance';
        }
        // Priority 3: Calculated irradiance
        else {
            const ghi = sunPosition?.irradiance ?? sunPosition?.ghi ?? 0;
            const efficacy = this.calculateLuminousEfficacy(elevation, skyType, null);
            globalHorizontal = ghi * efficacy.global;

            // Estimate direct/diffuse split based on sky type
            const splitRatio = skyType === 'clear' ? 0.85 : (skyType === 'overcast' ? 0.10 : 0.50);
            directHorizontal = globalHorizontal * splitRatio;
            diffuseHorizontal = globalHorizontal * (1 - splitRatio);
            source = 'calculated';
        }

        // Calculate color temperature based on sky conditions
        colorTemperature = this.estimateColorTemperature(elevation, skyType, diffuseHorizontal / Math.max(1, globalHorizontal));

        const diffuseFraction = globalHorizontal > 0 ? diffuseHorizontal / globalHorizontal : 0.5;

        return {
            directHorizontal,
            diffuseHorizontal,
            globalHorizontal,
            diffuseFraction,
            colorTemperature,
            source
        };
    }

    /**
     * Calculate luminous efficacy based on Perez model (simplified)
     * Returns lm/W for direct, diffuse, and global radiation
     */
    calculateLuminousEfficacy(elevation, skyType, epwWeather) {
        // Base efficacy varies with solar altitude (higher sun = bluer light = higher efficacy)
        const altitudeRad = elevation * Math.PI / 180;

        // Perez-derived luminous efficacy (lm/W)
        // Reference: Perez et al. 1990, "Modeling daylight availability and irradiance components"
        const sinAlt = Math.sin(altitudeRad);

        // Direct beam efficacy: 60-120 lm/W
        const directEfficacy = 60 + 60 * sinAlt;

        // Diffuse efficacy depends on sky type
        let diffuseEfficacy;
        switch (skyType) {
            case 'clear':
                diffuseEfficacy = 130 - 20 * sinAlt; // Bluer sky when sun high
                break;
            case 'overcast':
                diffuseEfficacy = 110; // Fairly constant for overcast
                break;
            default: // intermediate
                diffuseEfficacy = 120 - 10 * sinAlt;
        }

        // Global average (weighted by typical split)
        const globalEfficacy = 0.5 * directEfficacy + 0.5 * diffuseEfficacy;

        return {
            direct: directEfficacy,
            diffuse: diffuseEfficacy,
            global: globalEfficacy
        };
    }

    /**
     * Analyze room geometry for daylight calculations
     */
    analyzeRooms(detectedRooms, windowToFloorRatio) {
        if (!detectedRooms || detectedRooms.length === 0) {
            // Use grid dimensions as single room fallback
            const { nx, ny, nz } = this.gridConfig.dimensions;
            const res = this.gridConfig.resolution;
            const width = nx * res;
            const depth = nz * res;
            const height = ny * res;
            const floorArea = width * depth;
            const volume = floorArea * height;

            return {
                rooms: [{
                    width, depth, height, floorArea, volume,
                    roomIndex: this.calculateRoomIndex(depth, height)
                }],
                averageDepth: depth,
                averageHeight: height,
                totalFloorArea: floorArea,
                totalVolume: volume,
                averageRoomIndex: this.calculateRoomIndex(depth, height),
                summary: `grid-estimate (${width.toFixed(1)}×${depth.toFixed(1)}×${height.toFixed(1)}m)`
            };
        }

        // Analyze detected rooms
        const rooms = detectedRooms.map(room => {
            // Extract or estimate dimensions from volume
            const volume = room.volume || 50;
            const height = room.height || 3.0;
            const floorArea = volume / height;
            const depth = room.depth || Math.sqrt(floorArea);
            const width = floorArea / depth;

            return {
                width, depth, height, floorArea, volume,
                roomIndex: this.calculateRoomIndex(depth, height),
                id: room.id
            };
        });

        // Calculate averages weighted by floor area
        let totalFloorArea = 0;
        let totalVolume = 0;
        let weightedDepth = 0;
        let weightedHeight = 0;
        let weightedRoomIndex = 0;

        for (const room of rooms) {
            totalFloorArea += room.floorArea;
            totalVolume += room.volume;
            weightedDepth += room.depth * room.floorArea;
            weightedHeight += room.height * room.floorArea;
            weightedRoomIndex += room.roomIndex * room.floorArea;
        }

        return {
            rooms,
            averageDepth: weightedDepth / totalFloorArea,
            averageHeight: weightedHeight / totalFloorArea,
            totalFloorArea,
            totalVolume,
            averageRoomIndex: weightedRoomIndex / totalFloorArea,
            summary: `detected-rooms (n=${rooms.length}, avgDepth=${(weightedDepth / totalFloorArea).toFixed(1)}m)`
        };
    }

    /**
     * Calculate Room Index (RI) - CIBSE/BRE formula
     * RI = (L × W) / ((L + W) × Hm)
     * Where Hm is the height from task plane to window head
     */
    calculateRoomIndex(depth, height) {
        // Assume window head at 2.4m, task plane at 0.85m
        const windowHead = Math.min(height - 0.3, 2.4);
        const taskPlane = 0.85;
        const Hm = windowHead - taskPlane;

        // For rectangular room, use depth as limiting dimension
        const L = depth;
        const W = depth; // Assume square for simplicity

        return (L * W) / ((L + W) * Hm);
    }

    /**
     * Calculate Daylight Factor using Split-Flux Method (BRE Digest 310)
     *
     * DF = SC + ERC + IRC
     *   SC = Sky Component (direct sky visibility)
     *   ERC = Externally Reflected Component (from ground/buildings)
     *   IRC = Internally Reflected Component (inter-reflections)
     */
    calculateDaylightFactorSplitFlux(roomAnalysis, wfr, skyType, elevation) {
        const depth = roomAnalysis.averageDepth;
        const height = roomAnalysis.averageHeight;
        const roomIndex = roomAnalysis.averageRoomIndex;
        const tau = this.config.glazingTransmittance;
        const rho = this.config.reflectances;

        // === SKY COMPONENT (SC) ===
        // Based on window solid angle and sky luminance distribution
        // For CIE Overcast Sky: sky luminance varies with zenith angle
        // SC ≈ (Ag / Af) × tau × M × θ factor

        // Glazing ratio (window area / floor area)
        const glazingRatio = wfr;

        // Maintenance factor (dirt on glass, etc.)
        const maintenanceFactor = 0.85;

        // Sky view factor - decreases with room depth
        // Based on BRE no-sky line concept
        // IMPORTANT: For building analysis, clamp depth to reasonable room sizes
        // If we're analyzing an entire building grid, assume typical room depth
        const effectiveDepth = Math.min(depth, 12); // Max 12m for realistic room
        const effectiveHeight = Math.max(2.4, Math.min(height, 4.0)); // 2.4-4m typical
        const noSkyLineDepth = effectiveHeight * 2.5; // Approximate no-sky line (~6-10m)

        // Sky view factor with floor of 0.1 (never zero - some daylight always penetrates)
        const rawSkyViewFactor = 1 - (effectiveDepth / (noSkyLineDepth * 2));
        const skyViewFactor = Math.max(0.1, Math.min(1.0, rawSkyViewFactor));

        // CIE sky luminance distribution factor
        let skyDistributionFactor;
        switch (skyType) {
            case 'overcast':
                // CIE Overcast: zenith is 3× brighter than horizon
                skyDistributionFactor = 0.85;
                break;
            case 'clear':
                // CIE Clear: depends on sun position
                skyDistributionFactor = 0.50 + 0.30 * Math.sin(elevation * Math.PI / 180);
                break;
            default: // intermediate
                skyDistributionFactor = 0.65;
        }

        const SC = glazingRatio * tau * maintenanceFactor * skyViewFactor * skyDistributionFactor * 100;

        console.log(`[DaylightAnalysis] SC calc: depth=${depth.toFixed(1)}m(eff=${effectiveDepth.toFixed(1)}), ` +
            `height=${height.toFixed(1)}m(eff=${effectiveHeight.toFixed(1)}), ` +
            `SVF=${skyViewFactor.toFixed(2)}, WFR=${(glazingRatio * 100).toFixed(1)}%, SC=${SC.toFixed(2)}%`);

        // === EXTERNALLY REFLECTED COMPONENT (ERC) ===
        // Light reflected from ground and external obstructions
        const groundReflectance = 0.20; // Typical ground
        const obstructionAngle = 0; // Assume no obstructions (TODO: calculate from context)

        // ERC is typically 10-20% of SC for unobstructed windows
        const ERC = SC * 0.1 * groundReflectance / 0.20;

        // === INTERNALLY REFLECTED COMPONENT (IRC) ===
        // Inter-reflections within the room
        // BRE formula: IRC = 0.85 × Ag × tau × (C × ρfw + 5 × ρcw) / (Af × (1 - ρ̄))

        // Average room surface reflectance
        const avgReflectance = (rho.ceiling + rho.walls * 4 + rho.floor) / 6;

        // Coefficients based on window position
        const Cfw = 39; // Factor for floor/wall below window
        const Ccw = 5;  // Factor for ceiling/wall above window

        // IRC calculation with inter-reflection boost
        const interreflectionBoost = 1 / (1 - avgReflectance);
        const IRC = 0.85 * glazingRatio * tau * maintenanceFactor *
            ((Cfw * rho.floor + Ccw * rho.ceiling) / 100) *
            interreflectionBoost;

        // Clamp IRC to realistic values (typically 0.5-3% for side-lit rooms)
        const IRCclamped = Math.min(3.0, Math.max(0.2, IRC));

        // Total DF
        const totalDF = Math.max(0.5, SC + ERC + IRCclamped);

        return {
            total: totalDF,
            sky: SC,
            external: ERC,
            internal: IRCclamped,
            method: 'split-flux'
        };
    }

    /**
     * Calculate interior illuminance from DF and exterior illuminance
     */
    calculateInteriorIlluminance(exteriorIlluminance, daylightFactor, roomAnalysis) {
        // Average illuminance
        const average = exteriorIlluminance * daylightFactor.total / 100;

        // Estimate distribution (front to back of room)
        // Near window: DF × 2-3
        // Back of room: DF × 0.3-0.5
        const nearWindowDF = daylightFactor.sky + daylightFactor.internal * 1.5;
        const backRoomDF = daylightFactor.internal * 0.5;

        const maximum = exteriorIlluminance * nearWindowDF / 100;
        const minimum = exteriorIlluminance * Math.max(0.5, backRoomDF) / 100;

        return {
            average: Math.max(10, average), // Minimum 10 lux during day
            minimum: Math.max(5, minimum),
            maximum: Math.max(average, maximum)
        };
    }

    /**
     * Calculate uniformity ratio
     */
    calculateUniformity(roomAnalysis, skyType, diffuseFraction) {
        // Uniformity depends on:
        // - Sky type (overcast = more uniform)
        // - Room depth (deeper = less uniform)
        // - Diffuse fraction (more diffuse = more uniform)

        const baseUniformity = skyType === 'overcast' ? 0.5 : (skyType === 'clear' ? 0.2 : 0.35);

        // Depth penalty
        const depthPenalty = Math.min(0.2, roomAnalysis.averageDepth / 50);

        // Diffuse bonus
        const diffuseBonus = 0.3 * diffuseFraction;

        const uniformity = Math.min(0.8, Math.max(0.1, baseUniformity - depthPenalty + diffuseBonus));

        return uniformity;
    }

    /**
     * Calculate circadian and biological lighting metrics
     */
    calculateCircadianMetrics(illuminance, elevation, cct) {
        // Melanopic ratio depends on color temperature
        // Warmer light (lower CCT) has lower melanopic content
        // Daylight (5000-7500K) has high melanopic content
        const melanopicRatio = this.cctToMelanopicRatio(cct);

        // Equivalent Melanopic Lux
        const eml = illuminance * melanopicRatio;

        // Melanopic Lux (approximate)
        const melanopicLux = Math.round(eml);

        // Circadian Stimulus (Rea et al. 2012 model)
        const cs = this.calculateCircadianStimulus(eml);

        return {
            melanopicLux,
            melanopicRatio,
            eml: Math.round(eml),
            cs
        };
    }

    /**
     * Convert CCT to melanopic-photopic ratio
     * Based on CIE S 026:2018 melanopic sensitivity
     */
    cctToMelanopicRatio(cct) {
        // Lower CCT (warmer) = lower melanopic content
        // Higher CCT (cooler/bluer) = higher melanopic content
        // Range: ~0.4 (2700K) to ~1.1 (10000K)
        if (cct <= 2700) return 0.40;
        if (cct <= 3000) return 0.45;
        if (cct <= 4000) return 0.55;
        if (cct <= 5000) return 0.70;
        if (cct <= 6000) return 0.85;
        if (cct <= 6500) return 0.90;
        if (cct <= 7500) return 0.95;
        if (cct <= 10000) return 1.05;
        return 1.10;
    }

    /**
     * Calculate Circadian Stimulus (CS) from EML
     * Based on Rea et al. 2012 model
     */
    calculateCircadianStimulus(eml) {
        if (eml < 1) return 0;

        // Rea CS model (simplified)
        // CS saturates around 0.7 at very high light levels
        const cs = 0.7 * (1 - (1 / (1 + Math.pow(eml / 250, 1.5))));

        return Math.min(0.7, Math.max(0, cs));
    }

    /**
     * Estimate correlated color temperature from sky conditions
     */
    estimateColorTemperature(elevation, skyType, diffuseFraction) {
        // Base CCT depends on sky type
        let baseCCT;
        switch (skyType) {
            case 'overcast':
                baseCCT = 6500; // Overcast sky is fairly neutral
                break;
            case 'clear':
                // Clear sky CCT varies with elevation
                // Low sun = warmer, high sun = cooler (more blue sky contribution)
                baseCCT = 5000 + 3000 * Math.sin(elevation * Math.PI / 180);
                break;
            default: // intermediate
                baseCCT = 5500 + 1500 * Math.sin(elevation * Math.PI / 180);
        }

        // Diffuse-dominant light is bluer (sky contribution)
        const diffuseAdjustment = (diffuseFraction - 0.5) * 2000;

        return Math.round(baseCCT + diffuseAdjustment);
    }

    /**
     * Check compliance with standards
     */
    checkCompliance(avgIlluminance, df, uniformity) {
        const compliance = {
            // EN 12464-1 Office general (300 lux minimum)
            en12464_office: avgIlluminance >= 300,
            // EN 12464-1 Drawing/CAD (500 lux minimum)
            en12464_technical: avgIlluminance >= 500,
            // BREEAM Excellent (2% DF average)
            breeam_excellent: df >= 2.0,
            // BREEAM Good (1.5% DF)
            breeam_good: df >= 1.5,
            // LEED v4 Daylight Option 2 (55% area above 300 lux)
            leed_illuminance: avgIlluminance >= 300,
            // Uniformity ratio (0.4 minimum for most standards)
            uniformity_pass: uniformity >= 0.4
        };

        compliance.overall = compliance.en12464_office && compliance.breeam_good && compliance.uniformity_pass;

        return compliance;
    }

    /**
     * Create result for nighttime conditions
     */
    createNightResult() {
        return {
            avgIlluminance: 0,
            minIlluminance: 0,
            maxIlluminance: 0,
            daylightFactor: 0,
            daylightFactorComponents: { skyComponent: 0, externalReflected: 0, internalReflected: 0 },
            uniformity: 0,
            diversityRatio: 1,
            melanopicLux: 0,
            melanopicRatio: 0,
            circadianStimulus: 0,
            equivalentMelanopicLux: 0,
            skyType: 'night',
            exteriorIlluminance: 0,
            diffuseFraction: 0,
            colorTemperature: 0,
            dataSource: 'night',
            roomInfo: 'night',
            analysisMethod: 'none',
            compliance: {
                en12464_office: false,
                en12464_technical: false,
                breeam_excellent: false,
                breeam_good: false,
                leed_illuminance: false,
                uniformity_pass: false,
                overall: false
            }
        };
    }

    /**
     * Cleanup resources
     */
    destroy() {
        this.spectralBuffer?.destroy();
        this.accumulationBuffer?.destroy();
        this.bvhBuffer?.destroy();
        this.materialBuffer?.destroy();
        console.log('[DaylightAnalysis] Destroyed');
    }
}
