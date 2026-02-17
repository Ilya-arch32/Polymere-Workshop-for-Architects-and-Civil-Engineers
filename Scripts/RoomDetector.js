/**
 * AHI 2.0 - Room Detection Module (TRL 6)
 * 
 * Hybrid room detection using:
 * 1. 3D Flood Fill - find enclosed FLUID voxels bounded by SOLID walls
 * 2. IfcSpace extraction - validate/override with IFC room data
 * 
 * TRL 6: Algorithm validated, extensive diagnostics, production-ready
 */

import { VoxelState } from './VoxelTypes.js';

export class RoomDetector {
    constructor() {
        this.rooms = [];
        this.roomMask = null;
        this.gridConfig = null;
        this.stats = {
            totalFluid: 0,
            totalSolid: 0,
            totalEmpty: 0,
            roomVoxels: 0,
            exteriorVoxels: 0,
            ifcSpacesFound: 0
        };
        console.log('[RoomDetector] TRL 6 - Initialized');
    }

    /**
     * Main entry point: Hybrid room detection
     * Step 1: Flood fill to find enclosed spaces
     * Step 2: Check IFC for IfcSpace data
     * Step 3: Use IFC if available, otherwise use flood fill
     */
    async detectRoomsHybrid(voxelData, gridConfig, ifcAPI = null, modelID = null) {
        console.log('[RoomDetector] ========================================');
        console.log('[RoomDetector] Starting hybrid room detection...');
        console.log('[RoomDetector] Grid:', gridConfig.dimensions.nx, 'x', gridConfig.dimensions.ny, 'x', gridConfig.dimensions.nz);
        console.log('[RoomDetector] Resolution:', gridConfig.resolution, 'm/voxel');
        console.time('[RoomDetector] Total detection time');

        this.gridConfig = gridConfig;
        const { nx, ny, nz } = gridConfig.dimensions;
        const totalVoxels = nx * ny * nz;

        // Step 0: Analyze voxel distribution
        console.log('[RoomDetector] Step 0: Analyzing voxel distribution...');
        this.analyzeVoxelDistribution(voxelData, totalVoxels);

        // Step 1: Flood fill to find enclosed rooms
        console.log('[RoomDetector] Step 1: Flood fill room detection...');
        const floodFillRooms = this.detectRoomsFloodFill(voxelData, gridConfig);
        console.log('[RoomDetector] Flood fill found', floodFillRooms.length, 'potential rooms');

        // Step 2: Try to extract IfcSpace data from IFC
        let ifcSpaces = [];
        if (ifcAPI && modelID !== null) {
            console.log('[RoomDetector] Step 2: Extracting IfcSpace from IFC...');
            ifcSpaces = await this.extractIfcSpaces(ifcAPI, modelID);
            console.log('[RoomDetector] IFC IfcSpace elements found:', ifcSpaces.length);
            this.stats.ifcSpacesFound = ifcSpaces.length;
        } else {
            console.log('[RoomDetector] Step 2: Skipped (no IFC API provided)');
        }

        // Step 3: Decide which data to use
        // Prefer IfcSpace ONLY if it has meaningful volume data
        console.log('[RoomDetector] Step 3: Selecting best room data...');

        // Calculate total IfcSpace volume
        const ifcSpaceVolume = ifcSpaces.reduce((sum, s) => sum + (s.volume || 0), 0);
        const floodFillVolume = floodFillRooms.reduce((sum, r) => sum + (r.volume || 0), 0);

        console.log('[RoomDetector]   IfcSpace: ', ifcSpaces.length, 'spaces, total volume:', ifcSpaceVolume.toFixed(1), 'm³');
        console.log('[RoomDetector]   FloodFill:', floodFillRooms.length, 'rooms, total volume:', floodFillVolume.toFixed(1), 'm³');

        // Use IfcSpace only if:
        // 1. There are IfcSpace elements AND
        // 2. Total IfcSpace volume is greater than 10 m³ (reasonable minimum) AND
        // 3. IfcSpace volume is not vastly smaller than flood fill (indicating incomplete data)
        const useIfcSpace = ifcSpaces.length > 0 &&
            ifcSpaceVolume > 10 &&
            (floodFillVolume === 0 || ifcSpaceVolume > floodFillVolume * 0.1);

        if (useIfcSpace) {
            console.log('[RoomDetector] ✓ Using IfcSpace data from IFC (', ifcSpaces.length, 'spaces)');
            this.rooms = ifcSpaces;
        } else if (floodFillRooms.length > 0 && floodFillVolume > 0) {
            console.log('[RoomDetector] ✓ Using Flood Fill result (', floodFillRooms.length, 'rooms)');
            if (ifcSpaces.length > 0) {
                console.log('[RoomDetector]   (IfcSpace discarded: volume', ifcSpaceVolume.toFixed(1), 'm³ is too low)');
            }
            this.rooms = floodFillRooms;
        } else {
            console.warn('[RoomDetector] ⚠️ No valid room data from either method!');
            this.rooms = [];
        }

        // Step 4: Create room mask
        console.log('[RoomDetector] Step 4: Creating room mask buffer...');
        this.roomMask = this.createRoomMask(this.rooms, gridConfig, totalVoxels);

        // Final statistics
        this.logFinalStats();
        console.timeEnd('[RoomDetector] Total detection time');
        console.log('[RoomDetector] ========================================');

        return this.rooms;
    }

    /**
     * Analyze voxel state distribution for diagnostics
     */
    analyzeVoxelDistribution(voxelData, totalVoxels) {
        const stride = 8; // 8 floats per voxel
        let fluid = 0, solid = 0, empty = 0, other = 0;

        for (let i = 0; i < totalVoxels; i++) {
            const state = Math.round(voxelData[i * stride]);
            // Use bitfield check: FLUID = 2
            if ((state & 2) !== 0) fluid++;
            else if ((state & 1) !== 0) solid++;
            else if (state === 0) empty++;
            else other++;
        }

        this.stats.totalFluid = fluid;
        this.stats.totalSolid = solid;
        this.stats.totalEmpty = empty;

        console.log('[RoomDetector] Voxel distribution:');
        console.log('  - FLUID:', fluid, '(' + (100 * fluid / totalVoxels).toFixed(1) + '%)');
        console.log('  - SOLID:', solid, '(' + (100 * solid / totalVoxels).toFixed(1) + '%)');
        console.log('  - EMPTY:', empty, '(' + (100 * empty / totalVoxels).toFixed(1) + '%)');
        console.log('  - OTHER:', other);
    }

    /**
     * Detect rooms using EXTERIOR FLOOD-FILL approach (FIXED ALGORITHM)
     * 
     * Algorithm:
     * 1. Mark all FLUID voxels on grid boundaries as EXTERIOR seeds
     * 2. Flood-fill from exterior seeds (stops at SOLID boundaries)
     * 3. All FLUID voxels NOT marked as exterior = INTERIOR
     * 4. Group interior voxels into rooms via connected component analysis
     * 
     * This approach correctly handles:
     * - Multi-layer walls
     * - Complex site models with outdoor spaces
     * - Multiple separate buildings
     */
    detectRoomsFloodFill(voxelData, gridConfig) {
        const { nx, ny, nz } = gridConfig.dimensions;
        const totalVoxels = nx * ny * nz;
        const stride = 8;
        const resolution = gridConfig.resolution;

        console.log('[RoomDetector] Using EXTERIOR FLOOD-FILL algorithm (improved)...');
        console.log('[RoomDetector] Grid size:', nx, 'x', ny, 'x', nz, '=', totalVoxels, 'voxels');

        // ============ DIAGNOSTIC: Analyze SOLID shell integrity ============
        console.log('[RoomDetector] === DIAGNOSTIC: Analyzing building shell ===');
        const solidBounds = this.findSolidBoundingBox(voxelData, gridConfig);
        console.log('[RoomDetector] SOLID bounding box:', solidBounds);
        console.log('[RoomDetector] SOLID X range:', solidBounds.minX, '-', solidBounds.maxX, '(grid:', 0, '-', nx - 1, ')');
        console.log('[RoomDetector] SOLID Y range:', solidBounds.minY, '-', solidBounds.maxY, '(grid:', 0, '-', ny - 1, ')');
        console.log('[RoomDetector] SOLID Z range:', solidBounds.minZ, '-', solidBounds.maxZ, '(grid:', 0, '-', nz - 1, ')');

        // Check if building touches grid edges (problem!)
        const touchesXMin = solidBounds.minX <= 1;
        const touchesXMax = solidBounds.maxX >= nx - 2;
        const touchesYMin = solidBounds.minY <= 1;
        const touchesYMax = solidBounds.maxY >= ny - 2;
        const touchesZMin = solidBounds.minZ <= 1;
        const touchesZMax = solidBounds.maxZ >= nz - 2;

        if (touchesXMin || touchesXMax || touchesZMin || touchesZMax) {
            console.warn('[RoomDetector] ⚠️ DIAGNOSTIC: Building geometry is near or touching grid edges!');
            console.warn('[RoomDetector]   X: min=' + touchesXMin + ', max=' + touchesXMax);
            console.warn('[RoomDetector]   Y: min=' + touchesYMin + ', max=' + touchesYMax);
            console.warn('[RoomDetector]   Z: min=' + touchesZMin + ', max=' + touchesZMax);
        }

        // ============ DIAGNOSTIC: Sample center voxel states ============
        const centerX = Math.floor(nx / 2);
        const centerY = Math.floor(ny / 2);
        const centerZ = Math.floor(nz / 2);
        console.log('[RoomDetector] === DIAGNOSTIC: Sampling center region ===');
        console.log('[RoomDetector] Grid center: (', centerX, ',', centerY, ',', centerZ, ')');

        // Sample a vertical column at center
        console.log('[RoomDetector] Vertical column at center X,Z (Y=0 to Y=' + (ny - 1) + '):');
        for (let y = 0; y < ny; y++) {
            const idx = centerX + y * nx + centerZ * nx * ny;
            const state = Math.round(voxelData[idx * stride]);
            const isFluid = (state & 2) !== 0;
            const isSolid = (state & 1) !== 0;
            const stateDesc = isSolid ? 'SOLID' : (isFluid ? 'FLUID' : 'EMPTY(' + state + ')');
            // Log every 5th Y level for brevity
            if (y % 5 === 0 || y === ny - 1) {
                console.log('[RoomDetector]   Y=' + y + ': state=' + state + ' (' + stateDesc + ')');
            }
        }

        // Sample horizontal slice at typical room height (1/3 of ny)
        const roomY = Math.floor(ny / 3);
        console.log('[RoomDetector] Horizontal slice at Y=' + roomY + ' (sample every 20 voxels):');
        let sliceStats = { fluid: 0, solid: 0, empty: 0 };
        for (let z = 0; z < nz; z += Math.max(1, Math.floor(nz / 10))) {
            let row = '[RoomDetector]   Z=' + z.toString().padStart(3) + ': ';
            for (let x = 0; x < nx; x += Math.max(1, Math.floor(nx / 20))) {
                const idx = x + roomY * nx + z * nx * ny;
                const state = Math.round(voxelData[idx * stride]);
                const isFluid = (state & 2) !== 0;
                const isSolid = (state & 1) !== 0;
                if (isSolid) { sliceStats.solid++; row += 'S'; }
                else if (isFluid) { sliceStats.fluid++; row += '.'; }
                else { sliceStats.empty++; row += 'E'; }
            }
            console.log(row);
        }
        console.log('[RoomDetector] Slice stats: FLUID=' + sliceStats.fluid + ', SOLID=' + sliceStats.solid + ', EMPTY=' + sliceStats.empty);

        // Step 1: Create exterior mask and seed from grid boundaries
        console.log('[RoomDetector] Step 1: Seeding exterior from grid boundaries...');
        const exteriorMask = new Uint8Array(totalVoxels); // 0 = unknown, 1 = exterior

        // ============ DIAGNOSTIC: Check boundary face states ============
        console.log('[RoomDetector] === DIAGNOSTIC: Boundary face analysis ===');
        let boundaryStats = {
            xMin: { fluid: 0, solid: 0, empty: 0 },
            xMax: { fluid: 0, solid: 0, empty: 0 },
            yMin: { fluid: 0, solid: 0, empty: 0 },
            yMax: { fluid: 0, solid: 0, empty: 0 },
            zMin: { fluid: 0, solid: 0, empty: 0 },
            zMax: { fluid: 0, solid: 0, empty: 0 }
        };

        // Analyze X=0 face
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                const idx = 0 + y * nx + z * nx * ny;
                const state = Math.round(voxelData[idx * stride]);
                if ((state & 1) !== 0) boundaryStats.xMin.solid++;
                else if ((state & 2) !== 0) boundaryStats.xMin.fluid++;
                else boundaryStats.xMin.empty++;
            }
        }
        // Analyze X=nx-1 face
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                const idx = (nx - 1) + y * nx + z * nx * ny;
                const state = Math.round(voxelData[idx * stride]);
                if ((state & 1) !== 0) boundaryStats.xMax.solid++;
                else if ((state & 2) !== 0) boundaryStats.xMax.fluid++;
                else boundaryStats.xMax.empty++;
            }
        }
        // Analyze Y=0 face (floor)
        for (let z = 0; z < nz; z++) {
            for (let x = 0; x < nx; x++) {
                const idx = x + 0 * nx + z * nx * ny;
                const state = Math.round(voxelData[idx * stride]);
                if ((state & 1) !== 0) boundaryStats.yMin.solid++;
                else if ((state & 2) !== 0) boundaryStats.yMin.fluid++;
                else boundaryStats.yMin.empty++;
            }
        }
        // Analyze Y=ny-1 face (ceiling)
        for (let z = 0; z < nz; z++) {
            for (let x = 0; x < nx; x++) {
                const idx = x + (ny - 1) * nx + z * nx * ny;
                const state = Math.round(voxelData[idx * stride]);
                if ((state & 1) !== 0) boundaryStats.yMax.solid++;
                else if ((state & 2) !== 0) boundaryStats.yMax.fluid++;
                else boundaryStats.yMax.empty++;
            }
        }
        // Analyze Z=0 face
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                const idx = x + y * nx + 0 * nx * ny;
                const state = Math.round(voxelData[idx * stride]);
                if ((state & 1) !== 0) boundaryStats.zMin.solid++;
                else if ((state & 2) !== 0) boundaryStats.zMin.fluid++;
                else boundaryStats.zMin.empty++;
            }
        }
        // Analyze Z=nz-1 face
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                const idx = x + y * nx + (nz - 1) * nx * ny;
                const state = Math.round(voxelData[idx * stride]);
                if ((state & 1) !== 0) boundaryStats.zMax.solid++;
                else if ((state & 2) !== 0) boundaryStats.zMax.fluid++;
                else boundaryStats.zMax.empty++;
            }
        }

        console.log('[RoomDetector] Boundary X=0 (left):    FLUID=' + boundaryStats.xMin.fluid + ', SOLID=' + boundaryStats.xMin.solid + ', EMPTY=' + boundaryStats.xMin.empty);
        console.log('[RoomDetector] Boundary X=max (right): FLUID=' + boundaryStats.xMax.fluid + ', SOLID=' + boundaryStats.xMax.solid + ', EMPTY=' + boundaryStats.xMax.empty);
        console.log('[RoomDetector] Boundary Y=0 (bottom):  FLUID=' + boundaryStats.yMin.fluid + ', SOLID=' + boundaryStats.yMin.solid + ', EMPTY=' + boundaryStats.yMin.empty);
        console.log('[RoomDetector] Boundary Y=max (top):   FLUID=' + boundaryStats.yMax.fluid + ', SOLID=' + boundaryStats.yMax.solid + ', EMPTY=' + boundaryStats.yMax.empty);
        console.log('[RoomDetector] Boundary Z=0 (front):   FLUID=' + boundaryStats.zMin.fluid + ', SOLID=' + boundaryStats.zMin.solid + ', EMPTY=' + boundaryStats.zMin.empty);
        console.log('[RoomDetector] Boundary Z=max (back):  FLUID=' + boundaryStats.zMax.fluid + ', SOLID=' + boundaryStats.zMax.solid + ', EMPTY=' + boundaryStats.zMax.empty);

        // Get seeds from all 6 faces of the grid
        const seeds = [];

        // Face Z=0 and Z=nz-1 (front/back)
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                seeds.push([x, y, 0]);
                seeds.push([x, y, nz - 1]);
            }
        }
        // Face Y=0 and Y=ny-1 (bottom/top)
        for (let z = 0; z < nz; z++) {
            for (let x = 0; x < nx; x++) {
                seeds.push([x, 0, z]);
                seeds.push([x, ny - 1, z]);
            }
        }
        // Face X=0 and X=nx-1 (left/right)
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                seeds.push([0, y, z]);
                seeds.push([nx - 1, y, z]);
            }
        }

        console.log('[RoomDetector] Exterior seeds from boundaries:', seeds.length);

        // Step 2: Flood fill from all seeds to mark exterior
        console.log('[RoomDetector] Step 2: Flood-filling exterior space...');
        const exteriorCount = this.floodFillExterior(seeds, voxelData, gridConfig, exteriorMask);
        console.log('[RoomDetector] Marked', exteriorCount, 'voxels as EXTERIOR');

        // ============ DIAGNOSTIC: Check if center was marked exterior ============
        console.log('[RoomDetector] === DIAGNOSTIC: Post-flood analysis ===');
        const centerIdx = centerX + centerY * nx + centerZ * nx * ny;
        const centerState = Math.round(voxelData[centerIdx * stride]);
        const centerIsFluid = (centerState & 2) !== 0;
        const centerMarkedExterior = exteriorMask[centerIdx] === 1;
        console.log('[RoomDetector] Grid center voxel (' + centerX + ',' + centerY + ',' + centerZ + '):');
        console.log('[RoomDetector]   - State: ' + centerState + ' (FLUID=' + centerIsFluid + ')');
        console.log('[RoomDetector]   - Marked EXTERIOR: ' + centerMarkedExterior);
        if (centerIsFluid && centerMarkedExterior) {
            console.warn('[RoomDetector] ⚠️ CENTER OF GRID IS MARKED EXTERIOR!');
            console.warn('[RoomDetector]   This means the flood reached it from boundaries.');
            console.warn('[RoomDetector]   Possible causes:');
            console.warn('[RoomDetector]     1. Building has holes/gaps in walls');
            console.warn('[RoomDetector]     2. Windows/doors are marked as FLUID not SOLID');
            console.warn('[RoomDetector]     3. Building is not fully enclosed');
        }

        // Sample a line from center to edge to find where exterior starts
        console.log('[RoomDetector] Tracing path from center to X=0 edge:');
        let lastExteriorX = -1;
        for (let x = centerX; x >= 0; x--) {
            const idx = x + centerY * nx + centerZ * nx * ny;
            const state = Math.round(voxelData[idx * stride]);
            const isFluid = (state & 2) !== 0;
            const isSolid = (state & 1) !== 0;
            const isExterior = exteriorMask[idx] === 1;

            // Log transition points
            if (x === centerX || x === 0 || (x % 10 === 0)) {
                const stateDesc = isSolid ? 'SOLID' : (isFluid ? 'FLUID' : 'EMPTY');
                console.log('[RoomDetector]   X=' + x + ': ' + stateDesc + ', exterior=' + isExterior);
            }
            if (isExterior && lastExteriorX === -1) {
                lastExteriorX = x;
            }
        }
        if (lastExteriorX > 0) {
            console.log('[RoomDetector]   First EXTERIOR voxel at X=' + lastExteriorX);
        }

        // Step 3: Interior = FLUID AND NOT exterior
        console.log('[RoomDetector] Step 3: Identifying interior voxels...');
        const interiorIndices = [];
        let fluidCount = 0;

        for (let i = 0; i < totalVoxels; i++) {
            const state = Math.round(voxelData[i * stride]);
            const isFluid = (state & 2) !== 0;

            if (isFluid) {
                fluidCount++;
                if (exteriorMask[i] === 0) {
                    // This FLUID voxel was NOT reached by exterior flood = interior
                    interiorIndices.push(i);
                }
            }
        }

        console.log('[RoomDetector] Total FLUID voxels:', fluidCount);
        console.log('[RoomDetector] Interior FLUID voxels:', interiorIndices.length);
        console.log('[RoomDetector] Exterior FLUID voxels:', fluidCount - interiorIndices.length);

        // ============ DIAGNOSTIC: Sample interior voxel positions ============
        if (interiorIndices.length > 0 && interiorIndices.length < 10000) {
            console.log('[RoomDetector] === DIAGNOSTIC: Interior voxel locations ===');
            // Find bounding box of interior voxels
            let intMinX = nx, intMaxX = 0, intMinY = ny, intMaxY = 0, intMinZ = nz, intMaxZ = 0;
            for (const idx of interiorIndices) {
                const z = Math.floor(idx / (nx * ny));
                const y = Math.floor((idx % (nx * ny)) / nx);
                const x = idx % nx;
                intMinX = Math.min(intMinX, x); intMaxX = Math.max(intMaxX, x);
                intMinY = Math.min(intMinY, y); intMaxY = Math.max(intMaxY, y);
                intMinZ = Math.min(intMinZ, z); intMaxZ = Math.max(intMaxZ, z);
            }
            console.log('[RoomDetector] Interior bbox: X=[' + intMinX + ',' + intMaxX + '], Y=[' + intMinY + ',' + intMaxY + '], Z=[' + intMinZ + ',' + intMaxZ + ']');
            console.log('[RoomDetector] Interior size: ' + (intMaxX - intMinX + 1) + 'x' + (intMaxY - intMinY + 1) + 'x' + (intMaxZ - intMinZ + 1) + ' voxels');

            // World coords of interior
            const worldOffsetX = gridConfig.offset?.x || 0;
            const worldOffsetY = gridConfig.offset?.y || 0;
            const worldOffsetZ = gridConfig.offset?.z || 0;
            console.log('[RoomDetector] Interior world coords approx:');
            console.log('[RoomDetector]   X: ' + (worldOffsetX + intMinX * resolution).toFixed(1) + 'm to ' + (worldOffsetX + intMaxX * resolution).toFixed(1) + 'm');
            console.log('[RoomDetector]   Y: ' + (worldOffsetY + intMinY * resolution).toFixed(1) + 'm to ' + (worldOffsetY + intMaxY * resolution).toFixed(1) + 'm');
            console.log('[RoomDetector]   Z: ' + (worldOffsetZ + intMinZ * resolution).toFixed(1) + 'm to ' + (worldOffsetZ + intMaxZ * resolution).toFixed(1) + 'm');
        }

        // Validation: Check for suspicious results
        const interiorRatio = interiorIndices.length / fluidCount;
        if (interiorRatio < 0.01) {
            console.warn('[RoomDetector] ⚠️ CRITICAL: Interior ratio is only', (interiorRatio * 100).toFixed(3) + '%!');
            console.warn('[RoomDetector] This is extremely low - likely the entire building was marked as EXTERIOR.');
            console.warn('[RoomDetector] Most likely cause: gaps in walls, open doors/windows, or voxelization issues.');
        } else if (interiorRatio > 0.5) {
            console.warn('[RoomDetector] ⚠️ WARNING: Interior ratio is', (interiorRatio * 100).toFixed(1) + '% - this seems too high!');
            console.warn('[RoomDetector] Possible causes: geometry has holes, or grid boundaries are inside building');
        }
        if (interiorIndices.length === 0) {
            console.warn('[RoomDetector] ⚠️ No interior voxels found! Building may not be enclosed.');
            return [];
        }

        // Step 4: Group interior voxels into separate rooms
        console.log('[RoomDetector] Step 4: Grouping into separate rooms...');
        const rooms = this.groupIntoRooms(interiorIndices, voxelData, gridConfig, resolution);

        // Validation: Warn if largest room is suspiciously large
        if (rooms.length > 0) {
            const largestRoom = rooms.reduce((max, r) => r.voxelCount > max.voxelCount ? r : max, rooms[0]);
            const largestRatio = largestRoom.voxelCount / interiorIndices.length;
            if (largestRatio > 0.9 && rooms.length > 1) {
                console.warn('[RoomDetector] ⚠️ WARNING: Largest room contains', (largestRatio * 100).toFixed(1) + '% of all interior voxels');
                console.warn('[RoomDetector] This may indicate room detection issues (outdoor space detected as room)');
            }
        }

        return rooms;
    }


    /**
     * Find bounding box of all SOLID voxels
     */
    findSolidBoundingBox(voxelData, gridConfig) {
        const { nx, ny, nz } = gridConfig.dimensions;
        const stride = 8;

        let minX = nx, maxX = 0;
        let minY = ny, maxY = 0;
        let minZ = nz, maxZ = 0;
        let solidCount = 0;

        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    const state = Math.round(voxelData[idx * stride]);

                    if ((state & 1) !== 0) { // SOLID
                        solidCount++;
                        minX = Math.min(minX, x);
                        maxX = Math.max(maxX, x);
                        minY = Math.min(minY, y);
                        maxY = Math.max(maxY, y);
                        minZ = Math.min(minZ, z);
                        maxZ = Math.max(maxZ, z);
                    }
                }
            }
        }

        return {
            valid: solidCount > 0,
            minX, maxX, minY, maxY, minZ, maxZ,
            solidCount
        };
    }

    /**
     * Fallback: treat all FLUID within expanded SOLID bbox as interior
     */
    fallbackBboxInterior(voxelData, gridConfig, bbox, resolution) {
        const { nx, ny, nz } = gridConfig.dimensions;
        const stride = 8;

        console.log('[RoomDetector] Fallback: using expanded SOLID bbox as room...');

        // Expand bbox slightly inward (1 voxel padding)
        const padX = 2, padY = 1, padZ = 2;
        const roomVoxels = [];

        for (let z = bbox.minZ + padZ; z <= bbox.maxZ - padZ; z++) {
            for (let y = bbox.minY + padY; y <= bbox.maxY - padY; y++) {
                for (let x = bbox.minX + padX; x <= bbox.maxX - padX; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    const state = Math.round(voxelData[idx * stride]);

                    if ((state & 2) !== 0) { // FLUID
                        roomVoxels.push(idx);
                    }
                }
            }
        }

        if (roomVoxels.length > 100) {
            const volume = roomVoxels.length * Math.pow(resolution, 3);
            const room = {
                id: 0,
                name: 'Building_Interior',
                voxelIndices: roomVoxels,
                voxelCount: roomVoxels.length,
                volume: volume,
                source: 'bbox_fallback'
            };
            console.log('[RoomDetector] Fallback room:', roomVoxels.length, 'voxels,', volume.toFixed(1), 'm³');
            return [room];
        }

        return [];
    }

    /**
     * Get seed points from grid edges (guaranteed to be exterior)
     */
    getEdgeSeeds(nx, ny, nz) {
        const seeds = [];

        // Sample from all 6 faces of the grid
        const step = Math.max(1, Math.floor(Math.min(nx, ny, nz) / 10));

        // Face Z=0 and Z=nz-1
        for (let y = 0; y < ny; y += step) {
            for (let x = 0; x < nx; x += step) {
                seeds.push([x, y, 0]);
                seeds.push([x, y, nz - 1]);
            }
        }
        // Face Y=0 and Y=ny-1
        for (let z = 0; z < nz; z += step) {
            for (let x = 0; x < nx; x += step) {
                seeds.push([x, 0, z]);
                seeds.push([x, ny - 1, z]);
            }
        }
        // Face X=0 and X=nx-1
        for (let z = 0; z < nz; z += step) {
            for (let y = 0; y < ny; y += step) {
                seeds.push([0, y, z]);
                seeds.push([nx - 1, y, z]);
            }
        }

        return seeds;
    }

    /**
     * Flood fill from edge seeds to mark all exterior FLUID
     */
    floodFillExterior(seeds, voxelData, gridConfig, exteriorMask) {
        const { nx, ny, nz } = gridConfig.dimensions;
        const stride = 8;

        const queue = [...seeds];
        let queueHead = 0;
        let markedCount = 0;
        let solidHits = 0;
        let iterations = 0;

        const neighbors = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ];

        // Track flood extents
        let floodMinX = nx, floodMaxX = 0;
        let floodMinY = ny, floodMaxY = 0;
        let floodMinZ = nz, floodMaxZ = 0;

        const logInterval = 500000; // Log every 500k iterations

        while (queueHead < queue.length) {
            const [x, y, z] = queue[queueHead++];
            iterations++;

            // Progress logging
            if (iterations % logInterval === 0) {
                console.log('[RoomDetector] Flood progress: iterations=' + iterations + ', marked=' + markedCount + ', queue=' + (queue.length - queueHead));
            }

            // Bounds check
            if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) continue;

            const idx = x + y * nx + z * nx * ny;

            // Already marked?
            if (exteriorMask[idx] === 1) continue;

            // Check if FLUID
            const state = Math.round(voxelData[idx * stride]);
            const isFluid = (state & 2) !== 0;
            const isSolid = (state & 1) !== 0;

            if (!isFluid) {
                if (isSolid) solidHits++;
                continue; // Stop at SOLID
            }

            // Mark as exterior
            exteriorMask[idx] = 1;
            markedCount++;

            // Track flood extent
            floodMinX = Math.min(floodMinX, x);
            floodMaxX = Math.max(floodMaxX, x);
            floodMinY = Math.min(floodMinY, y);
            floodMaxY = Math.max(floodMaxY, y);
            floodMinZ = Math.min(floodMinZ, z);
            floodMaxZ = Math.max(floodMaxZ, z);

            // Add neighbors
            for (const [dx, dy, dz] of neighbors) {
                queue.push([x + dx, y + dy, z + dz]);
            }
        }

        // Diagnostic: Log flood results
        console.log('[RoomDetector] === DIAGNOSTIC: Flood fill results ===');
        console.log('[RoomDetector] Total iterations:', iterations);
        console.log('[RoomDetector] Voxels marked EXTERIOR:', markedCount);
        console.log('[RoomDetector] SOLID wall hits (flood blocked):', solidHits);
        console.log('[RoomDetector] Flood extent X:', floodMinX, '-', floodMaxX, '(full grid: 0-' + (nx - 1) + ')');
        console.log('[RoomDetector] Flood extent Y:', floodMinY, '-', floodMaxY, '(full grid: 0-' + (ny - 1) + ')');
        console.log('[RoomDetector] Flood extent Z:', floodMinZ, '-', floodMaxZ, '(full grid: 0-' + (nz - 1) + ')');

        // Check if flood reached the building center region
        const centerX = Math.floor(nx / 2);
        const centerY = Math.floor(ny / 2);
        const centerZ = Math.floor(nz / 2);
        const floodReachedCenter = (floodMinX <= centerX && floodMaxX >= centerX &&
            floodMinY <= centerY && floodMaxY >= centerY &&
            floodMinZ <= centerZ && floodMaxZ >= centerZ);
        console.log('[RoomDetector] Flood reached center region:', floodReachedCenter);

        if (floodReachedCenter) {
            console.warn('[RoomDetector] ⚠️ Flood filled through entire grid center!');
            console.warn('[RoomDetector]   This indicates the building is not enclosed OR has gaps.');
        }

        return markedCount;
    }

    /**
     * Group interior voxel indices into separate connected rooms
     */
    groupIntoRooms(interiorIndices, voxelData, gridConfig, resolution) {
        const { nx, ny, nz } = gridConfig.dimensions;
        const stride = 8;
        const rooms = [];

        // Create set for fast lookup
        const interiorSet = new Set(interiorIndices);
        const visited = new Set();

        const neighbors = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ];

        for (const startIdx of interiorIndices) {
            if (visited.has(startIdx)) continue;

            // BFS to find connected interior voxels
            const roomVoxels = [];
            const queue = [startIdx];
            let qHead = 0;

            while (qHead < queue.length) {
                const idx = queue[qHead++];

                if (visited.has(idx)) continue;
                if (!interiorSet.has(idx)) continue;

                visited.add(idx);
                roomVoxels.push(idx);

                // Convert to 3D coords
                const z = Math.floor(idx / (nx * ny));
                const y = Math.floor((idx % (nx * ny)) / nx);
                const x = idx % nx;

                // Add neighbors
                for (const [dx, dy, dz] of neighbors) {
                    const nx2 = x + dx, ny2 = y + dy, nz2 = z + dz;
                    if (nx2 >= 0 && nx2 < nx && ny2 >= 0 && ny2 < ny && nz2 >= 0 && nz2 < nz) {
                        const nIdx = nx2 + ny2 * nx + nz2 * nx * ny;
                        if (interiorSet.has(nIdx) && !visited.has(nIdx)) {
                            queue.push(nIdx);
                        }
                    }
                }
            }

            // Create room if large enough (>100 voxels = ~0.3m³ at typical resolution)
            if (roomVoxels.length > 100) {
                const volume = roomVoxels.length * Math.pow(resolution, 3);
                const room = {
                    id: rooms.length,
                    name: 'Room_' + rooms.length,
                    voxelIndices: roomVoxels,
                    voxelCount: roomVoxels.length,
                    volume: volume,
                    source: 'exterior_elimination'
                };
                rooms.push(room);
                console.log('[RoomDetector] Room', room.id, ':', roomVoxels.length, 'voxels,', volume.toFixed(1), 'm³');
            }
        }

        return rooms;
    }

    /**
     * Find seed points - FLUID voxels likely to be inside buildings
     * Strategy: Look for FLUID voxels with SOLID neighbors on multiple sides
     */
    findSeedPoints(voxelData, gridConfig) {
        const { nx, ny, nz } = gridConfig.dimensions;
        const stride = 8;
        const seeds = [];

        // Sample every Nth voxel for efficiency
        const sampleStep = Math.max(1, Math.floor(Math.min(nx, ny, nz) / 20));
        console.log('[RoomDetector] Scanning for seeds with step:', sampleStep);

        const neighbors = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ];

        for (let z = sampleStep; z < nz - sampleStep; z += sampleStep) {
            for (let y = sampleStep; y < ny - sampleStep; y += sampleStep) {
                for (let x = sampleStep; x < nx - sampleStep; x += sampleStep) {
                    const idx = x + y * nx + z * nx * ny;
                    const state = Math.round(voxelData[idx * stride]);

                    // Must be FLUID
                    if ((state & 2) === 0) continue;

                    // Count SOLID neighbors
                    let solidNeighbors = 0;
                    for (const [dx, dy, dz] of neighbors) {
                        const nx2 = x + dx, ny2 = y + dy, nz2 = z + dz;
                        if (nx2 < 0 || nx2 >= nx || ny2 < 0 || ny2 >= ny || nz2 < 0 || nz2 >= nz) continue;
                        const nIdx = nx2 + ny2 * nx + nz2 * nx * ny;
                        const nState = Math.round(voxelData[nIdx * stride]);
                        if ((nState & 1) !== 0) solidNeighbors++;
                    }

                    // Seed if has at least 1 SOLID neighbor (near wall = inside building)
                    if (solidNeighbors >= 1) {
                        seeds.push({ x, y, z, solidNeighbors });
                    }
                }
            }
        }

        // Sort by number of solid neighbors (more = more likely inside)
        seeds.sort((a, b) => b.solidNeighbors - a.solidNeighbors);

        // Return top N seeds (limit to avoid too many floods)
        const maxSeeds = 10;
        console.log('[RoomDetector] Top seeds (max', maxSeeds, '):');
        for (let i = 0; i < Math.min(seeds.length, 5); i++) {
            console.log('  - Seed', i, ':', seeds[i].x, seeds[i].y, seeds[i].z, '- solidNeighbors:', seeds[i].solidNeighbors);
        }

        return seeds.slice(0, maxSeeds);
    }

    /**
     * 3D Flood Fill algorithm (OPTIMIZED)
     * Fills connected FLUID voxels, stops at SOLID/EMPTY boundaries
     * Uses index-based queue for O(1) dequeue instead of O(n) shift()
     */
    floodFill3D(startX, startY, startZ, voxelData, gridConfig, globalVisited) {
        const { nx, ny, nz } = gridConfig.dimensions;
        const stride = 8;
        const roomVoxels = [];

        // Use index-based queue for O(1) dequeue
        const queue = [[startX, startY, startZ]];
        let queueHead = 0;
        const localVisited = new Set();

        const neighbors = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ];

        // LOWER limit - typical room is ~100-10000 voxels, exterior is 500k+
        const maxVoxels = 50000; // 50k max = clearly a room, not exterior
        let iterations = 0;
        const logInterval = 10000; // Log progress every 10k iterations

        while (queueHead < queue.length && roomVoxels.length < maxVoxels) {
            const [x, y, z] = queue[queueHead++]; // O(1) dequeue
            iterations++;

            // Progress logging
            if (iterations % logInterval === 0) {
                console.log('[RoomDetector]   ... progress: iterations=' + iterations + ', roomVoxels=' + roomVoxels.length + ', queue=' + (queue.length - queueHead));
            }

            // Bounds check
            if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) continue;

            const idx = x + y * nx + z * nx * ny;

            // Already visited?
            if (localVisited.has(idx) || globalVisited.has(idx)) continue;
            localVisited.add(idx);
            globalVisited.add(idx);

            // Check voxel state
            const state = Math.round(voxelData[idx * stride]);
            const isFluid = (state & 2) !== 0;

            // Stop at non-FLUID
            if (!isFluid) continue;

            // Add to room
            roomVoxels.push(idx);

            // Add neighbors to queue
            for (const [dx, dy, dz] of neighbors) {
                queue.push([x + dx, y + dy, z + dz]);
            }
        }

        if (roomVoxels.length >= maxVoxels) {
            console.warn('[RoomDetector] ⚠️ Hit max voxel limit (' + maxVoxels + ') - this is likely EXTERIOR space, discarding!');
            return []; // Return empty - this was exterior, not a room
        }

        console.log('[RoomDetector]   Flood fill: iterations=' + iterations + ', voxels=' + roomVoxels.length);
        return roomVoxels;
    }

    /**
     * Extract IfcSpace elements from IFC model
     * Uses web-ifc API to query space geometry
     */
    async extractIfcSpaces(ifcAPI, modelID) {
        const IFCSPACE = 3856911033; // IFC type ID for IfcSpace
        const spaces = [];

        try {
            console.log('[RoomDetector] Querying IFC for IfcSpace elements (typeID:', IFCSPACE, ')...');
            const spaceIDs = ifcAPI.GetLineIDsWithType(modelID, IFCSPACE);
            console.log('[RoomDetector] Found', spaceIDs.size(), 'IfcSpace elements in IFC');

            for (let i = 0; i < spaceIDs.size(); i++) {
                const spaceID = spaceIDs.get(i);
                const space = ifcAPI.GetLine(modelID, spaceID);

                // Extract properties
                const name = space.Name?.value || 'Space_' + i;
                const longName = space.LongName?.value || '';
                const description = space.Description?.value || '';

                // Get geometry bounding box if available
                let volume = 0;
                let boundingBox = null;

                try {
                    const geometry = ifcAPI.GetFlatMesh(modelID, spaceID);
                    if (geometry.geometries.size() > 0) {
                        // Calculate bounding box from geometry
                        // This is simplified - full implementation would triangulate
                        const geom = geometry.geometries.get(0);
                        const verts = ifcAPI.GetVertexArray(geom.geometryExpressID, ifcAPI.wasmModule);

                        if (verts && verts.length > 0) {
                            let minX = Infinity, minY = Infinity, minZ = Infinity;
                            let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

                            for (let v = 0; v < verts.length; v += 3) {
                                minX = Math.min(minX, verts[v]);
                                maxX = Math.max(maxX, verts[v]);
                                minY = Math.min(minY, verts[v + 1]);
                                maxY = Math.max(maxY, verts[v + 1]);
                                minZ = Math.min(minZ, verts[v + 2]);
                                maxZ = Math.max(maxZ, verts[v + 2]);
                            }

                            boundingBox = { minX, minY, minZ, maxX, maxY, maxZ };
                            volume = (maxX - minX) * (maxY - minY) * (maxZ - minZ);
                        }
                    }
                } catch (geomErr) {
                    console.log('[RoomDetector]   Could not get geometry for space', name);
                }

                const spaceData = {
                    id: i,
                    ifcID: spaceID,
                    name: name,
                    longName: longName,
                    description: description,
                    volume: volume,
                    boundingBox: boundingBox,
                    source: 'ifc_space'
                };

                console.log('[RoomDetector]   IfcSpace', i, ':', name, '- Volume:', volume.toFixed(1), 'm³');
                spaces.push(spaceData);
            }
        } catch (err) {
            console.error('[RoomDetector] Error extracting IfcSpace:', err.message);
        }

        return spaces;
    }

    /**
     * Create a binary mask indicating which voxels are inside rooms
     * roomMask[idx] = 1 if inside any room, 0 otherwise
     */
    createRoomMask(rooms, gridConfig, totalVoxels) {
        console.log('[RoomDetector] Creating room mask for', rooms.length, 'rooms...');
        const mask = new Uint8Array(totalVoxels);
        mask.fill(0);

        let totalRoomVoxels = 0;

        for (const room of rooms) {
            if (room.voxelIndices) {
                // Flood fill rooms have voxel indices
                for (const idx of room.voxelIndices) {
                    if (idx >= 0 && idx < totalVoxels) {
                        mask[idx] = 1;
                        totalRoomVoxels++;
                    }
                }
            } else if (room.boundingBox) {
                // IFC spaces have bounding box - mark voxels inside box
                const { nx, ny, nz } = gridConfig.dimensions;
                const res = gridConfig.resolution;
                const bb = room.boundingBox;

                // Convert world coords to voxel coords
                const minVx = Math.max(0, Math.floor(bb.minX / res));
                const maxVx = Math.min(nx - 1, Math.ceil(bb.maxX / res));
                const minVy = Math.max(0, Math.floor(bb.minY / res));
                const maxVy = Math.min(ny - 1, Math.ceil(bb.maxY / res));
                const minVz = Math.max(0, Math.floor(bb.minZ / res));
                const maxVz = Math.min(nz - 1, Math.ceil(bb.maxZ / res));

                for (let z = minVz; z <= maxVz; z++) {
                    for (let y = minVy; y <= maxVy; y++) {
                        for (let x = minVx; x <= maxVx; x++) {
                            const idx = x + y * nx + z * nx * ny;
                            mask[idx] = 1;
                            totalRoomVoxels++;
                        }
                    }
                }
            }
        }

        this.stats.roomVoxels = totalRoomVoxels;
        this.stats.exteriorVoxels = this.stats.totalFluid - totalRoomVoxels;

        console.log('[RoomDetector] Room mask created:');
        console.log('  - Interior voxels (rooms):', totalRoomVoxels);
        console.log('  - Exterior FLUID voxels:', this.stats.exteriorVoxels);

        return mask;
    }

    /**
     * Get the room mask as Uint8Array
     */
    getRoomMask() {
        return this.roomMask;
    }

    /**
     * Get detected rooms array
     */
    getDetectedRooms() {
        return this.rooms;
    }

    /**
     * Get total interior volume in m³
     */
    getTotalInteriorVolume() {
        return this.rooms.reduce((sum, room) => sum + (room.volume || 0), 0);
    }

    /**
     * Get statistics
     */
    getStats() {
        return this.stats;
    }

    /**
     * Log final statistics
     */
    logFinalStats() {
        const totalVolume = this.getTotalInteriorVolume();
        const voxelVolume = Math.pow(this.gridConfig.resolution, 3);
        const fullGridVolume = this.stats.totalFluid * voxelVolume;

        console.log('[RoomDetector] ========== FINAL STATISTICS ==========');
        console.log('[RoomDetector] Detected rooms:', this.rooms.length);
        console.log('[RoomDetector] Total interior volume:', totalVolume.toFixed(1), 'm³');
        console.log('[RoomDetector] Full grid FLUID volume:', fullGridVolume.toFixed(1), 'm³');
        console.log('[RoomDetector] Reduction factor:', (fullGridVolume / totalVolume).toFixed(1), 'x');
        console.log('[RoomDetector] Interior voxels:', this.stats.roomVoxels);
        console.log('[RoomDetector] Exterior voxels:', this.stats.exteriorVoxels);

        if (this.stats.ifcSpacesFound > 0) {
            console.log('[RoomDetector] Data source: IfcSpace from IFC');
        } else {
            console.log('[RoomDetector] Data source: Flood Fill algorithm');
        }

        // List each room
        console.log('[RoomDetector] Rooms:');
        for (const room of this.rooms) {
            console.log('  -', room.name, ':', room.volume?.toFixed(1) || '?', 'm³', '(' + (room.voxelCount || '') + ' voxels)', '[' + room.source + ']');
        }
    }
}
