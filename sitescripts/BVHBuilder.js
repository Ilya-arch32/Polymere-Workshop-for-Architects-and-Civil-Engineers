/**
 * BVH (Bounding Volume Hierarchy) Builder for Ray Tracing
 * 
 * Converts voxel grid to BVH acceleration structure using SAH partitioning.
 * Essential for fast ray-voxel intersection in GPU path tracing.
 * 
 * Reference: "On fast Construction of SAH-based Bounding Volume Hierarchies"
 *            Ingo Wald (2007)
 */

export class BVHBuilder {
    constructor() {
        this.nodes = [];
        this.primitives = [];
    }

    /**
     * Convert voxel grid to AABBs (only SOLID voxels)
     * @param {Uint32Array} voxelStateBuffer - Voxel state data (SOLID=1, FLUID=2, etc)
     * @param {Object} gridConfig - Grid configuration
     * @returns {Array} Array of AABB objects
     */
    voxelsToAABBs(voxelStateBuffer, gridConfig) {
        const { nx, ny, nz } = gridConfig.dimensions;
        const resolution = gridConfig.resolution;
        const aabbs = [];

        console.log('[BVH] Converting voxels to AABBs...');

        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const idx = x + y * nx + z * nx * ny;
                    const state = voxelStateBuffer[idx];

                    // Only include SOLID voxels (walls, floors, etc)
                    // VoxelState uses bit flags, so check if SOLID bit (0x01) is set
                    if ((state & 1) !== 0) { // VoxelState.SOLID bit
                        // Calculate world-space bounds
                        const minX = x * resolution;
                        const minY = y * resolution;
                        const minZ = z * resolution;
                        const maxX = (x + 1) * resolution;
                        const maxY = (y + 1) * resolution;
                        const maxZ = (z + 1) * resolution;

                        aabbs.push({
                            min: [minX, minY, minZ],
                            max: [maxX, maxY, maxZ],
                            voxelIndex: idx,
                            centroid: [
                                (minX + maxX) * 0.5,
                                (minY + maxY) * 0.5,
                                (minZ + maxZ) * 0.5
                            ]
                        });
                    }
                }
            }
        }

        console.log(`[BVH] Created ${aabbs.length} AABBs from ${nx}x${ny}x${nz} grid`);
        this.primitives = aabbs;
        return aabbs;
    }

    /**
     * Build BVH using Surface Area Heuristic (SAH)
     * ITERATIVE version to avoid stack overflow with large datasets
     * @param {Array} aabbs - Array of AABB primitives
     * @param {number} maxLeafSize - Maximum primitives per leaf node
     * @returns {number} Root node index
     */
    buildSAH(aabbs, maxLeafSize = 8) {
        console.log('[BVH] Building SAH-BVH tree (iterative)...');
        const startTime = performance.now();

        this.nodes = [];
        this.primitives = aabbs;

        if (aabbs.length === 0) {
            console.warn('[BVH] No primitives to build tree');
            return -1;
        }

        // Work item: {start, end, parentIdx, isLeftChild}
        // parentIdx = -1 means this is the root
        const workStack = [{ start: 0, end: aabbs.length, parentIdx: -1, isLeftChild: true }];
        let rootIdx = -1;

        while (workStack.length > 0) {
            const work = workStack.pop();
            const { start, end, parentIdx, isLeftChild } = work;
            const primitiveCount = end - start;

            // Calculate bounds for this node
            const bounds = this.calculateBounds(start, end);

            // Leaf node condition
            if (primitiveCount <= maxLeafSize) {
                const nodeIdx = this.nodes.length;
                this.nodes.push({
                    min: bounds.min,
                    max: bounds.max,
                    leftChild: -1,
                    rightChild: -1,
                    primitiveStart: start,
                    primitiveCount: primitiveCount,
                    isLeaf: true
                });

                // Link to parent
                if (parentIdx === -1) {
                    rootIdx = nodeIdx;
                } else if (isLeftChild) {
                    this.nodes[parentIdx].leftChild = nodeIdx;
                } else {
                    this.nodes[parentIdx].rightChild = nodeIdx;
                }
                continue;
            }

            // Find best split using SAH
            const split = this.findBestSplit(start, end, bounds);

            // If no good split found, make leaf
            if (split.axis === -1) {
                const nodeIdx = this.nodes.length;
                this.nodes.push({
                    min: bounds.min,
                    max: bounds.max,
                    leftChild: -1,
                    rightChild: -1,
                    primitiveStart: start,
                    primitiveCount: primitiveCount,
                    isLeaf: true
                });

                // Link to parent
                if (parentIdx === -1) {
                    rootIdx = nodeIdx;
                } else if (isLeftChild) {
                    this.nodes[parentIdx].leftChild = nodeIdx;
                } else {
                    this.nodes[parentIdx].rightChild = nodeIdx;
                }
                continue;
            }

            // Partition primitives
            let mid = this.partition(start, end, split.axis, split.position);

            // Handle degenerate partition (all on one side)
            if (mid === start || mid === end) {
                mid = Math.floor((start + end) / 2);
            }

            // Create internal node
            const nodeIdx = this.nodes.length;
            this.nodes.push({
                min: bounds.min,
                max: bounds.max,
                leftChild: -1,  // Will be set when children are processed
                rightChild: -1,
                primitiveStart: start,
                primitiveCount: primitiveCount,
                isLeaf: false
            });

            // Link to parent
            if (parentIdx === -1) {
                rootIdx = nodeIdx;
            } else if (isLeftChild) {
                this.nodes[parentIdx].leftChild = nodeIdx;
            } else {
                this.nodes[parentIdx].rightChild = nodeIdx;
            }

            // Push children onto work stack (right first so left is processed first)
            workStack.push({ start: mid, end: end, parentIdx: nodeIdx, isLeftChild: false });
            workStack.push({ start: start, end: mid, parentIdx: nodeIdx, isLeftChild: true });
        }

        const buildTime = performance.now() - startTime;
        console.log(`[BVH] Built tree in ${buildTime.toFixed(1)}ms: ${this.nodes.length} nodes`);

        return rootIdx;
    }

    /**
     * Calculate bounding box for primitive range
     */
    calculateBounds(start, end) {
        const bounds = {
            min: [Infinity, Infinity, Infinity],
            max: [-Infinity, -Infinity, -Infinity]
        };

        for (let i = start; i < end; i++) {
            const aabb = this.primitives[i];
            for (let axis = 0; axis < 3; axis++) {
                bounds.min[axis] = Math.min(bounds.min[axis], aabb.min[axis]);
                bounds.max[axis] = Math.max(bounds.max[axis], aabb.max[axis]);
            }
        }

        return bounds;
    }

    /**
     * Find best split using Surface Area Heuristic
     */
    findBestSplit(start, end, bounds) {
        const primitiveCount = end - start;
        const numBuckets = 12;

        let bestCost = Infinity;
        let bestAxis = -1;
        let bestPosition = 0;

        // Try each axis
        for (let axis = 0; axis < 3; axis++) {
            const extent = bounds.max[axis] - bounds.min[axis];
            if (extent < 1e-6) continue; // Skip degenerate axis

            // Initialize buckets
            const buckets = Array(numBuckets).fill(0).map(() => ({
                count: 0,
                bounds: {
                    min: [Infinity, Infinity, Infinity],
                    max: [-Infinity, -Infinity, -Infinity]
                }
            }));

            // Assign primitives to buckets
            for (let i = start; i < end; i++) {
                const centroid = this.primitives[i].centroid[axis];
                let bucketIdx = Math.floor((centroid - bounds.min[axis]) / extent * numBuckets);
                bucketIdx = Math.max(0, Math.min(numBuckets - 1, bucketIdx));

                buckets[bucketIdx].count++;
                const aabb = this.primitives[i];
                for (let a = 0; a < 3; a++) {
                    buckets[bucketIdx].bounds.min[a] = Math.min(buckets[bucketIdx].bounds.min[a], aabb.min[a]);
                    buckets[bucketIdx].bounds.max[a] = Math.max(buckets[bucketIdx].bounds.max[a], aabb.max[a]);
                }
            }

            // Evaluate split positions
            for (let i = 1; i < numBuckets; i++) {
                // Left side: buckets [0, i)
                let leftCount = 0;
                const leftBounds = {
                    min: [Infinity, Infinity, Infinity],
                    max: [-Infinity, -Infinity, -Infinity]
                };
                for (let j = 0; j < i; j++) {
                    leftCount += buckets[j].count;
                    for (let a = 0; a < 3; a++) {
                        leftBounds.min[a] = Math.min(leftBounds.min[a], buckets[j].bounds.min[a]);
                        leftBounds.max[a] = Math.max(leftBounds.max[a], buckets[j].bounds.max[a]);
                    }
                }

                // Right side: buckets [i, numBuckets)
                let rightCount = 0;
                const rightBounds = {
                    min: [Infinity, Infinity, Infinity],
                    max: [-Infinity, -Infinity, -Infinity]
                };
                for (let j = i; j < numBuckets; j++) {
                    rightCount += buckets[j].count;
                    for (let a = 0; a < 3; a++) {
                        rightBounds.min[a] = Math.min(rightBounds.min[a], buckets[j].bounds.min[a]);
                        rightBounds.max[a] = Math.max(rightBounds.max[a], buckets[j].bounds.max[a]);
                    }
                }

                // Skip if one side is empty
                if (leftCount === 0 || rightCount === 0) continue;

                // SAH cost = traversal cost + (leftSA * leftCount + rightSA * rightCount) / parentSA
                const leftSA = this.surfaceArea(leftBounds);
                const rightSA = this.surfaceArea(rightBounds);
                const parentSA = this.surfaceArea(bounds);

                const cost = 1.0 + (leftSA * leftCount + rightSA * rightCount) / parentSA;

                if (cost < bestCost) {
                    bestCost = cost;
                    bestAxis = axis;
                    bestPosition = bounds.min[axis] + (i / numBuckets) * extent;
                }
            }
        }

        // Check if splitting is worth it (vs making a leaf)
        const leafCost = primitiveCount;
        if (bestCost >= leafCost) {
            return { axis: -1, position: 0 }; // No split
        }

        return { axis: bestAxis, position: bestPosition };
    }

    /**
     * Surface area of an AABB
     */
    surfaceArea(bounds) {
        const dx = bounds.max[0] - bounds.min[0];
        const dy = bounds.max[1] - bounds.min[1];
        const dz = bounds.max[2] - bounds.min[2];
        return 2.0 * (dx * dy + dy * dz + dz * dx);
    }

    /**
     * Partition primitives around split position
     */
    partition(start, end, axis, position) {
        let left = start;
        let right = end - 1;

        while (left <= right) {
            while (left <= right && this.primitives[left].centroid[axis] < position) {
                left++;
            }
            while (left <= right && this.primitives[right].centroid[axis] >= position) {
                right--;
            }
            if (left < right) {
                // Swap
                const temp = this.primitives[left];
                this.primitives[left] = this.primitives[right];
                this.primitives[right] = temp;
                left++;
                right--;
            }
        }

        return left;
    }

    /**
     * Flatten BVH tree to linear array for GPU upload
     * Each node: [min_x, min_y, min_z, max_x, max_y, max_z, left/primStart, right/primCount]
     */
    flattenToGPUBuffer(rootNodeIdx) {
        console.log('[BVH] Flattening tree to GPU buffer format...');

        // Each node = 8 floats (6 for AABB + 2 for children/prims)
        const buffer = new Float32Array(this.nodes.length * 8);

        for (let i = 0; i < this.nodes.length; i++) {
            const node = this.nodes[i];
            const offset = i * 8;

            // AABB min/max
            buffer[offset + 0] = node.min[0];
            buffer[offset + 1] = node.min[1];
            buffer[offset + 2] = node.min[2];
            buffer[offset + 3] = node.max[0];
            buffer[offset + 4] = node.max[1];
            buffer[offset + 5] = node.max[2];

            if (node.isLeaf) {
                // Leaf: store primitive range as negative indices (to distinguish from internal)
                buffer[offset + 6] = -node.primitiveStart - 1; // -1 to distinguish from 0
                buffer[offset + 7] = node.primitiveCount;
            } else {
                // Internal: store child indices
                buffer[offset + 6] = node.leftChild;
                buffer[offset + 7] = node.rightChild;
            }
        }

        console.log(`[BVH] Flattened ${this.nodes.length} nodes to ${buffer.byteLength} bytes`);
        return buffer;
    }

    /**
     * Get primitive indices (for uploading to GPU)
     */
    getPrimitiveIndices() {
        return new Uint32Array(this.primitives.map(p => p.voxelIndex));
    }
}
