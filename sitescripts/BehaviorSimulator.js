/**
 * Polymère — BehaviorSimulator.js
 *
 * ACI (Agent-based Crowd Intelligence) scaffold.
 * Manages autonomous agents whose initial placement is derived from
 * Space Syntax `spaceSyntaxData.observerPoints` (Space Syntax VGA nodes).
 *
 * ════════════════════════════════════════════════════════════════════════
 *  Phase 1 (current) — STATIC PLACEMENT
 *    • spawnAgents(count) creates count capsule meshes and places them
 *      randomly at available observer-point world positions.
 *    • No movement logic yet — scaffolding only.
 *
 *  Phase 2 (TODO) — MOVEMENT & BEHAVIOUR
 *    • step(dt): tick each agent using integration-weighted potential fields.
 *    • Agents prefer high-integration nodes (Space Syntax "attractors").
 *    • Crowd density feedback via adjacency matrix edge weights.
 * ════════════════════════════════════════════════════════════════════════
 *
 * Dependencies: THREE.js (global)
 */
export class BehaviorSimulator {
    /**
     * @param {THREE.Scene}  scene           — The live Three.js scene.
     * @param {Object}       spaceSyntaxData — Result object from SpaceSyntaxSolver.analyze().
     *   Expected fields used here:
     *     .observerPoints  — Array<{x,y,z}> voxel coords (+ optional .heading, .worldPos).
     *     .gridConfig      — { resolution, bounds: { min: {x,y,z} } }
     *     .perNode         — { integration: Float64Array, connectivity: Float64Array }
     */
    constructor(scene, spaceSyntaxData) {
        if (!scene) throw new Error('[BehaviorSimulator] scene is required.');

        this.scene           = scene;
        this.spaceSyntaxData = spaceSyntaxData || null;

        /** @type {Array<THREE.Mesh>} All spawned agent meshes. */
        this.agents = [];

        /** @type {THREE.Group} Container group for easy show/hide/dispose. */
        this.agentGroup = new THREE.Group();
        this.agentGroup.name = 'BehaviorSimulatorAgents';
        this.scene.add(this.agentGroup);

        /** @type {Object} Stigmergy traces */
        this.stigmergyBuffer = {};

        // ── Shared geometry / material (instances share the same geometry) ──
        // Capsule = sphere-capped cylinder approximation.
        //   radius:       0.2 m  (shoulder width ÷ 2 for a slim silhouette)
        //   total height: 0.6 m  (waist-level marker, not full body)
        //   radiusTop/Bottom equal → cylinder; addSpheres below for caps.
        this._capsuleRadius    = 0.2; // metres
        this._capsuleHeight    = 0.6; // metres
        this._capsuleGeom      = null; // lazy-created on first spawn
        this._agentMaterial    = null; // lazy-created on first spawn

        console.log('[BehaviorSimulator] Initialised. Call spawnAgents(count) to place agents.');
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — spawnAgents(count)
    //
    //  Creates `count` agent capsule meshes and places them at randomly
    //  selected observer-point world positions derived from spaceSyntaxData.
    //
    //  If spaceSyntaxData or observerPoints is unavailable, agents are placed
    //  at the scene origin with a warning.
    //
    //  @param {number} count  Number of agents to spawn (default 10).
    //  @returns {Array<THREE.Mesh>} Array of spawned agent meshes.
    // ═══════════════════════════════════════════════════════════════════════

    spawnAgents(count = null) {
        // ── Clear any previously spawned agents ──
        this.clear();

        // ── Build shared geometry & material on first call ──
        this._ensureSharedResources();

        // ── Resolve world-space spawn positions from Space Syntax data ──
        this._resolvedPositions = this._resolveWorldPositions();
        const positions = this._resolvedPositions;

        if (!this._resolvedPositions || this._resolvedPositions.length === 0) return [];
        
        let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
        for (const p of this._resolvedPositions) {
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (p.z < minZ) minZ = p.z;
            if (p.z > maxZ) maxZ = p.z;
        }

        const observerCount = this.spaceSyntaxData?.observerPoints?.length || 0;
        // DATA SCIENCE MODE: agent count must equal observer count exactly.
        // No phantom "seeker" agents — each agent maps 1:1 to a VGA node.
        const agentCount = count !== null ? count : observerCount;
        if (agentCount === 0) {
            console.warn('[BehaviorSimulator] 0 observer points found. Spawning 0 agents.');
            return [];
        }

        console.log(`[BehaviorSimulator] Spawning ${agentCount} agent(s)...`);

        for (let i = 0; i < agentCount; i++) {
            let pos;
            if (i < observerCount) {
                // Squatter: exactly on the node
                pos = this._resolvedPositions[i % this._resolvedPositions.length].clone();
            } else {
                // Seeker: randomly inside the safe bounding box
                pos = new THREE.Vector3(
                    minX + Math.random() * (maxX - minX),
                    this._resolvedPositions[0].y, // Keep floor level
                    minZ + Math.random() * (maxZ - minZ)
                );
            }

            // ── Create agent mesh (shared geometry, cloned material for tint) ──
            const mesh = new THREE.Mesh(
                this._capsuleGeom,
                this._makeAgentMaterial(i, agentCount)
            );

            // ── Cosmetic Stick-Figure Limbs ─────────────────────────────────
            // Pure visual decoration — added as children of the mesh so they
            // inherit its transform.  Does NOT affect physics or material logic.
            const lineMat = new THREE.LineBasicMaterial({ color: 0x333333, linewidth: 2 });

            // Arms — horizontal bar across the upper torso
            const armGeom = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(-0.35, 0.1, 0),
                new THREE.Vector3( 0.35, 0.1, 0)
            ]);
            const arms = new THREE.Line(armGeom, lineMat);
            arms.name = 'Agent_limb';
            mesh.add(arms);

            // Left Leg — drops down from the lower body towards the floor
            const leftLegGeom = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(-0.15, -0.2, 0),
                new THREE.Vector3(-0.15, -1.5, 0) // Reaches the floor at 1.5m eye-level
            ]);
            const leftLeg = new THREE.Line(leftLegGeom, lineMat);
            leftLeg.name = 'Agent_limb';
            mesh.add(leftLeg);

            // Right Leg
            const rightLegGeom = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0.15, -0.2, 0),
                new THREE.Vector3(0.15, -1.5, 0) // Reaches the floor at 1.5m eye-level
            ]);
            const rightLeg = new THREE.Line(rightLegGeom, lineMat);
            rightLeg.name = 'Agent_limb';
            mesh.add(rightLeg);
            // ────────────────────────────────────────────────────────────────

            mesh.position.copy(pos);
            mesh.name        = `Agent_${i}`;
            mesh.castShadow  = true;

            // ── Attach per-agent state for Phase 2 movement ──
            const baseSpeed = Math.random() * 0.6 + 1.2;

            mesh.userData = {
                agentId:    i,
                currentNodeIndex: -1,
                targetNodeIndex: null,
                targetWorldPos: null,
                systemState: 1,
                cognitiveTax: 0,
                baseSpeed:  baseSpeed,
                speed:      baseSpeed,
                velocity:   new THREE.Vector3(),
                isActive:   true,
            };

            this.agentGroup.add(mesh);
            this.agents.push(mesh);
            
            this._pickNextTarget(mesh);
        }

        console.log(`[BehaviorSimulator] ✓ ${this.agents.length} agent(s) placed.`);
        return this.agents;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — step(dt)  [STUB — Phase 2]
    //
    //  Will advance agent positions by dt seconds per frame.
    //  Currently a no-op; call structure is in place for Phase 2.
    //
    //  @param {number} dt  Delta-time in seconds.
    // ═══════════════════════════════════════════════════════════════════════

    step(dt = 0.016) {
        if (!this.spaceSyntaxData || !this._resolvedPositions || this._resolvedPositions.length === 0) return;

        const DECAY_RATE = 0.05; // 5% decay per second
        for (const key in this.stigmergyBuffer) {
            this.stigmergyBuffer[key] *= (1.0 - DECAY_RATE * dt);
            if (this.stigmergyBuffer[key] < 0.01) {
                delete this.stigmergyBuffer[key];
            }
        }

        const SEPARATION_RADIUS = 0.8;
        const SEPARATION_WEIGHT = 1.5;

        for (const mesh of this.agents) {
            // ── Accumulate horizontal separation force ───────────────────
            const separationVector = new THREE.Vector3(0, 0, 0);
            
            for (const otherMesh of this.agents) {
                if (mesh === otherMesh || !otherMesh.userData.isActive) continue;
                
                // STRICT 3D OVERLAP CHECK: Only push if they are on the same vertical level
                const yDiff = Math.abs(mesh.position.y - otherMesh.position.y);
                if (yDiff > 0.4) continue; // Ignore agents on higher/lower steps

                const d = mesh.position.distanceTo(otherMesh.position);
                if (d > 0.01 && d < SEPARATION_RADIUS) {
                    let push = mesh.position.clone().sub(otherMesh.position);
                    push.y = 0; // Separation is strictly horizontal
                    push.normalize().multiplyScalar((SEPARATION_RADIUS - d) / SEPARATION_RADIUS);
                    separationVector.add(push);
                }
            }

            // ── Build unified horizontal intent (target + separation) ───
            let moveX = 0;
            let moveZ = 0;
            let reachedTarget = false;

            const targetPos = mesh.userData.targetWorldPos;
            if (targetPos) {
                const currentPos = mesh.position;
                const dist = currentPos.distanceTo(targetPos);
                const moveStep = mesh.userData.speed * dt;

                // Overshoot Clamp: if close enough, snap and re-pick
                if (moveStep >= dist) {
                    mesh.position.copy(targetPos);
                    mesh.userData.targetNodeIndex = null;
                    mesh.userData.targetWorldPos = null;
                    reachedTarget = true;
                    this._pickNextTarget(mesh);
                } else {
                    // Horizontal intent from target direction
                    const targetDir = targetPos.clone().sub(currentPos);
                    targetDir.y = 0; // Horizontal intent only — Y is handled by floor probe
                    if (targetDir.lengthSq() > 0.0001) {
                        targetDir.normalize();
                        moveX += targetDir.x * moveStep;
                        moveZ += targetDir.z * moveStep;
                    }
                }
            } else {
                // Continuous scanning while stationary
                if (Math.random() < dt * 2.0) { // Approx every 0.5 seconds
                    this._pickNextTarget(mesh);
                }
            }

            // Fold in separation push
            if (separationVector.lengthSq() > 0) {
                const push = separationVector.multiplyScalar(SEPARATION_WEIGHT * dt);
                moveX += push.x;
                moveZ += push.z;
            }

            // ── STRICT 3D PHYSICS: Floor-Snapped Movement ───────────────
            // Only attempt movement if we have horizontal intent and didn't
            // just snap to the target.
            if (!reachedTarget && (Math.abs(moveX) > 0.0001 || Math.abs(moveZ) > 0.0001)) {
                const nextX = mesh.position.x + moveX;
                const nextZ = mesh.position.z + moveZ;

                // Probe the floor geometry at the destination XZ
                const floorCheck = this._getFloorData(nextX, nextZ, mesh.position.y);

                if (floorCheck.valid) {
                    const projectedPos = new THREE.Vector3(nextX, floorCheck.agentY, nextZ);

                    if (this._isPathClear(mesh.position, projectedPos)) {
                        mesh.position.copy(projectedPos);

                        // Keep lookAt horizontal to prevent pitch rotation
                        const lookTarget = mesh.position.clone();
                        lookTarget.x += moveX;
                        lookTarget.z += moveZ;
                        mesh.lookAt(lookTarget);
                    } else {
                        mesh.userData.targetWorldPos = null; // Wall hit
                    }
                } else {
                    mesh.userData.targetWorldPos = null; // Cliff edge or void
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _worldToGrid(pos)
    // ═══════════════════════════════════════════════════════════════════════

    _worldToGrid(pos) {
        if (!this.spaceSyntaxData || !this.spaceSyntaxData.gridConfig) return null;
        const b = this.spaceSyntaxData.gridConfig.bounds;
        const minX = b.min?.x ?? b.minX ?? 0;
        const minZ = b.min?.z ?? b.minZ ?? 0;
        const res = this.spaceSyntaxData.gridConfig.resolution ?? 1.0;
        return {
            gx: Math.floor((pos.x - minX) / res),
            gz: Math.floor((pos.z - minZ) / res)
        };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _getSpatialCongestion(targetPos, excludeAgent)
    // ═══════════════════════════════════════════════════════════════════════

    _getSpatialCongestion(targetPos, excludeAgent) {
        let count = 0;
        const SEARCH_RADIUS = 0.5; // Reduced from 1.2m to prevent lateral wall-to-wall overlaps
        
        for (const other of this.agents) {
            if (other === excludeAgent || !other.userData.isActive) continue;
            
            // STRICT 3D TOPOLOGY: Ignore agents on different vertical levels
            const yDiff = Math.abs(other.position.y - targetPos.y);
            if (yDiff > 0.5) continue; // They are above/below the current step/landing

            if (other.position.distanceTo(targetPos) < SEARCH_RADIUS) {
                count++;
            }
        }
        return count;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _isPathClear(startPos, endPos)
    // ═══════════════════════════════════════════════════════════════════════

    _isPathClear(startPos, endPos) {
        if (!this.raycaster) this.raycaster = new THREE.Raycaster();

        // Drop both endpoints to "waist height" (0.7 m above floor / −0.8 m
        // below the 1.5 m eye-level centre).  This prevents the horizontal
        // collision ray from clipping on stair risers that are below knee-level.
        const origin = startPos.clone();
        origin.y -= 0.8; // 1.5 m eye → 0.7 m waist

        const target = endPos.clone();
        target.y -= 0.8;

        const direction = target.clone().sub(origin);
        const distance  = direction.length();
        if (distance < 0.01) return true;

        direction.normalize();

        this.raycaster.set(origin, direction);
        this.raycaster.near = 0;
        this.raycaster.far  = distance;

        // Intersect against scene geometry, filtering out agents / limbs
        const intersects = this.raycaster.intersectObjects(this.scene.children, true);

        for (const hit of intersects) {
            if (hit.object && !hit.object.name.includes('Agent')) {
                return false; // Wall or obstacle found!
            }
        }
        return true;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _getFloorData(x, z, currentAgentY)
    //
    //  Vertical floor probe.  Casts a downward ray at the given XZ to find
    //  the nearest walkable floor surface.  Returns the correct agent Y
    //  (floor + 1.5 m eye-level offset) if the step height is biomechanically
    //  walkable, otherwise flags the position as invalid (cliff / void).
    //
    //  @returns {{ valid: boolean, agentY: number }}
    // ═══════════════════════════════════════════════════════════════════════

    _getFloorData(x, z, currentAgentY) {
        if (!this.raycaster) this.raycaster = new THREE.Raycaster();

        // Start 0.5 m ABOVE the agent's current head to catch ascending stairs
        const origin = new THREE.Vector3(x, currentAgentY + 0.5, z);
        this.raycaster.set(origin, new THREE.Vector3(0, -1, 0));
        this.raycaster.near = 0;
        this.raycaster.far  = 4.0; // Scan far enough to catch deep drops

        const hits = this.raycaster.intersectObjects(this.scene.children, true);
        for (const hit of hits) {
            if (hit.object && !hit.object.name.includes('Agent')) {
                const floorY    = hit.point.y;
                const newAgentY = floorY + 1.5; // Re-apply 1.5 m eye-level offset
                const heightDiff = newAgentY - currentAgentY;

                // Biomechanics: max step up +0.5 m, max drop down −0.8 m
                if (heightDiff >= -0.8 && heightDiff <= 0.5) {
                    return { valid: true, agentY: newAgentY };
                }
            }
        }
        return { valid: false, agentY: currentAgentY }; // Cliff or void
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _pickNextTarget(agentMesh)
    // ═══════════════════════════════════════════════════════════════════════

    _pickNextTarget(agentMesh) {
        const samplePoints = [];
        const angles = [0, 45, 90, 135, 180, 225, 270, 315];
        const STEP_DIST = 0.4; // Reduced from 0.8 to fit inside narrow stairwells
        
        // Add current pos as the "STAY" option
        samplePoints.push({ pos: agentMesh.position.clone(), isStay: true });
        
        // ── 3D Floor-Snapping Radar ──────────────────────────────────────────
        // For each compass angle, probe the floor at the target XZ using the
        // shared _getFloorData helper.  Only accept walkable elevations.
        for (let a of angles) {
            const rad   = a * Math.PI / 180;
            const testX = agentMesh.position.x + Math.cos(rad) * STEP_DIST;
            const testZ = agentMesh.position.z + Math.sin(rad) * STEP_DIST;

            const floorCheck = this._getFloorData(testX, testZ, agentMesh.position.y);

            if (floorCheck.valid) {
                const samplePos = new THREE.Vector3(testX, floorCheck.agentY, testZ);
                // Only accept if the direct 3-D path to this point is clear of walls
                if (this._isPathClear(agentMesh.position, samplePos)) {
                    samplePoints.push({ pos: samplePos, isStay: false });
                }
            }
        }

        const BASE_CONGESTION_WEIGHT = 0.40;
        const STRESS_THRESHOLD = 0.30;
        
        // Increase congestion penalty by up to 50% based on cognitive tax
        const dynamicCongestionWeight = BASE_CONGESTION_WEIGHT * (1.0 + (agentMesh.userData.cognitiveTax * 0.1));

        let bestUtility = -Infinity;
        let bestSample = null;

        // Line-of-Sight Inverse Distance Weighting (IDW) interpolation.
        // Instead of an absolute height cutoff, we use a raycast to test if a
        // direct line-of-sight exists between the sampled position and each
        // observer node. Nodes blocked by a solid ceiling/floor slab are ignored,
        // preventing floor-bleed. Nodes reachable through open stairwells are
        // included, preserving vertical continuity for staircase navigation.
        if (!this.raycaster) this.raycaster = new THREE.Raycaster();
        const getIntegrationAt = (pos) => {
            if (!this.spaceSyntaxData || !this.spaceSyntaxData.perNode || !this.spaceSyntaxData.perNode.integration) return 0;
            if (!this._resolvedPositions || this._resolvedPositions.length === 0) return 0;

            let sumI = 0;
            let sumWeight = 0;

            for (let i = 0; i < this._resolvedPositions.length; i++) {
                const nodePos = this._resolvedPositions[i];

                // STRICT 3D TOPOLOGY: Line-of-Sight check instead of absolute height limit.
                // Elevate both endpoints by +1.0m to clear low stair risers during the check.
                const sightOrigin = pos.clone();
                sightOrigin.y += 1.0;
                const sightTarget = nodePos.clone();
                sightTarget.y += 1.0;

                const direction = sightTarget.clone().sub(sightOrigin);
                const distance = direction.length();

                let isVisible = true;
                if (distance > 0.1) {
                    direction.normalize();
                    this.raycaster.set(sightOrigin, direction);
                    this.raycaster.near = 0;
                    this.raycaster.far = distance;

                    const hits = this.raycaster.intersectObjects(this.scene.children, true);
                    for (const hit of hits) {
                        if (hit.object && !hit.object.name.includes('Agent')) {
                            isVisible = false; // Blocked by a wall, floor, or ceiling slab
                            break;
                        }
                    }
                }

                if (!isVisible) continue; // Ignore nodes we cannot directly see (prevents floor-bleed)

                const dist = nodePos.distanceTo(pos);
                if (dist < 0.05) return this.spaceSyntaxData.perNode.integration[i]; // Exact match

                const weight = 1.0 / Math.pow(dist, 2); // Inverse square law
                sumI += this.spaceSyntaxData.perNode.integration[i] * weight;
                sumWeight += weight;
            }

            // Guard: if no same-floor nodes were found, return neutral value
            return sumWeight > 0 ? (sumI / sumWeight) : 0;
        };

        for (const sample of samplePoints) {
            const isClear = sample.isStay ? true : this._isPathClear(agentMesh.position, sample.pos);
            
            if (!isClear) {
                // Wall detected. Infinite cognitive friction.
                const U = -Infinity;
                if (U > bestUtility) { bestUtility = U; bestSample = sample; }
                continue; // Skip further calculations for this blocked sample
            }

            const gridCoord = this._worldToGrid(sample.pos);
            const traceKey = gridCoord ? `${gridCoord.gx}_${gridCoord.gz}` : null;
            const traceBonus = (traceKey && this.stigmergyBuffer && this.stigmergyBuffer[traceKey]) ? this.stigmergyBuffer[traceKey] : 0;
            
            const crowd = this._getSpatialCongestion(sample.pos, agentMesh);
            const I = getIntegrationAt(sample.pos);
            
            const U = I - (sample.isStay ? 0 : 0.005) - (crowd * dynamicCongestionWeight) + traceBonus;
            
            if (U > bestUtility) {
                bestUtility = U;
                bestSample = sample;
            }
        }

        const previousState = agentMesh.userData.systemState;
        
        // --- System 1 vs System 2 State Trigger ---
        if (bestUtility < STRESS_THRESHOLD) {
            // The environment sucks everywhere. System 2 Active.
            agentMesh.userData.systemState = 2;
            agentMesh.material.color.setHex(0xff1744); // Red
            agentMesh.userData.speed = agentMesh.userData.baseSpeed * 0.5; // Deliberation is slow
            agentMesh.userData.cognitiveTax = Math.min(5.0, agentMesh.userData.cognitiveTax + 0.1);

            // Stigmergy Write
            const gridCoord = this._worldToGrid(bestSample.pos);
            const traceKey = gridCoord ? `${gridCoord.gx}_${gridCoord.gz}` : null;
            if (gridCoord && !bestSample.isStay) {
                if (!this.stigmergyBuffer) this.stigmergyBuffer = {};
                this.stigmergyBuffer[traceKey] = Math.min(0.5, (this.stigmergyBuffer[traceKey] || 0) + 0.15);
            }
        } else {
            // Agent is comfortable (System 1).
            agentMesh.userData.systemState = 1;
            agentMesh.material.color.setHex(0x00e676); // Green
            agentMesh.userData.speed = agentMesh.userData.baseSpeed;
            agentMesh.userData.cognitiveTax = Math.max(0, agentMesh.userData.cognitiveTax - 0.05);
        }

        console.log(`🧠 [Agent ${agentMesh.userData.agentId}] Action: ${bestSample.isStay ? 'STAY' : 'MOVE'} | BestU: ${bestUtility.toFixed(2)} | System: ${agentMesh.userData.systemState}`);
        
        if (previousState !== agentMesh.userData.systemState) {
            console.log(`⚠️ [Agent ${agentMesh.userData.agentId}] State Change: System ${previousState} ➡️ System ${agentMesh.userData.systemState}`);
        }

        if (bestSample.isStay) {
            agentMesh.userData.targetNodeIndex = null;
            agentMesh.userData.targetWorldPos = null;
        } else {
            agentMesh.userData.targetNodeIndex = -1; // Indicator for legacy loop checks
            agentMesh.userData.targetWorldPos = bestSample.pos;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — setVisible(visible)
    // ═══════════════════════════════════════════════════════════════════════

    setVisible(visible) {
        this.agentGroup.visible = visible;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — clear()
    //  Removes all agent meshes from the scene and resets internal state.
    // ═══════════════════════════════════════════════════════════════════════

    clear() {
        for (const mesh of this.agents) {
            // Dispose per-agent material clone (geometry is shared — don't dispose it)
            if (mesh.material && mesh.material.uuid !== this._agentMaterial?.uuid) {
                mesh.material.dispose();
            }
            this.agentGroup.remove(mesh);
        }
        this.agents = [];
        this.stigmergyBuffer = {};
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — destroy()
    //  Full cleanup — disposes geometry, shared material, and removes group.
    // ═══════════════════════════════════════════════════════════════════════

    destroy() {
        this.clear();
        this._capsuleGeom?.dispose();
        this._agentMaterial?.dispose();
        this.scene.remove(this.agentGroup);
        this._capsuleGeom   = null;
        this._agentMaterial = null;
        console.log('[BehaviorSimulator] Destroyed.');
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _resolveWorldPositions()
    //
    //  Converts spaceSyntaxData.observerPoints (voxel coords) to Three.js
    //  world-space Vector3 positions using the gridConfig bounding box.
    //
    //  Falls back to the raw x/y/z values if no gridConfig is present (in
    //  case the caller already populated worldPos on each observer point).
    //
    //  @returns {Array<THREE.Vector3>}
    // ═══════════════════════════════════════════════════════════════════════

    _resolveWorldPositions() {
        const ssd = this.spaceSyntaxData;
        if (!ssd || !ssd.observerPoints || ssd.observerPoints.length === 0) {
            return [];
        }

        const pts        = ssd.observerPoints;
        const gridConfig = ssd.gridConfig || null;
        const resolution = gridConfig?.resolution ?? 1.0;

        // Bounding-box origin (voxel [0,0,0] maps here)
        let minX = 0, minY = 0, minZ = 0;
        if (gridConfig?.bounds) {
            const b = gridConfig.bounds;
            if (b.min !== undefined) {
                minX = b.min.x ?? 0; minY = b.min.y ?? 0; minZ = b.min.z ?? 0;
            } else {
                minX = b.minX ?? 0; minY = b.minY ?? 0; minZ = b.minZ ?? 0;
            }
        }

        return pts.map(pt => {
            // If worldPos is already cached on the point, prefer it
            if (pt.worldPos) {
                return new THREE.Vector3(pt.worldPos.x, pt.worldPos.y, pt.worldPos.z);
            }
            // Otherwise convert voxel → world
            return new THREE.Vector3(
                minX + pt.x * resolution,
                minY + pt.y * resolution + resolution * 0.5, // raise to half-voxel centre
                minZ + pt.z * resolution
            );
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _ensureSharedResources()
    //  Lazy-creates the shared capsule geometry used by all agents.
    // ═══════════════════════════════════════════════════════════════════════

    _ensureSharedResources() {
        if (this._capsuleGeom) return;

        // THREE.CapsuleGeometry(radius, length, capSegments, radialSegments)
        // length = cylinder shaft height (total height includes the two caps)
        // Available in Three.js r139+. Fall back to CylinderGeometry for older builds.
        if (typeof THREE.CapsuleGeometry !== 'undefined') {
            // shaft length = total height − 2×radius
            const shaft = Math.max(0.01, this._capsuleHeight - 2 * this._capsuleRadius);
            this._capsuleGeom = new THREE.CapsuleGeometry(
                this._capsuleRadius, // radius
                shaft,               // shaft length
                4,                   // cap segments (low-poly for performance)
                8                    // radial segments
            );
        } else {
            // Fallback: plain cylinder
            console.warn('[BehaviorSimulator] THREE.CapsuleGeometry not available — using CylinderGeometry.');
            this._capsuleGeom = new THREE.CylinderGeometry(
                this._capsuleRadius, this._capsuleRadius,
                this._capsuleHeight, 8
            );
        }

        // Translate geometry up by half height so agents sit on the floor plane
        this._capsuleGeom.translate(0, this._capsuleHeight * 0.5, 0);

        // Shared base material (each agent gets a clone with an individual tint)
        this._agentMaterial = new THREE.MeshStandardMaterial({
            color:     0x2979ff,
            emissive:  0x1a237e,
            emissiveIntensity: 0.25,
            metalness: 0.2,
            roughness: 0.6,
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _makeAgentMaterial(agentIndex, totalCount)
    //  Returns a per-agent MeshStandardMaterial with a hue shift.
    //  Uses HSL hue cycling across the agent population so they're
    //  visually distinct but cohesive.
    // ═══════════════════════════════════════════════════════════════════════

    _makeAgentMaterial(agentIndex, totalCount) {
        const hue = (agentIndex / Math.max(1, totalCount)) * 360;
        const color = new THREE.Color();
        color.setHSL(hue / 360, 0.75, 0.55);

        return new THREE.MeshStandardMaterial({
            color,
            emissive:          color.clone().multiplyScalar(0.15),
            emissiveIntensity: 0.3,
            metalness:         0.15,
            roughness:         0.55,
        });
    }
}
