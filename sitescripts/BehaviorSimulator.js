/**
 * Polymère — BehaviorSimulator
 *
 * Управляет автономными агентами; начальное размещение берётся из VGA-узлов
 * Space Syntax (`spaceSyntaxData.observerPoints`).
 *
 * Текущее состояние:
 *   • spawnAgents(count) — создаёт капсульные меши агентов на observer points
 *   • step(dt) @ 60 Hz — локомоция по потенциальным полям Space Syntax
 *   • Architectural Chaos Index (ACI) — fire-and-forget readback из ACISolver @ 5 Hz,
 *     кэшируется в mesh.userData.aci
 *   • MelanopicGlareModule — биологическая световая нагрузка на сетчатку, CPU
 *
 * Зависимости: THREE.js (глобал); ACISolver (стрипается бандлером, в IIFE-скоупередед BehaviorSimulator).
 */
import { ACISolver } from './ACISolver.js';
// MelanopicGlareModule is concatenated before BehaviorSimulator in the bundle
// (build-bundle.ps1 ordering). This import is stripped by the bundler; the
// ESM path resolves it here. Both paths share the same IIFE scope symbol.
import { MelanopicGlareModule } from './MelanopicGlareModule.js';

export class BehaviorSimulator {
    /**
     * @param {THREE.Scene}  scene           — The live Three.js scene.
     * @param {Object}       spaceSyntaxData — Result object from SpaceSyntaxSolver.analyze().
     *   Expected fields used here:
     *     .observerPoints  — Array<{x,y,z}> voxel coords (+ optional .heading, .worldPos).
     *     .gridConfig      — { resolution, bounds: { min: {x,y,z} } }
     *     .perNode         — { integration: Float64Array, connectivity: Float64Array }
     */
    /**
     * @param {Object} [options] — Sprint 3 ACI wiring (optional; ACI is skipped
     *   silently if absent, so the CPU/no-WebGPU path is unaffected).
     * @param {GPUDevice} [options.device]     — falls back to window.AHIModules.device.
     * @param {GPUBuffer} [options.voxelBuffer] — the live stride-8 voxel grid buffer
     *   (owned by SpaceSyntaxVisualizer; the only place it exists).
     */
    constructor(scene, spaceSyntaxData, options = {}) {
        if (!scene) throw new Error('[BehaviorSimulator] scene is required.');

        this.scene           = scene;
        this.spaceSyntaxData = spaceSyntaxData || null;

        // ── ACI (Architectural Chaos Index) cognition wiring [Sprint 3] ─────
        //  Locomotion runs at 60 Hz in step(); ACI geometric tax is read back
        //  from the GPU at 5 Hz (200 ms) and cached onto each mesh.userData.aci.
        //  The readback is fire-and-forget — step() NEVER awaits it, so the
        //  60 Hz loop can't stutter waiting on mapAsync (Commandment #2).
        this.aciSolver          = null;
        this.voxelBuffer        = options.voxelBuffer || null;
        this._aciIntervalMs     = 200;   // 5 Hz cognition tick
        this._aciEnabled        = false;
        this._aciInitialized    = false; // true once ACISolver.initialize() resolves
        this._aciInitPromise    = null;
        this._lastAciTickMs     = 0;

        // ── Melanopic Glare Module (MGM) — Sprint 7 ─────────────────────────
        //  CPU-only; constructed once and shared across all agents.  The
        //  light-source registry defaults to two generic sources; call
        //  window.mgmModule.setLightSources([...]) at any time to update it
        //  with real window/lamp geometry from the loaded IFC model.
        this.mgmModule = new MelanopicGlareModule(
            options.mgmConfig || {} // pass { E_opt, E_max, MGM_max, gamma, … } to override
        );
        if (typeof window !== 'undefined') window.mgmModule = this.mgmModule;

        if (typeof window !== 'undefined') window.dumpACI = () => this.dumpACI();

        const device = options.device
            || (typeof window !== 'undefined' && window.AHIModules && window.AHIModules.device)
            || null;
        const aciGrid = this._resolveACIGridConfig();
        if (device && this.voxelBuffer && aciGrid) {
            this._instantiateACISolver(device, aciGrid);
        } else {
            console.log('[BehaviorSimulator] ACI deferred (need device + voxelBuffer + gridConfig). Call attachACI() when ready.');
        }

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

        // Fixed bubble used only as a fallback until the ACI proxemics field is
        // populated (pre-first-readback, or no WebGPU). Once mesh.userData.aci
        // .proxemics exists, separation is ANISOTROPIC — the comfort radius toward
        // each neighbour follows the room shape (Sprint 4 dynamic proxemics).
        const SEPARATION_RADIUS_FALLBACK = 0.8;
        const SEPARATION_WEIGHT = 1.5;

        for (const mesh of this.agents) {
            // ── Accumulate horizontal separation force ───────────────────
            const separationVector = new THREE.Vector3(0, 0, 0);

            for (const otherMesh of this.agents) {
                if (mesh === otherMesh || !otherMesh.userData.isActive) continue;

                // STRICT 3D OVERLAP CHECK: Only push if they are on the same vertical level
                const yDiff = Math.abs(mesh.position.y - otherMesh.position.y);
                if (yDiff > 0.4) continue; // Ignore agents on higher/lower steps

                const dx = otherMesh.position.x - mesh.position.x;
                const dz = otherMesh.position.z - mesh.position.z;
                const d  = mesh.position.distanceTo(otherMesh.position);

                // Sample THIS agent's personal-space radius in the direction of
                // the neighbour: intimate (~0.4 m) toward walls, social (~3.0 m)
                // toward openings. Falls back to the fixed bubble pre-ACI.
                const sepRadius = this._proxemicRadius(mesh, dx, dz, SEPARATION_RADIUS_FALLBACK);

                if (d > 0.01 && d < sepRadius) {
                    let push = mesh.position.clone().sub(otherMesh.position);
                    push.y = 0; // Separation is strictly horizontal
                    push.normalize().multiplyScalar((sepRadius - d) / sepRadius);
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

        // ── 5 Hz cognition tick (ACI geometric tax) ─────────────────────────
        //  Runs AFTER locomotion so it snapshots post-move positions/facings.
        //  Fire-and-forget: never awaited here, so the 60 Hz loop is untouched.
        this._maybeRunCognitionTick();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — attachACI({ device, voxelBuffer })
    //  Late-binds the ACI GPU dependencies when they weren't available at
    //  construction (e.g. WebGPU came up after the sim was created). Idempotent.
    // ═══════════════════════════════════════════════════════════════════════
    attachACI({ device, voxelBuffer } = {}) {
        if (voxelBuffer) this.voxelBuffer = voxelBuffer;
        const dev = device
            || (typeof window !== 'undefined' && window.AHIModules && window.AHIModules.device)
            || null;
        const aciGrid = this._resolveACIGridConfig();
        if (this.aciSolver) return true; // already wired
        if (dev && this.voxelBuffer && aciGrid) {
            this._instantiateACISolver(dev, aciGrid);
            return this._aciEnabled;
        }
        console.warn('[BehaviorSimulator] attachACI: still missing device/voxelBuffer/gridConfig.');
        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — dumpACI()   (console.table of every agent's cached ACI)
    //  Exposed as window.dumpACI() for live debugging (Commandment #4).
    // ═══════════════════════════════════════════════════════════════════════
    dumpACI() {
        if (!this.agents || this.agents.length === 0) {
            console.warn('[BehaviorSimulator] dumpACI: no agents spawned.');
            return [];
        }
        const rows = this.agents.map((m) => {
            const a = m.userData && m.userData.aci;
            if (!a) return { agent: m.name, systemState: m.userData?.systemState, aci: 'not yet read' };
            const prox = a.proxemics;
            let pMin = null, pMax = null, pMean = null;
            if (prox && prox.length) {
                pMin = Math.min(...prox); pMax = Math.max(...prox);
                pMean = this._meanProxemicRadius(m, null);
            }
            return {
                agent:           m.name,
                systemState:     m.userData.systemState,
                prospect_m:      +a.prospect.toFixed(2),
                dBack_m:         +a.dBack.toFixed(2),
                compactness:     +a.compactness.toFixed(3),
                refugeProtected: a.refugeProtected,
                prox_min_m:      pMin !== null ? +pMin.toFixed(2) : '—',
                prox_max_m:      pMax !== null ? +pMax.toFixed(2) : '—',
                prox_mean_m:     pMean !== null ? +pMean.toFixed(2) : '—',
                // ── Sprint 7: MGM biological lighting fields ──────────────────
                E_ambient:       a.mgm ? +a.mgm.E_ambient.toFixed(2)    : '—',
                E_mel:           a.mgm ? +a.mgm.E_mel.toFixed(2)        : '—',
                mgm:             a.mgm ? +a.mgm.mgm.toFixed(3)          : '—',
                aciBase:         a.mgm ? +a.mgm.aciBase.toFixed(3)      : '—',
                aciFinal:        a.mgm ? +a.mgm.aciFinal.toFixed(3)     : '—',
                inDarkness:      a.mgm ? a.mgm.inDarkness               : '—',
            };
        });
        console.table(rows);
        return rows;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _proxemicRadius(mesh, dx, dz, fallback)
    //  Anisotropic personal-space radius (metres) in the XZ-plane direction
    //  (dx,dz) toward a neighbour, sampled from the ACI proxemics field on
    //  mesh.userData.aci. Nearest-bucket over proxDirs directions, where index
    //  d ↔ angle d·(2π/D), d=0 → +X, CCW toward +Z — matching atan2(dz,dx) and
    //  the WGSL ray layout. Returns `fallback` until the first readback lands.
    // ═══════════════════════════════════════════════════════════════════════
    _proxemicRadius(mesh, dx, dz, fallback) {
        const aci   = mesh.userData && mesh.userData.aci;
        const field = aci && aci.proxemics;
        if (!field || field.length === 0) return fallback;
        const D = field.length;
        let theta = Math.atan2(dz, dx);        // [-π, π]
        if (theta < 0) theta += Math.PI * 2;    // → [0, 2π)
        const bucket = Math.round(theta / ((Math.PI * 2) / D)) % D; // wraps D→0
        const r = field[bucket];
        return (Number.isFinite(r) && r > 0) ? r : fallback;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _meanProxemicRadius(mesh, fallback)
    //  Direction-averaged personal-space radius → an isotropic openness measure
    //  for congestion sensing. Returns `fallback` until ACI is populated.
    // ═══════════════════════════════════════════════════════════════════════
    _meanProxemicRadius(mesh, fallback) {
        const field = mesh && mesh.userData && mesh.userData.aci && mesh.userData.aci.proxemics;
        if (!field || field.length === 0) return fallback;
        let sum = 0, n = 0;
        for (let i = 0; i < field.length; i++) {
            if (Number.isFinite(field[i])) { sum += field[i]; n++; }
        }
        return n > 0 ? sum / n : fallback;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _aciCognition(mesh)   [Sprint 5]
    //
    //  Decodes the 5 Hz GPU ACI readback cached on mesh.userData.aci into the
    //  cognitive terms the utility function folds in and the System-1/2 trigger
    //  reads. Consumes ONLY cached data — no GPU work, no await — so cognition
    //  stays fully decoupled from the 60 Hz kinematic clock (Commandment #2).
    //
    //  Returns (all utility terms in the same scale as Space Syntax integration):
    //    • utilityBias      prospectBonus + refugeBonus  (both ≥ 0 comforts)
    //    • forceSystem2     genuine spatial stress — crowding tax, spatial
    //                       confinement, or a severe prospect/refuge deficit.
    //                       (Affordance Theory was removed: visual permeability
    //                       without passage is a Prospect feature, not stress.)
    //    • meanProx         mean anisotropic comfort radius (openness proxy)
    //    • tightness        1 → intimate/enclosed, ↓ toward social/open
    //    • crowdTax         proxemic crowding at the agent's current stance
    //    • confined         corridor-tight isovist (low compactness) + no view
    //    • exposedBlind     worst prospect-refuge quadrant (blind AND unprotected)
    //  hasData is false until the first readback lands (spawn / no-WebGPU path),
    //  in which case every term is neutral and behaviour falls back to Sprint 4.
    // ═══════════════════════════════════════════════════════════════════════
    _aciCognition(mesh) {
        const aci = mesh && mesh.userData && mesh.userData.aci;
        const meanProxFallback = 0.8;
        if (!aci) {
            return {
                hasData: false, utilityBias: 0, forceSystem2: false,
                prospectBonus: 0, refugeBonus: 0,
                crowdTax: 0, meanProx: meanProxFallback, tightness: 1.0,
                confined: false, exposedBlind: false,
            };
        }

        // Prospect–Refuge Theory (Appleton 1975): spatial comfort rises with an
        // open forward view (prospect) AND a protected back (refuge). Both lower
        // the cognitive tax, so they read as a positive utility bias. prospect is
        // Σ forward-ray depth (m) → saturate so a cavernous atrium can't dwarf I.
        const PROSPECT_SCALE_M = 12.0; // forward reach that earns the full bonus
        const prospectBonus = Math.min(0.35, Math.max(0, aci.prospect) / PROSPECT_SCALE_M * 0.35);
        const refugeBonus   = aci.refugeProtected ? 0.20 : 0.0;

        // Proxemic crowding at the agent's CURRENT stance. Neighbours inside the
        // comfort field cost more when that field is TIGHT (enclosed space →
        // small mean radius); crowding in an open plaza barely registers.
        const meanProx   = this._meanProxemicRadius(mesh, meanProxFallback);
        const tightness  = Math.min(1.5, 0.8 / Math.max(0.4, meanProx)); // →1.5 intimate, ↓ social
        const neighbours = this._getSpatialCongestion(mesh.position, mesh);
        const crowdTax   = neighbours * 0.18 * tightness;

        // Spatial stress signals that (unlike glazing) DO warrant System-2:
        //  • prospectDeficit — the eye reaches almost nowhere ahead (enclosed/blind).
        //  • confined        — corridor-tight isovist (low compactness) AND blind.
        //  • exposedBlind    — worst prospect-refuge quadrant: blind AND no refuge.
        const prospectDeficit = prospectBonus < 0.08;
        const confined     = Number.isFinite(aci.compactness) && aci.compactness < 0.12 && prospectDeficit;
        const exposedBlind = prospectDeficit && !aci.refugeProtected;

        const utilityBias  = prospectBonus + refugeBonus;
        const forceSystem2 = crowdTax > 0.45 || confined || exposedBlind;

        return {
            hasData: true, utilityBias, forceSystem2,
            prospectBonus, refugeBonus,
            crowdTax, meanProx, tightness,
            confined, exposedBlind,
        };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _resolveACIGridConfig()
    //  ACISolver needs { dimensions:{nx,ny,nz}, resolution, bounds }. The live
    //  spaceSyntaxData.gridConfig already carries exactly that (flattened bounds
    //  are handled inside ACISolver._worldToVoxel), so we pass it through.
    // ═══════════════════════════════════════════════════════════════════════
    _resolveACIGridConfig() {
        const gc = this.spaceSyntaxData && this.spaceSyntaxData.gridConfig;
        if (gc && gc.dimensions && gc.dimensions.nx && gc.resolution) return gc;
        return null;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _instantiateACISolver(device, gridConfig)
    //  Constructs the solver (cheap) but DEFERS the async initialize() (shader
    //  compile + pipeline) to the first cognition tick, since the constructor
    //  can't be async. Exposes it globally for console debugging (Commandment #4).
    // ═══════════════════════════════════════════════════════════════════════
    _instantiateACISolver(device, gridConfig) {
        try {
            this.aciSolver   = new ACISolver(device, gridConfig, {});
            this._aciEnabled = true;
            if (typeof window !== 'undefined') window.aciSolver = this.aciSolver;
            console.log('[BehaviorSimulator] ACISolver instantiated (init deferred to first cognition tick).');
        } catch (e) {
            this.aciSolver   = null;
            this._aciEnabled = false;
            console.warn('[BehaviorSimulator] ACISolver instantiation failed; ACI disabled:', e.message);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PRIVATE — _maybeRunCognitionTick()
    //  5 Hz throttle → lazy async init → guarded fire-and-forget GPU readback.
    //  Every early-return path keeps the 60 Hz loop non-blocking.
    // ═══════════════════════════════════════════════════════════════════════
    _maybeRunCognitionTick() {
        if (!this._aciEnabled || !this.aciSolver || !this.voxelBuffer) return;

        const now = (typeof performance !== 'undefined' && performance.now)
            ? performance.now() : Date.now();
        if (now - this._lastAciTickMs < this._aciIntervalMs) return; // < 200 ms → skip

        // Lazy one-time async init. Kick it off, then wait for it to resolve.
        if (!this._aciInitialized) {
            if (!this._aciInitPromise) {
                this._aciInitPromise = this.aciSolver.initialize()
                    .then(() => { this._aciInitialized = true; })
                    .catch((e) => {
                        this._aciEnabled = false; // structural failure → stop retrying
                        console.warn('[BehaviorSimulator] ACI initialize() failed; ACI disabled:', e.message);
                    });
            }
            return; // don't compute until initialized
        }

        // Guard: never overlap a readback still mapped in flight (Commandment #5).
        if (this.aciSolver._mapInFlight) return;

        // Snapshot active agents' world position + facing at fire time. Hold the
        // mesh refs locally so results map back correctly even if this.agents
        // mutates before the async readback resolves.
        const meshes = this.agents.filter((m) => m.userData && m.userData.isActive);
        if (meshes.length === 0) return;

        const dir = new THREE.Vector3();
        const payload = new Array(meshes.length);
        for (let i = 0; i < meshes.length; i++) {
            const m = meshes[i];
            m.getWorldDirection(dir); // +Z world axis = the direction lookAt() faces
            payload[i] = {
                x: m.position.x, y: m.position.y, z: m.position.z,
                fx: dir.x, fy: dir.y, fz: dir.z,
            };
        }

        this._lastAciTickMs = now; // start the next 200 ms window from the fire time

        // Fire-and-forget. computeGeometry() awaits mapAsync internally; we do
        // NOT await it here, so locomotion continues uninterrupted.
        this.aciSolver.computeGeometry(this.voxelBuffer, payload)
            .then((results) => {
                if (!results || results.length === 0) return; // null = throttled tick
                for (let i = 0; i < results.length && i < meshes.length; i++) {
                    const mesh = meshes[i];
                    if (!mesh || !mesh.userData) continue;
                    mesh.userData.aci = results[i];            // Sprint 3 deliverable
                    // cognitiveTax is now a purely behavioural accumulator
                    // (raised in System 2, decayed in System 1 by _pickNextTarget);
                    // it is no longer seeded from the removed affordance gap.

                    // ── Sprint 7: Melanopic Glare Multiplier (MGM) ───────────
                    //  Compute the biological light load for this agent using
                    //  their world position and gaze direction as snapshotted in
                    //  `payload[i]` at fire time (same positions used by the GPU
                    //  raycaster above, so everything is temporally consistent).
                    //
                    //  This is pure CPU math (~μs), never awaited, and does not
                    //  touch the GPU or the 60 Hz locomotion loop.
                    if (this.mgmModule) {
                        const p = payload[i]; // { x, y, z, fx, fy, fz }
                        const agentPos = { x: p.x, y: p.y, z: p.z };
                        const gazeDir  = { x: p.fx, y: p.fy, z: p.fz };

                        // D_vis = prospect depth from GPU raycaster (metres).
                        // Pass it so the darkness clamp can physically reduce it.
                        const D_vis = results[i].prospect;

                        // aciBase: use compactness as a proxy for the aggregate
                        // geometric-stress scalar, clamped to (0, 1].
                        // compactness = 0 → perfect elongated corridor (high stress)
                        // compactness = 1 → perfectly compact isovist (low stress)
                        // We invert it so aciBase = 1 means maximum geometric stress.
                        const compactness = results[i].compactness;
                        const aciBase = Math.max(0.001,
                            Number.isFinite(compactness) ? (1.0 - compactness) : 0.5
                        );

                        const mgmResult = this.mgmModule.computeForAgent(
                            agentPos, gazeDir, aciBase, D_vis
                        );

                        // Store full MGM result alongside the geometric ACI data.
                        // Downstream logic (_aciCognition, dumpACI) reads these.
                        mesh.userData.aci.mgm = mgmResult;
                    }
                }
            })
            .catch((e) => {
                console.warn('[BehaviorSimulator] ACI cognition tick error:', e.message);
            });
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
        // Sprint 4: congestion sensing scales with the querying agent's openness.
        // Mean anisotropic proxemic radius → tight (~0.4 m) in enclosed space,
        // social (up to ~3 m) in open space. Falls back to the legacy 0.5 m.
        const SEARCH_RADIUS = this._meanProxemicRadius(excludeAgent, 0.5);

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

        // ── Sprint 5: cognitive terms from the 5 Hz GPU ACI readback ─────────
        //  Cached-only decode (no GPU, no await) → decoupled from the 60 Hz clock.
        const aciCog = this._aciCognition(agentMesh);

        // Congestion penalty scales with (a) accumulated cognitive tax and (b)
        // proxemic tightness — the SAME crowd is far more taxing in an enclosed
        // corridor (tightness→1.5) than an open plaza (tightness↓).
        const dynamicCongestionWeight =
            BASE_CONGESTION_WEIGHT
            * (1.0 + (agentMesh.userData.cognitiveTax * 0.1))
            * aciCog.tightness;

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

            // Prospect-seeking steer: bias toward headings whose anisotropic
            // comfort radius is wider than the mean (i.e. toward openings/views).
            // Only the proxemics field is directional, so it's the one ACI term
            // that can legitimately vary per candidate sample; prospect & refuge
            // describe the agent's current stance and enter uniformly via
            // aciCog.utilityBias below.
            let opennessBonus = 0;
            if (aciCog.hasData && !sample.isStay) {
                const ddx = sample.pos.x - agentMesh.position.x;
                const ddz = sample.pos.z - agentMesh.position.z;
                const dirProx = this._proxemicRadius(agentMesh, ddx, ddz, aciCog.meanProx);
                opennessBonus = Math.max(-0.15, Math.min(0.15, (dirProx - aciCog.meanProx) * 0.12));
            }

            const U = I
                + aciCog.utilityBias            // prospect + refuge comfort (per-agent)
                + opennessBonus                 // directional prospect-seeking (per-sample)
                - (sample.isStay ? 0 : 0.005)
                - (crowd * dynamicCongestionWeight)
                + traceBonus;

            if (U > bestUtility) {
                bestUtility = U;
                bestSample = sample;
            }
        }

        const previousState = agentMesh.userData.systemState;

        // --- System 1 vs System 2 State Trigger [Sprint 5, refactored Sprint 6] ---
        //  Not a pure bestUtility heuristic: genuine spatial stress from the ACI
        //  readback can force System 2 on its own — heavy proxemic crowding,
        //  spatial confinement, or a severe prospect/refuge deficit. Visual
        //  permeability (glass / doors / atria) is NOT stress and never triggers.
        const stressed = (bestUtility < STRESS_THRESHOLD) || aciCog.forceSystem2;

        if (stressed) {
            // System 2 — deliberate, slow replanning.
            agentMesh.userData.systemState = 2;
            agentMesh.material.color.setHex(0xff1744); // Red — genuine spatial stress
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

        // Compact reason tag for the live console: which genuine stressor (if any) fired.
        const stressReason = aciCog.exposedBlind ? 'EXPOSED-BLIND'
            : aciCog.confined ? 'CONFINED'
            : aciCog.crowdTax > 0.45 ? 'CROWDED' : '';
        const aciTag = aciCog.hasData
            ? ` | ACI[bias:${aciCog.utilityBias.toFixed(2)} crowd:${aciCog.crowdTax.toFixed(2)}${stressReason ? ' ' + stressReason : ''}]`
            : ' | ACI[pending]';
        console.log(`🧠 [Agent ${agentMesh.userData.agentId}] Action: ${bestSample.isStay ? 'STAY' : 'MOVE'} | BestU: ${bestUtility.toFixed(2)} | System: ${agentMesh.userData.systemState}${aciTag}`);

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
        // Release ACI GPU buffers/pipeline (persistent — must be torn down here).
        if (typeof window !== 'undefined' && window.aciSolver === this.aciSolver) {
            window.aciSolver = null;
        }
        this.aciSolver?.destroy?.();
        this.aciSolver       = null;
        this._aciEnabled     = false;
        this._aciInitialized = false;
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
