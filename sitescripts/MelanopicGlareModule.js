/**
 * Polymère — MelanopicGlareModule
 *
 * Вычисляет Melanopic Glare Multiplier (MGM) — коэффициент биологической
 * световой нагрузки, который масштабирует базовый ACI-балл (Architectural Chaos Index)
 * по закону Стивенса (Stevens' Power Law, гамма ³). CPU-only.
 *
 * Key steps performed per-agent:
 *
 *   1. Retinal Load (E_mel):
 *      For each active light source (window / lamp) the module accumulates
 *      a distance-attenuated, gaze-direction-weighted contribution.
 *
 *        E_mel = Σ [ (I_base * W_mel / d²) * max(0, dot(V_gaze, V_light)) ]
 *
 *      Where:
 *        I_base   — source base illuminance in Lux
 *        W_mel    — melanopic weight  (≈1.5 daylight/windows, ≈0.4 warm-lamp)
 *        d        — Euclidean distance from agent to source (metres)
 *        V_gaze   — normalised forward gaze vector of the agent
 *        V_light  — normalised vector from agent → source centroid
 *
 *   2. MGM (Stevens' Power Law, piecewise):
 *      Comfort zone  (E_mel ≤ E_opt):
 *        MGM = 1.0 - 0.15 * (E_mel / E_opt)        → [1.0 … 0.85]
 *      Glare zone    (E_mel > E_opt):
 *        MGM = 0.85 + (MGM_max - 0.85) * pow( (clamp(E_mel, E_opt, E_max) - E_opt)
 *                                               / (E_max - E_opt),  gamma )
 *
 *   3. ACI application:
 *        ACI_final = pow(ACI_base, 1.0 / MGM)
 *
 *   4. Darkness clamp (D_vis guard):
 *      When E_mel falls below `darknessThreshold` the agent is in near-total
 *      darkness; their visual prospect depth (D_vis) is physically clamped to
 *      `darknessMaxDvis` metres, which increases the base ACI (less prospect
 *      = more geometric stress before the MGM is applied).
 *
 * No GPU, no THREE dependency — pure CPU math.  Designed to run inside the
 * 5 Hz cognition tick of BehaviorSimulator._maybeRunCognitionTick() and
 * also inside HeadlessACIPipeline.run(), both of which already await async
 * work separately and call this synchronously.
 *
 * No ES imports — self-contained for build-bundle.ps1 concatenation.
 *
 * ════════════════════════════════════════════════════════════════════════
 */

// ─────────────────────────────────────────────────────────────────────────────
//  MODULE-LEVEL CONSTANTS  (shared so factory helpers can reference them)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Melanopic weight constants.
 * References: CIE S 026:2018, Rea et al. (2012), WELL Building Standard v2.
 *   W_mel ≈ 1.5  — Daylight / north-facing windows (peak ipRGC at 480 nm)
 *   W_mel ≈ 0.4  — Warm-white artificial lamp (≤3000 K, low melanopic content)
 *   W_mel ≈ 0.85 — Neutral-white LED (4000 K, moderate melanopic content)
 */
const MGM_W_MEL_DAYLIGHT  = 1.5;
const MGM_W_MEL_WARM_LAMP = 0.4;
const MGM_W_MEL_LED_4000  = 0.85;

// ─────────────────────────────────────────────────────────────────────────────
//  MelanopicGlareModule
// ─────────────────────────────────────────────────────────────────────────────
export class MelanopicGlareModule {

    /**
     * @param {Object} [config]  Override any of the default MGM parameters.
     *
     *   E_opt             — Optimal comfort illuminance [Lux].
     *                       Below this, light REDUCES geometric stress (MGM < 1).
     *                       Default: 300 lux  (EN 12464-1 office comfort target).
     *
     *   E_max             — Painful glare threshold [Lux].
     *                       At and above this, MGM saturates at MGM_max.
     *                       Default: 3000 lux (direct sunbeam through window).
     *
     *   MGM_max           — Maximum multiplier at E_max (Stevens' saturation).
     *                       Default: 3.0  (triples ACI stress at extreme glare).
     *
     *   gamma             — Stevens' power-law exponent for the glare ramp.
     *                       Default: 2.0  (quadratic onset, matches psychophysics).
     *
     *   darknessThreshold — E_mel below this → agent is in near-total darkness [Lux].
     *                       Default: 10 lux (scotopic / mesopic boundary).
     *
     *   darknessMaxDvis   — D_vis is clamped to this when agent is in darkness [m].
     *                       Default: 2.0 m  (≈ arm's length — loss of prospect).
     *
     *   minSourceDist     — Minimum distance clamp for d² denominator (prevents
     *                       singularity when agent is at/inside a source).
     *                       Default: 0.5 m.
     */
    constructor(config = {}) {
        this.config = {
            E_opt:              config.E_opt              ?? 300.0,
            E_max:              config.E_max              ?? 3000.0,
            MGM_max:            config.MGM_max            ?? 3.0,
            gamma:              config.gamma              ?? 2.0,
            darknessThreshold:  config.darknessThreshold  ?? 10.0,
            darknessMaxDvis:    config.darknessMaxDvis    ?? 2.0,
            minSourceDist:      config.minSourceDist      ?? 0.5,
        };

        // ── Default light-source registry ────────────────────────────────────
        // Each entry is a "virtual" source representing an aperture or fixture.
        // Replace via setLightSources() once actual geometry is known.
        //
        // Shape: { id:string, x, y, z, I_base:Lux, W_mel:float, type:string }
        //   id      — unique string for debugging / upsert by id
        //   x,y,z   — world-space centroid of the source face (metres)
        //   I_base  — photopic base illuminance at source face [Lux]
        //   W_mel   — melanopic weight (use MGM_W_MEL_* constants or a float)
        //   type    — 'window' | 'lamp' | 'skylight'  (cosmetic only)
        //
        // These defaults represent a generic single-story space with one
        // south-facing daylight window and one overhead warm-lamp zone.
        this._lightSources = [
            {
                id:     'default_window_south',
                x:       0.0,
                y:       1.5,    // centroid at typical window head height
                z:      -5.0,    // south wall (−Z in Three.js convention)
                I_base:  5000.0, // exterior illuminance through glazing (τ≈65%)
                W_mel:   MGM_W_MEL_DAYLIGHT,
                type:   'window',
            },
            {
                id:     'default_lamp_overhead',
                x:       0.0,
                y:       3.0,    // ceiling
                z:       0.0,
                I_base:  500.0,  // typical office overhead lamp
                W_mel:   MGM_W_MEL_WARM_LAMP,
                type:   'lamp',
            },
        ];
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — setLightSources(sources)
    //
    //  Replace the entire light-source registry.  Call this from preview.html
    //  or ProjectManager after SpectralTracer / RoomDetector have identified
    //  actual windows and lamps for the loaded IFC model.
    //
    //  @param {Array<{id, x, y, z, I_base, W_mel, type}>} sources
    // ═══════════════════════════════════════════════════════════════════════
    setLightSources(sources) {
        if (!Array.isArray(sources) || sources.length === 0) {
            console.warn('[MelanopicGlareModule] setLightSources: empty or invalid array — keeping current sources.');
            return;
        }
        this._lightSources = sources.map(s => ({
            id:     s.id     || 'unnamed',
            x:      Number(s.x)      || 0,
            y:      Number(s.y)      || 0,
            z:      Number(s.z)      || 0,
            I_base: Number(s.I_base) || 0,
            W_mel:  s.W_mel !== undefined ? Number(s.W_mel) : MGM_W_MEL_DAYLIGHT,
            type:   s.type   || 'window',
        }));
        console.log(`[MelanopicGlareModule] Light-source registry updated: ${this._lightSources.length} source(s).`);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — addLightSource(source)
    //
    //  Append or upsert (by id) a single source.  Use for streaming window
    //  detection events from the voxelizer / IFC parser.
    //
    //  @param {{ id, x, y, z, I_base, W_mel, type }} source
    // ═══════════════════════════════════════════════════════════════════════
    addLightSource(source) {
        const existing = this._lightSources.findIndex(s => s.id === source.id);
        const entry = {
            id:     source.id     || 'unnamed',
            x:      Number(source.x)      || 0,
            y:      Number(source.y)      || 0,
            z:      Number(source.z)      || 0,
            I_base: Number(source.I_base) || 0,
            W_mel:  source.W_mel !== undefined ? Number(source.W_mel) : MGM_W_MEL_DAYLIGHT,
            type:   source.type   || 'window',
        };
        if (existing !== -1) {
            this._lightSources[existing] = entry; // upsert
        } else {
            this._lightSources.push(entry);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — getLightSources()
    //  Returns a shallow copy of the current registry (for debugging).
    // ═══════════════════════════════════════════════════════════════════════
    getLightSources() {
        return this._lightSources.slice();
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — computeRetinalLoad(agentPos, gazeDir, [sources], [grid])
    //
    //  Step 1: Accumulate E_mel (melanopic retinal load) for a single agent
    //  by summing contributions from all registered (or provided) light sources.
    //
    //  If `grid` is provided, a 3D ray marching (Bresenham/DDA) check is
    //  performed between the agent and each light source. Light from sources
    //  occluded by solid walls/slabs is ignored.
    //
    //  Formula per source:
    //    contribution = (I_base * W_mel / max(minSourceDist, d)²)
    //                   * max(0, dot(V_gaze_norm, V_light_norm))
    //
    //  @param {{ x:number, y:number, z:number }} agentPos
    //    World-space eye position (metres).
    //
    //  @param {{ x:number, y:number, z:number }} gazeDir
    //    Forward gaze vector of the agent (need not be normalised; will be
    //    re-normalised internally to guard against floating-point drift).
    //
    //  @param {Array|undefined} [sources]
    //    Optional override source list for room-aware filtering.
    //
    //  @param {{ voxelData: Float32Array, gridConfig: Object }|undefined} [grid]
    //    Optional CPU voxel grid reference for line-of-sight occlusion checks.
    //
    //  @returns {{ E_ambient: number, E_mel: number }} 
    //    E_ambient (unoccluded ambient Lux) and E_mel (directional melanopic Lux).
    // ═══════════════════════════════════════════════════════════════════════
    computeRetinalLoad(agentPos, gazeDir, sources, grid) {
        const srcs = sources || this._lightSources;
        if (!srcs || srcs.length === 0) return 0.0;

        const { minSourceDist } = this.config;

        // Normalise the gaze vector defensively — THREE.js lookAt() is usually
        // unit-length but accumulation of floating-point error over many ticks
        // can introduce sub-percent drift.
        const gLen = Math.sqrt(
            gazeDir.x * gazeDir.x +
            gazeDir.y * gazeDir.y +
            gazeDir.z * gazeDir.z
        );
        const gx = gLen > 1e-6 ? gazeDir.x / gLen : 0;
        const gy = gLen > 1e-6 ? gazeDir.y / gLen : 0;
        const gz = gLen > 1e-6 ? gazeDir.z / gLen : 0;

        let E_mel = 0.0;
        let E_ambient = 0.0;

        for (let k = 0; k < srcs.length; k++) {
            const src = srcs[k];

            // Vector from agent eye → source centroid
            const vx = src.x - agentPos.x;
            const vy = src.y - agentPos.y;
            const vz = src.z - agentPos.z;

            // Distance (clamped to avoid 1/d² singularity at source centroid)
            const d = Math.max(minSourceDist,
                Math.sqrt(vx * vx + vy * vy + vz * vz));

            // Check occlusion if grid is provided
            let occluded = false;
            if (grid && grid.voxelData && grid.gridConfig) {
                const lightPos = { x: src.x, y: src.y, z: src.z };
                occluded = this.checkOcclusion(agentPos, lightPos, grid);
            }

            if (!occluded) {
                // Ambient illuminance ignores directionality and melanopic weights
                E_ambient += (src.I_base / (d * d));

                // Normalised direction to source (V_light)
                const vlx = vx / d;
                const vly = vy / d;
                const vlz = vz / d;

                // Directionality: retinal exposure only for light arriving in front
                // of the gaze plane.  Light hitting from behind (dot < 0) contributes 0.
                const dotProduct = Math.max(0.0, gx * vlx + gy * vly + gz * vlz);

                if (dotProduct > 0) {
                    // Accumulate: inverse-square law × melanopic weight × directionality
                    E_mel += (src.I_base * src.W_mel / (d * d)) * dotProduct;
                }
            }
        }

        return { E_ambient, E_mel };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — checkOcclusion(startPos, endPos, grid)
    //
    //  Fast Voxel Ray Marching (Amanatides & Woo 3D DDA algorithm)
    //  Casts a discrete line from startPos to endPos. If it hits an opaque
    //  solid voxel (e.g. WALL, SLAB), returns true (occluded).
    //  Allows light to pass through EMPTY, FLUID, GLASS, and WINDOW.
    //
    //  @param {{ x, y, z }} startPos
    //  @param {{ x, y, z }} endPos
    //  @param {{ voxelData: Float32Array, gridConfig: Object }} grid
    //  @returns {boolean} true if line-of-sight is blocked by a wall
    // ═══════════════════════════════════════════════════════════════════════
    checkOcclusion(startPos, endPos, grid) {
        if (!grid || !grid.voxelData || !grid.gridConfig) return false;
        
        const { voxelData, gridConfig } = grid;
        const res = gridConfig.resolution;
        const b = gridConfig.bounds;
        const { nx, ny, nz } = gridConfig.dimensions;

        // Convert world coordinates to voxel indices
        const worldToVoxel = (x, y, z) => ({
            vx: Math.floor((x - b.minX) / res),
            vy: Math.floor((y - b.minY) / res),
            vz: Math.floor((z - b.minZ) / res)
        });

        const startV = worldToVoxel(startPos.x, startPos.y, startPos.z);
        const endV = worldToVoxel(endPos.x, endPos.y, endPos.z);

        // Ray direction
        const dx = endPos.x - startPos.x;
        const dy = endPos.y - startPos.y;
        const dz = endPos.z - startPos.z;

        // Current voxel coordinates
        let currentX = startV.vx;
        let currentY = startV.vy;
        let currentZ = startV.vz;

        // Step directions
        const stepX = dx > 0 ? 1 : (dx < 0 ? -1 : 0);
        const stepY = dy > 0 ? 1 : (dy < 0 ? -1 : 0);
        const stepZ = dz > 0 ? 1 : (dz < 0 ? -1 : 0);

        // How far we have to move along the ray to cross a voxel boundary
        const tDeltaX = dx !== 0 ? Math.abs(res / dx) : Infinity;
        const tDeltaY = dy !== 0 ? Math.abs(res / dy) : Infinity;
        const tDeltaZ = dz !== 0 ? Math.abs(res / dz) : Infinity;

        // Initial intersection with voxel boundaries
        let tMaxX = Infinity, tMaxY = Infinity, tMaxZ = Infinity;

        if (dx !== 0) {
            const nextX = currentX + (stepX > 0 ? 1 : 0);
            const voxelBoundaryX = b.minX + nextX * res;
            tMaxX = (voxelBoundaryX - startPos.x) / dx;
        }
        if (dy !== 0) {
            const nextY = currentY + (stepY > 0 ? 1 : 0);
            const voxelBoundaryY = b.minY + nextY * res;
            tMaxY = (voxelBoundaryY - startPos.y) / dy;
        }
        if (dz !== 0) {
            const nextZ = currentZ + (stepZ > 0 ? 1 : 0);
            const voxelBoundaryZ = b.minZ + nextZ * res;
            tMaxZ = (voxelBoundaryZ - startPos.z) / dz;
        }

        const VOXEL_STRIDE = 8;
        // Voxel states: EMPTY(0), SOLID(1), FLUID(2), GLASS(4), WINDOW(16)
        // Treat 0, 2, 4, 16 as transparent. All others (like SOLID, BOUNDARY, DOOR) as opaque.
        const isTransparent = (state) => (state === 0 || state === 2 || state === 4 || state === 16);

        let reached = false;
        // Limit steps to prevent infinite loops from floating point errors
        let maxSteps = nx + ny + nz; 

        while (!reached && maxSteps-- > 0) {
            // Check bounds
            if (currentX >= 0 && currentX < nx &&
                currentY >= 0 && currentY < ny &&
                currentZ >= 0 && currentZ < nz) {
                
                const idx = currentX + currentY * nx + currentZ * nx * ny;
                const state = Math.round(voxelData[idx * VOXEL_STRIDE]);
                
                if (!isTransparent(state)) {
                    return true; // Occluded!
                }
            }

            // Check if we've reached the target voxel
            if (currentX === endV.vx && currentY === endV.vy && currentZ === endV.vz) {
                reached = true;
                break;
            }

            // Advance to the next voxel
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    currentX += stepX;
                    if (tMaxX > 1) reached = true;
                    tMaxX += tDeltaX;
                } else {
                    currentZ += stepZ;
                    if (tMaxZ > 1) reached = true;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    currentY += stepY;
                    if (tMaxY > 1) reached = true;
                    tMaxY += tDeltaY;
                } else {
                    currentZ += stepZ;
                    if (tMaxZ > 1) reached = true;
                    tMaxZ += tDeltaZ;
                }
            }
        }

        return false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — computeMGM(E_mel)
    //
    //  Step 2: Map E_mel → Melanopic Glare Multiplier via piecewise
    //  Stevens' Power Law.
    //
    //  Comfort zone (0 ≤ E_mel ≤ E_opt):
    //    MGM = 1.0 - 0.15 * (E_mel / E_opt)           → [1.0 … 0.85]
    //    Interpretation: optimal daylight lowers cognitive stress slightly.
    //
    //  Glare zone (E_mel > E_opt):
    //    t   = (clamp(E_mel, E_opt, E_max) - E_opt) / (E_max - E_opt)   ∈ [0,1]
    //    MGM = 0.85 + (MGM_max - 0.85) * t^gamma                        → [0.85, MGM_max]
    //    Interpretation: retinal overload exponentially amplifies ACI stress.
    //
    //  Special case E_mel = 0 (total darkness / no sources): returns 1.0.
    //  The darkness-clamp on D_vis (step 3) is the primary stress mechanism
    //  for the zero-light case.
    //
    //  @param {number} E_mel   Output of computeRetinalLoad().
    //  @returns {number}       MGM multiplier ∈ [0.85, MGM_max] (or 1.0 for zero E_mel).
    // ═══════════════════════════════════════════════════════════════════════
    computeMGM(E_mel) {
        const { E_opt, E_max, MGM_max, gamma } = this.config;

        // No light at all — neutral multiplier; D_vis clamp handles darkness.
        if (E_mel <= 0) return 1.0;

        if (E_mel <= E_opt) {
            // ── Comfort zone ────────────────────────────────────────────────
            // Linear attenuation: E_mel = E_opt → MGM = 0.85.
            return 1.0 - 0.15 * (E_mel / E_opt);
        } else {
            // ── Glare zone ──────────────────────────────────────────────────
            // Clamp at E_max so the multiplier never exceeds MGM_max.
            const E_clamped = Math.min(E_mel, E_max);

            // Normalised position within the glare ramp: 0 at E_opt, 1 at E_max.
            const t = (E_clamped - E_opt) / (E_max - E_opt);

            // Stevens' Power Law with gamma exponent (γ > 1 = accelerating onset).
            return 0.85 + (MGM_max - 0.85) * Math.pow(t, gamma);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — applyToACI(aciBase, E_mel, E_ambient, [D_vis])
    //
    //  Step 3 + darkness-clamp: produce the final biologically-scaled ACI.
    //
    //  DARKNESS CLAMP (pre-ACI):
    //    When E_ambient < darknessThreshold:
    //      D_vis_clamped = min(D_vis, darknessMaxDvis)
    //      aciBase_adjusted = aciBase * (1 + (1 − D_vis_clamped/D_vis) * 0.5)
    //    This approximates the increased geometric stress that results from
    //    a truncated visual prospect in near-darkness, without re-running the
    //    GPU raycaster (Commandment #2 — never block the 60 Hz loop).
    //    The 0.5 factor is conservative to avoid double-penalising when the
    //    GPU raycaster already captured a low-light isovist.
    //
    //  MGM APPLICATION:
    //    ACI_final = pow(ACI_adjusted, 1.0 / MGM)
    //      MGM < 1  → exponent > 1  → ACI_final < ACI_adjusted  (comfort relief)
    //      MGM = 1  → exponent = 1  → ACI_final = ACI_adjusted   (neutral)
    //      MGM > 1  → exponent < 1  → ACI_final > ACI_adjusted   (glare stress)
    //
    //  @param {number}           aciBase    Base ACI from GPU raycaster (≥ 0).
    //  @param {number}           E_mel      Output of computeRetinalLoad().
    //  @param {number}           E_ambient  Ambient illuminance.
    //  @param {number|undefined} D_vis      Prospect depth [m] from aci.prospect.
    //                                       Pass undefined to skip darkness clamp.
    //
    //  @returns {{
    //    aciBase          : number,   — original input (unchanged)
    //    E_mel            : number,   — echo of input
    //    E_ambient        : number,
    //    mgm              : number,   — computed MGM multiplier
    //    D_vis_clamped    : number|undefined,
    //    aciBase_adjusted : number,   — aciBase after darkness penalty
    //    aciFinal         : number,   — final biological-scaled ACI
    //    inDarkness       : boolean,
    //  }}
    // ═══════════════════════════════════════════════════════════════════════
    applyToACI(aciBase, E_mel, E_ambient, D_vis) {
        const { darknessThreshold, darknessMaxDvis } = this.config;

        // ── Step A: Darkness clamp ───────────────────────────────────────────
        // Use ambient illuminance (E_ambient) instead of directional E_mel.
        const inDarkness = E_ambient < darknessThreshold;
        let D_vis_clamped = D_vis;
        let aciBase_adjusted = aciBase;

        if (inDarkness && D_vis !== undefined && D_vis !== null) {
            // Clamp visual prospect depth to arm's-length in near-darkness.
            D_vis_clamped = Math.min(D_vis, darknessMaxDvis);

            // Apply proportional darkness penalty to the base ACI if D_vis
            // was actually reduced (i.e. the agent was seeing further than
            // the darkness cap allows).
            if (D_vis > darknessMaxDvis) {
                const dvisFraction = D_vis_clamped / Math.max(0.01, D_vis);
                // penalty ∈ (1.0, 1.5] — capped at +50% stress.
                const darkPenalty = 1.0 + (1.0 - dvisFraction) * 0.5;
                aciBase_adjusted = aciBase * darkPenalty;
            }
        }

        // ── Step B: Compute MGM from (possibly darkness-adjusted) E_mel ─────
        const mgm = this.computeMGM(E_mel);

        // ── Step C: Apply MGM — ACI_final = pow(ACI_adjusted, 1 / MGM) ──────
        let aciFinal;
        if (aciBase_adjusted <= 0) {
            // Zero prospect → zero scaled score (nothing to amplify or reduce).
            aciFinal = 0;
        } else {
            // Guard MGM against a near-zero denominator (theoretical edge case
            // if config is misconfigured with MGM_max << 1).
            aciFinal = Math.pow(aciBase_adjusted, 1.0 / Math.max(0.01, mgm));
        }

        return {
            aciBase,
            E_mel,
            E_ambient,
            mgm,
            D_vis_clamped,
            aciBase_adjusted,
            aciFinal,
            inDarkness,
        };
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  PUBLIC — computeForAgent(agentPos, gazeDir, aciBase, D_vis, [sources], [grid])
    //
    //  Convenience one-shot wrapper: runs all three steps (E_mel → MGM →
    //  ACI_final) for a single agent in one call.
    //
    //  This is the primary method called from:
    //    a) BehaviorSimulator._maybeRunCognitionTick() (live 5 Hz tick)
    //    b) HeadlessACIPipeline.run()                  (batch headless mode)
    //
    //  @param {{ x, y, z }}      agentPos  World-space eye position.
    //  @param {{ x, y, z }}      gazeDir   Forward gaze vector.
    //  @param {number}           aciBase   ACI from GPU raycaster (aci.prospect).
    //  @param {number|undefined} D_vis     Prospect depth [m] (aci.prospect value).
    //  @param {Array|undefined}  sources   Optional source-list override.
    //  @param {Object|undefined} grid      Optional { voxelData, gridConfig }.
    //
    //  @returns {Object}  Full result from applyToACI().
    // ═══════════════════════════════════════════════════════════════════════
    computeForAgent(agentPos, gazeDir, aciBase, D_vis, sources, grid) {
        const { E_ambient, E_mel } = this.computeRetinalLoad(agentPos, gazeDir, sources, grid);
        return this.applyToACI(aciBase, E_mel, E_ambient, D_vis);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  FACTORY HELPERS
//  Construct typed light-source objects with standard melanopic weights.
//  Use these when populating the registry from window/lamp geometry detected
//  by RoomDetector or SpectralTracer.
//
//  Example:
//    const mgm = new MelanopicGlareModule();
//    mgm.setLightSources([
//        MGM_makeWindow('win_south',  2,  1.5, -8, 6000),
//        MGM_makeLamp  ('lamp_desk',  1,  0.8,  0,  300),
//        MGM_makeSkylight('skylight', 0, 3.5,   2, 4000),
//    ]);
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Create a daylight-weighted window light source.
 * @param {string} id        Unique identifier.
 * @param {number} x         World X centroid [m].
 * @param {number} y         World Y centroid [m].
 * @param {number} z         World Z centroid [m].
 * @param {number} I_base    Illuminance at source face [Lux].
 * @param {number} [W_mel]   Melanopic weight override (default: 1.5).
 * @returns {{ id, x, y, z, I_base, W_mel, type }}
 */
function MGM_makeWindow(id, x, y, z, I_base, W_mel) {
    return {
        id,
        x, y, z,
        I_base,
        W_mel: W_mel !== undefined ? W_mel : MGM_W_MEL_DAYLIGHT,
        type: 'window',
    };
}

/**
 * Create a warm-lamp light source.
 * @param {string} id        Unique identifier.
 * @param {number} x         World X centroid [m].
 * @param {number} y         World Y centroid [m].
 * @param {number} z         World Z centroid [m].
 * @param {number} I_base    Illuminance at source face [Lux].
 * @param {number} [W_mel]   Melanopic weight override (default: 0.4).
 * @returns {{ id, x, y, z, I_base, W_mel, type }}
 */
function MGM_makeLamp(id, x, y, z, I_base, W_mel) {
    return {
        id,
        x, y, z,
        I_base,
        W_mel: W_mel !== undefined ? W_mel : MGM_W_MEL_WARM_LAMP,
        type: 'lamp',
    };
}

/**
 * Create a skylight source (high melanopic, overhead).
 * @param {string} id        Unique identifier.
 * @param {number} x         World X centroid [m].
 * @param {number} y         World Y centroid [m] — typically ceiling height.
 * @param {number} z         World Z centroid [m].
 * @param {number} I_base    Illuminance at source face [Lux].
 * @param {number} [W_mel]   Melanopic weight override (default: 1.5).
 * @returns {{ id, x, y, z, I_base, W_mel, type }}
 */
function MGM_makeSkylight(id, x, y, z, I_base, W_mel) {
    return {
        id,
        x, y, z,
        I_base,
        W_mel: W_mel !== undefined ? W_mel : MGM_W_MEL_DAYLIGHT,
        type: 'skylight',
    };
}

// ─────────────────────────────────────────────────────────────────────────────
//  GLOBAL EXPOSURE  (mirrors the pattern of every other AHI module)
// ─────────────────────────────────────────────────────────────────────────────
if (typeof window !== 'undefined') {
    window.MelanopicGlareModule = MelanopicGlareModule;
    window.MGM_makeWindow       = MGM_makeWindow;
    window.MGM_makeLamp         = MGM_makeLamp;
    window.MGM_makeSkylight     = MGM_makeSkylight;
    window.MGM_W_MEL_DAYLIGHT   = MGM_W_MEL_DAYLIGHT;
    window.MGM_W_MEL_WARM_LAMP  = MGM_W_MEL_WARM_LAMP;
    window.MGM_W_MEL_LED_4000   = MGM_W_MEL_LED_4000;
}
