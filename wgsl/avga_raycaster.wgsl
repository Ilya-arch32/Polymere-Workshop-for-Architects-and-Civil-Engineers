// ============================================================================
//  avga_raycaster.wgsl — Polymère Space Syntax GPU Compute Shaders
//
//  Entry points:
//    compute_headings  — Pass 0: 16-ray DDA heading generation per observer
//    compute_isovist   — Pass A: 128-ray Fibonacci 3D spherical isovist
//    compute_adjacency — Pass B: pairwise LOS visibility graph (3D DDA)
//
//  IMPORTANT — Binding design:
//    Each entry point is deployed in a separate WebGPU pipeline with its
//    own bind group layout.  To make all three entry points compilable from
//    a single WGSL module, every binding slot used by ANY entry point must
//    be declared here.  The JS binds all slots for every dispatch even if
//    a given entry point does not read/write that slot.
//
//    Slot map (group 0):
//      0 — uniforms         (uniform,            all passes)
//      1 — voxels           (read-only-storage,  all passes)
//      2 — observers        (read-only-storage,  all passes)
//      3 — headings         (storage r/w,        Pass 0 writes; A & B read)
//      4 — output_metrics   (storage r/w,        Pass A writes; 0 & B unused)
//      5 — adjacency        (storage r/w atomic, Pass B writes; 0 & A unused)
//
//  Voxel buffer layout (stride 8 × f32):
//    [0] state   : 0=EMPTY, 1=SOLID, 2=FLUID
//    [1..7]        reserved
//
//  Uniform buffer layout (48 bytes):
//    offset  0 : grid_size_x  (u32)
//    offset  4 : grid_size_y  (u32)
//    offset  8 : grid_size_z  (u32)
//    offset 12 : observer_count (u32)
//    offset 16 : resolution     (f32, metres/voxel)
//    offset 20 : max_ray_dist   (f32, voxel units)
//    offset 24 : ray_step       (f32, voxel units)
//    offset 28 : fov_horizontal (f32, degrees)
//    offset 32 : fov_vertical   (f32, degrees)
//    offset 36 : _pad3 (u32)
//    offset 40 : _pad4 (u32)
//    offset 44 : _pad5 (u32)
// ============================================================================

// ── Shared Uniform Block ────────────────────────────────────────────────────
struct Uniforms {
    grid_size_x:     u32,
    grid_size_y:     u32,
    grid_size_z:     u32,
    observer_count:  u32,
    resolution:      f32,
    max_ray_dist:    f32,
    ray_step:        f32,
    fov_horizontal:  f32,
    fov_vertical:    f32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
}

// ── Bindings (group 0, all six slots declared for shared-module compatibility) ─
@group(0) @binding(0) var<uniform>             uniforms       : Uniforms;
@group(0) @binding(1) var<storage, read>       voxels         : array<f32>;
@group(0) @binding(2) var<storage, read>       observers      : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> headings       : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> output_metrics : array<f32>;
@group(0) @binding(5) var<storage, read_write> adjacency      : array<atomic<u32>>;

// ── Shared Constants ─────────────────────────────────────────────────────────
const VOXEL_STRIDE : u32 = 8u;
const PI           : f32 = 3.14159265358979323846;
// Fibonacci lattice increment: 2π × (1 − 1/φ)  ≈ 2π / φ²
const GOLDEN_RATIO : f32 = 2.39996322972865332;

// ============================================================================
//  HELPER — Sample voxel state at a continuous 3-D position.
//  Returns true if the voxel is SOLID or out-of-bounds (ray left the scene).
// ============================================================================
fn isSolid(px: f32, py: f32, pz: f32) -> bool {
    let nx = i32(uniforms.grid_size_x);
    let ny = i32(uniforms.grid_size_y);
    let nz = i32(uniforms.grid_size_z);

    let ix = i32(floor(px + 0.5));
    let iy = i32(floor(py + 0.5));
    let iz = i32(floor(pz + 0.5));

    if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) {
        return true; // out-of-bounds → treat as solid boundary
    }

    let flat = u32(ix) + u32(iy) * u32(nx) + u32(iz) * u32(nx * ny);
    return u32(voxels[flat * VOXEL_STRIDE]) == 1u;
}

// ============================================================================
//  HELPER — Amanatides & Woo 3-D DDA traversal.
//  Returns TRUE  → ray reaches maxDistanceGrid without hitting SOLID.
//  Returns FALSE → SOLID voxel encountered before maxDistanceGrid.
// ============================================================================
fn traverseDDA(rayOrigin: vec3<f32>, rayDir: vec3<f32>, maxDistanceGrid: f32) -> bool {
    var cur = vec3<i32>(floor(rayOrigin + 0.5));

    var step_v = vec3<i32>(0);
    if (rayDir.x > 0.0) { step_v.x =  1; } else if (rayDir.x < 0.0) { step_v.x = -1; }
    if (rayDir.y > 0.0) { step_v.y =  1; } else if (rayDir.y < 0.0) { step_v.y = -1; }
    if (rayDir.z > 0.0) { step_v.z =  1; } else if (rayDir.z < 0.0) { step_v.z = -1; }

    let INF : f32 = 3.402823466e+38;
    var deltaDist = vec3<f32>(INF);
    if (abs(rayDir.x) > 1e-8) { deltaDist.x = abs(1.0 / rayDir.x); }
    if (abs(rayDir.y) > 1e-8) { deltaDist.y = abs(1.0 / rayDir.y); }
    if (abs(rayDir.z) > 1e-8) { deltaDist.z = abs(1.0 / rayDir.z); }

    var tMax = vec3<f32>(INF);
    if (rayDir.x > 0.0) { tMax.x = (f32(cur.x) + 0.5 - rayOrigin.x) * deltaDist.x; }
    else if (rayDir.x < 0.0) { tMax.x = (rayOrigin.x - (f32(cur.x) - 0.5)) * deltaDist.x; }
    if (rayDir.y > 0.0) { tMax.y = (f32(cur.y) + 0.5 - rayOrigin.y) * deltaDist.y; }
    else if (rayDir.y < 0.0) { tMax.y = (rayOrigin.y - (f32(cur.y) - 0.5)) * deltaDist.y; }
    if (rayDir.z > 0.0) { tMax.z = (f32(cur.z) + 0.5 - rayOrigin.z) * deltaDist.z; }
    else if (rayDir.z < 0.0) { tMax.z = (rayOrigin.z - (f32(cur.z) - 0.5)) * deltaDist.z; }

    var t : f32 = 0.0;
    let nx = i32(uniforms.grid_size_x);
    let ny = i32(uniforms.grid_size_y);
    let nz = i32(uniforms.grid_size_z);

    loop {
        if (t >= maxDistanceGrid) { return true; }
        if (cur.x < 0 || cur.x >= nx ||
            cur.y < 0 || cur.y >= ny ||
            cur.z < 0 || cur.z >= nz) { return true; }

        let flat = u32(cur.x) + u32(cur.y) * u32(nx) + u32(cur.z) * u32(nx * ny);
        if (u32(voxels[flat * VOXEL_STRIDE]) == 1u) { return false; }

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) { t = tMax.x; tMax.x += deltaDist.x; cur.x += step_v.x; }
            else                  { t = tMax.z; tMax.z += deltaDist.z; cur.z += step_v.z; }
        } else {
            if (tMax.y < tMax.z) { t = tMax.y; tMax.y += deltaDist.y; cur.y += step_v.y; }
            else                  { t = tMax.z; tMax.z += deltaDist.z; cur.z += step_v.z; }
        }
    }
    return true;
}

// ============================================================================
//  HELPER — Step-based raymarcher.
//  Advances in steps of `step_size` voxels along (dx,dy,dz).
//  Returns the hit distance (≤ max_dist).
// ============================================================================
fn marchRay(ox: f32, oy: f32, oz: f32,
             dx: f32, dy: f32, dz: f32,
             step_size: f32, max_dist: f32) -> f32 {
    var t : f32 = step_size;
    loop {
        if (t >= max_dist) { return max_dist; }
        if (isSolid(ox + dx * t, oy + dy * t, oz + dz * t)) { return t; }
        t += step_size;
    }
    return max_dist;
}

// ============================================================================
//  PASS 0 — compute_headings
//
//  1 thread per observer (workgroup 64×1×1).
//  Casts 16 radial rays (22.5° apart) in the XZ plane.
//  Writes the normalised direction of the longest unobstructed ray to
//  headings[obs_idx] as vec4(dx, 0.0, dz, 0.0).
//
//  GPU replacement for the CPU _findLongestLineOfSight() bottleneck.
// ============================================================================
@compute @workgroup_size(64, 1, 1)
fn compute_headings(@builtin(global_invocation_id) gid: vec3<u32>) {
    let obs_idx = gid.x;
    if (obs_idx >= uniforms.observer_count) { return; }

    let obs      = observers[obs_idx];
    let ox       = obs.x;
    let oy       = obs.y;
    let oz       = obs.z;
    let step_sz  = uniforms.ray_step;
    let max_dist = uniforms.max_ray_dist;

    const NUM_H_RAYS  : u32 = 16u;
    const DELTA_ANGLE : f32 = 0.392699081698724; // 2π / 16

    var best_dist : f32 = -1.0;
    var best_dx   : f32 =  1.0;
    var best_dz   : f32 =  0.0;

    for (var r : u32 = 0u; r < NUM_H_RAYS; r++) {
        let angle = f32(r) * DELTA_ANGLE;
        let dx    = cos(angle);
        let dz    = sin(angle);
        let hit   = marchRay(ox, oy, oz, dx, 0.0, dz, step_sz, max_dist);
        if (hit > best_dist) {
            best_dist = hit;
            best_dx   = dx;
            best_dz   = dz;
        }
    }

    let len = sqrt(best_dx * best_dx + best_dz * best_dz);
    let ndx = select(1.0, best_dx / len, len > 0.0001);
    let ndz = select(0.0, best_dz / len, len > 0.0001);

    headings[obs_idx] = vec4<f32>(ndx, 0.0, ndz, 0.0);
}

// ============================================================================
//  PASS A — compute_isovist (3D Spherical Isovist via Fibonacci Lattice)
//
//  1 thread per observer (workgroup 64×1×1).
//  Casts 128 rays distributed uniformly over the full unit sphere.
//
//  Metrics per observer (written to output_metrics[obs_idx * 4 + 0..3]):
//    [0] Volume       V = (4π / 3N) × Σ d_i³    (voxel³)
//    [1] Surface Area S = (4π / N)  × Σ d_i²    (voxel²)
//    [2] Compactness  C = clamp((36π × V²) / S³, 0, 1)  (sphere = 1.0)
//    [3] 0.0  (padding)
// ============================================================================
@compute @workgroup_size(64, 1, 1)
fn compute_isovist(@builtin(global_invocation_id) gid: vec3<u32>) {
    let obs_idx = gid.x;
    if (obs_idx >= uniforms.observer_count) { return; }

    let obs      = observers[obs_idx];
    let ox       = obs.x;
    let oy       = obs.y;
    let oz       = obs.z;
    let step_sz  = uniforms.ray_step;
    let max_dist = uniforms.max_ray_dist;

    const NUM_RAYS : u32 = 128u;

    // ── Cast 128 Fibonacci-sphere rays ────────────────────────────────────
    var ray_dist : array<f32, 128>;

    for (var r : u32 = 0u; r < NUM_RAYS; r++) {
        // Fibonacci sphere parameterisation:
        //   cosTheta = 1 − 2*(i+0.5)/N   (uniform cos distribution in [−1, 1])
        //   phi      = GOLDEN_RATIO * i   (golden-angle azimuthal step)
        let fi       = f32(r);
        let fN       = f32(NUM_RAYS);
        let cosTheta = clamp(1.0 - 2.0 * (fi + 0.5) / fN, -1.0, 1.0);
        let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
        let phi      = GOLDEN_RATIO * fi;

        let dx = sinTheta * cos(phi); // X
        let dy = cosTheta;            // Y (polar axis)
        let dz = sinTheta * sin(phi); // Z

        ray_dist[r] = marchRay(ox, oy, oz, dx, dy, dz, step_sz, max_dist);
    }

    // ── Accumulate sums ───────────────────────────────────────────────────
    // We need three moments of the ray-distance distribution:
    //   sum_d  = Σ d_i         (first moment — for compactness numerator)
    //   sum_d2 = Σ d_i²        (second moment — surface area)
    //   sum_d3 = Σ d_i³        (third moment — volume)
    //
    // V = (4π / 3N) × Σ d³    (spherical cone volume, voxel³)
    // S = (4π / N)  × Σ d²    (spherical cap area,   voxel²)
    let solid_angle = (4.0 * PI) / f32(NUM_RAYS);

    var sum_d  : f32 = 0.0;
    var sum_d2 : f32 = 0.0;
    var sum_d3 : f32 = 0.0;

    for (var r : u32 = 0u; r < NUM_RAYS; r++) {
        let d  = ray_dist[r];
        let d2 = d * d;
        sum_d  += d;
        sum_d2 += d2;
        sum_d3 += d2 * d;
    }

    let fN           = f32(NUM_RAYS);
    let volume       = (solid_angle / 3.0) * sum_d3;  // voxel³
    let surface_area = solid_angle * sum_d2;            // voxel²

    // ── 3D Compactness — Power-Mean Ratio (PMR) ───────────────────────────
    //
    // The classic isoperimetric formula  C = (36π·V²) / S³  is scale-invariant
    // but overflows for  **heterogeneous ray sets** (some rays blocked near,
    // others reaching max_dist).  The cubic term in V dominates over S,
    // systematically pushing C above 1.0 before the clamp — yielding 1.0 for
    // every scene regardless of actual shape.
    //
    // Fix: Power-Mean Ratio  C_PMR = mean(d)³ / mean(d³)
    //   • mean(d)  = μ₁ = sum_d  / N
    //   • mean(d³) = μ₃ = sum_d3 / N
    //   • C_PMR    = μ₁³ / μ₃
    //
    // Proof of bound [0, 1]:
    //   By the Power Mean Inequality, PM(1) ≤ PM(3) always:
    //     mean(d) ≤ (mean(d³))^(1/3)    →    mean(d)³ ≤ mean(d³)
    //   Therefore C_PMR =  mean(d)³ / mean(d³)  ∈ (0, 1].
    //
    // Interpretation:
    //   C_PMR = 1.0  →  all d_i equal (perfect sphere isovist)
    //   C_PMR ≈ 0.5-0.8  →  typical rectangular room
    //   C_PMR ≈ 0.05-0.25 →  long narrow corridor
    //
    var compactness : f32 = 0.0;
    if (sum_d3 > 0.001 && sum_d > 0.001) {
        let mean_d  = sum_d  / fN;
        let mean_d3 = sum_d3 / fN;
        // mean_d^3 ≤ mean_d3 by PMI — safe, no clamp overflow possible
        compactness = clamp((mean_d * mean_d * mean_d) / mean_d3, 0.0, 1.0);
    }

    // ── Write output: [volume, surface_area, compactness, 0] ─────────────
    let base = obs_idx * 4u;
    output_metrics[base + 0u] = volume;
    output_metrics[base + 1u] = surface_area;
    output_metrics[base + 2u] = compactness;
    output_metrics[base + 3u] = 0.0;
}


// ============================================================================
//  PASS B — compute_adjacency (3D Visibility Graph / Pairwise LOS)
//
//  2D dispatch (workgroup 16×16×1).  Each thread tests one (A, B) pair.
//  Applies horizontal FOV cone culling against headings[] (written by Pass 0).
//
//  Output: bit-packed atomic<u32> adjacency matrix.
//    Bit at position (x * N + y) is set iff observer A sees observer B.
// ============================================================================
@compute @workgroup_size(16, 16, 1)
fn compute_adjacency(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x     = gid.x;
    let y     = gid.y;
    let count = uniforms.observer_count;

    if (x >= count || y >= count) { return; }

    var visible : bool = false;

    if (x == y) {
        visible = true; // self always visible
    } else {
        let obsA = observers[x].xyz;
        let obsB = observers[y].xyz;
        let diff = obsB - obsA;
        let dist = length(diff);

        if (dist < 0.001) {
            visible = true; // coincident observers
        } else {
            let dir = diff / dist;

            // Horizontal FOV cone culling (dot-product test in XZ plane)
            let forward   = normalize(headings[x].xyz);
            let cos_angle = dot(forward, dir);
            let fov_deg   = max(uniforms.fov_horizontal, 120.0);
            let threshold = cos(radians(fov_deg) * 0.5);

            if (cos_angle >= threshold) {
                // Inside cone → full 3D line-of-sight test
                visible = traverseDDA(obsA, dir, dist);
            }
            // Outside cone → visible stays false
        }
    }

    if (visible) {
        let flat_idx = x * count + y;
        let u32_idx  = flat_idx / 32u;
        let bit_idx  = flat_idx % 32u;
        atomicOr(&adjacency[u32_idx], 1u << bit_idx);
    }
}
