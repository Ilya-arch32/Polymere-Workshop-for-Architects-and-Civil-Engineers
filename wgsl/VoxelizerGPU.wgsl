// AHI 2.0 Ultimate - GPU Batched Voxelization
// Processes triangles in batches to avoid GPU TDR timeout

struct Triangle {
    v0: vec3<f32>,
    v1: vec3<f32>, 
    v2: vec3<f32>,
    material_id: f32,  // Changed from u32 - JS writes Float32Array so this must be f32
}

struct VoxelGrid {
    dimensions: vec3<f32>,   // offset 0: nx, ny, nz
    _pad0: f32,              // offset 12: padding
    bounds_min: vec3<f32>,   // offset 16
    _pad1: f32,              // offset 28: padding  
    bounds_max: vec3<f32>,   // offset 32
    _pad2: f32,              // offset 44: padding
    resolution: f32,         // offset 48
    tri_start: u32,          // offset 52: first triangle to process (BATCHING)
    tri_end: u32,            // offset 56: last triangle (exclusive)
    _pad3: u32,              // offset 60: padding to 64 bytes
}

// Voxel struct - exactly 32 bytes (8 floats)
// State values: EMPTY=0, SOLID=1, FLUID=2
struct Voxel {
    state: f32,           // 0=EMPTY, 1=SOLID, 2=FLUID
    material: f32,        
    density: f32,         
    specific_heat: f32,   
    temperature: f32,     
    vx: f32,              
    vy: f32,              
    vz: f32,              
}

@group(0) @binding(0) var<uniform> grid: VoxelGrid;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read_write> voxels: array<Voxel>;
@group(0) @binding(3) var<storage, read_write> voxel_solid: array<atomic<u32>>;


// AABB-AABB intersection with 10% expansion for thin walls
fn aabb_triangle_intersect(box_center: vec3<f32>, half_size: vec3<f32>, tri: Triangle) -> bool {
    let tri_min = min(min(tri.v0, tri.v1), tri.v2);
    let tri_max = max(max(tri.v0, tri.v1), tri.v2);
    
    let expand = half_size * 1.01;  // 1% expansion for thin geometry (reduced from 10%)
    let box_min = box_center - expand;
    let box_max = box_center + expand;
    
    if (tri_max.x < box_min.x || tri_min.x > box_max.x) { return false; }
    if (tri_max.y < box_min.y || tri_min.y > box_max.y) { return false; }
    if (tri_max.z < box_min.z || tri_min.z > box_max.z) { return false; }
    
    return true;
}

@compute @workgroup_size(4, 4, 4)
fn voxelize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim_x = u32(grid.dimensions.x);
    let dim_y = u32(grid.dimensions.y);
    let dim_z = u32(grid.dimensions.z);
    
    if (gid.x >= dim_x || gid.y >= dim_y || gid.z >= dim_z) {
        return;
    }
    
    let voxel_idx = gid.x + gid.y * dim_x + gid.z * dim_x * dim_y;
    
    // Voxel center in world space
    let voxel_center = grid.bounds_min + vec3<f32>(gid) * grid.resolution + 
                       vec3<f32>(grid.resolution * 0.5);
    let half_size = vec3<f32>(grid.resolution * 0.5);
    
    // BATCHED: Only process triangles in current range [tri_start, tri_end)
    var intersected = false;
    var material_id = 0u;
    
    for (var i = grid.tri_start; i < grid.tri_end; i++) {
        if (aabb_triangle_intersect(voxel_center, half_size, triangles[i])) {
            intersected = true;
            material_id = u32(triangles[i].material_id);  // Cast f32 to u32
            break;
        }
    }
    
    // Only set state if we found intersection
    // (keep existing state if already marked by previous batch)
    if (intersected) {
        // FIXED: Set proper bitmask state for window/glass vs opaque solid
        // MaterialID.GLASS = 3 (from extractTriangles when category is WINDOW)
        // State bits: SOLID=1, GLASS=4, WINDOW=16 → combined = 21
        if (material_id == 3u) {
            voxels[voxel_idx].state = 21.0; // SOLID | GLASS | WINDOW (1+4+16)
        } else {
            voxels[voxel_idx].state = 1.0;  // SOLID only (opaque)
        }
        voxels[voxel_idx].material = f32(material_id);  // Store as f32 for voxel struct
        atomicOr(&voxel_solid[voxel_idx / 32u], 1u << (voxel_idx % 32u));
    } else if (grid.tri_start == 0u) {
        // Only initialize to FLUID on first batch
        voxels[voxel_idx].state = 2.0; // FLUID
        voxels[voxel_idx].material = 0.0;
    }
    
    // Initialize physics fields (only on first batch)
    if (grid.tri_start == 0u) {
        voxels[voxel_idx].density = 0.0;
        voxels[voxel_idx].specific_heat = 0.0;
        voxels[voxel_idx].temperature = 293.0;
        voxels[voxel_idx].vx = 0.0;
        voxels[voxel_idx].vy = 0.0;
        voxels[voxel_idx].vz = 0.0;
    }
}

// Flood fill (unchanged)
@compute @workgroup_size(256)
fn flood_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim_x = u32(grid.dimensions.x);
    let dim_y = u32(grid.dimensions.y);
    let dim_z = u32(grid.dimensions.z);
    let idx = gid.x;
    let total_voxels = dim_x * dim_y * dim_z;
    
    if (idx >= total_voxels) { return; }
    let dummy_state = voxels[idx].state; // Keep binding active
}

// Material assignment (unchanged)
@compute @workgroup_size(64)
fn assign_materials(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim_x = u32(grid.dimensions.x);
    let dim_y = u32(grid.dimensions.y);
    let dim_z = u32(grid.dimensions.z);
    let idx = gid.x;
    let total_voxels = dim_x * dim_y * dim_z;
    
    if (idx >= total_voxels) { return; }
    
    let material_id = u32(voxels[idx].material);
    var density = 1.225;
    var specific_heat = 1005.0;
    
    if (material_id == 1u) {      // Concrete
        density = 2400.0;
        specific_heat = 880.0;
    } else if (material_id == 2u) { // Wood
        density = 600.0;
        specific_heat = 1700.0;
    } else if (material_id == 3u) { // Glass
        density = 2500.0;
        specific_heat = 840.0;
    }
    
    voxels[idx].density = density;
    voxels[idx].specific_heat = specific_heat;
}
