/**
 * GPU Path Tracing Shader for Daylight Analysis
 * 
 * Implements Monte Carlo path tracing with:
 * - BVH acceleration structure traversal
 * - CIE sky models
 * - Cosine-weighted hemisphere sampling
 * - Next Event Estimation (direct sun lighting)
 * - Progressive refinement
 */

// === STRUCTURES ===

struct BVHNode {
    min: vec3<f32>,
    max: vec3<f32>,
    left_right: vec2<f32>,  // left/primitiveStart, right/primitiveCount (as floats)
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

struct HitInfo {
    hit: bool,
    t: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    voxel_idx: u32,
}

struct Uniforms {
    grid_size: vec3<u32>,
    _pad1: u32,
    resolution: f32,
    sample_iteration: u32,
    sun_elevation: f32,
    sun_azimuth: f32,
    sun_direction: vec3<f32>,
    _pad2: f32,
    sun_intensity: f32,  // Watt/m² from EPW
    sky_type: u32,       // CIE sky type 1-15
    diffuse_fraction: f32,
    exterior_illuminance: f32,
}

// === BUFFERS ===

@group(0) @binding(0) var<storage, read> bvh_nodes: array<BVHNode>;
@group(0) @binding(1) var<storage, read> primitives: array<u32>;
@group(0) @binding(2) var<storage, read> voxel_state: array<u32>;
@group(0) @binding(3) var<storage, read_write> radiance_output: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// === CONSTANTS ===

const MAX_BOUNCES: u32 = 6u;
const PI: f32 = 3.14159265359;
const EPSILON: f32 = 0.001;

// === RANDOM NUMBER GENERATION ===

// PCG Hash (for seeding)
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Random float [0, 1) from seed
var<private> rng_state: u32;

fn init_rng(pixel_id: u32, sample: u32) {
    rng_state = pcg_hash(pixel_id ^ pcg_hash(sample));
}

fn random() -> f32 {
    rng_state = pcg_hash(rng_state);
    return f32(rng_state) / 4294967296.0;
}

// === RAY-AABB INTERSECTION ===

fn intersect_aabb(ray: Ray, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray.direction;
    let t1 = (aabb_min - ray.origin) * inv_dir;
    let t2 = (aabb_max - ray.origin) * inv_dir;
    
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    
    return vec2<f32>(t_near, t_far);
}

// === BVH TRAVERSAL ===

fn trace_bvh(ray: Ray) -> HitInfo {
    var hit_info: HitInfo;
    hit_info.hit = false;
    hit_info.t = 1e10;
    
    // Stack for BVH traversal (32 levels deep)
    var stack: array<i32, 32>;
    var stack_ptr = 0;
    stack[0] = 0; // Root node
    
    while (stack_ptr >= 0) {
        let node_idx = stack[stack_ptr];
        stack_ptr -= 1;
        
        let node = bvh_nodes[node_idx];
        let hit_bounds = intersect_aabb(ray, node.min, node.max);
        
        // Skip if ray misses or bound is farther than current closest hit
        if (hit_bounds.x > hit_bounds.y || hit_bounds.y < 0.0 || hit_bounds.x > hit_info.t) {
            continue;
        }
        
        // Check if leaf node (left < 0)
        let left_child = i32(node.left_right.x);
        
        if (left_child < 0) {
            // Leaf node - test primitives
            let prim_start = u32(-left_child - 1);
            let prim_count = u32(node.left_right.y);
            
            for (var i = 0u; i < prim_count; i++) {
                let voxel_idx = primitives[prim_start + i];
                
                // Calculate voxel AABB
                let grid_x = voxel_idx % uniforms.grid_size.x;
                let grid_y = (voxel_idx / uniforms.grid_size.x) % uniforms.grid_size.y;
                let grid_z = voxel_idx / (uniforms.grid_size.x * uniforms.grid_size.y);
                
                let voxel_min = vec3<f32>(f32(grid_x), f32(grid_y), f32(grid_z)) * uniforms.resolution;
                let voxel_max = voxel_min + vec3<f32>(uniforms.resolution);
                
                let hit = intersect_aabb(ray, voxel_min, voxel_max);
                
                if (hit.x < hit.y && hit.x >= 0.0 && hit.x < hit_info.t) {
                    hit_info.hit = true;
                    hit_info.t = hit.x;
                    hit_info.position = ray.origin + ray.direction * hit.x;
                    hit_info.voxel_idx = voxel_idx;
                    
                    // Calculate normal from hit position
                    let local_pos = hit_info.position - voxel_min;
                    let epsilon = 0.0001;
                    
                    if (abs(local_pos.x) < epsilon) {
                        hit_info.normal = vec3<f32>(-1.0, 0.0, 0.0);
                    } else if (abs(local_pos.x - uniforms.resolution) < epsilon) {
                        hit_info.normal = vec3<f32>(1.0, 0.0, 0.0);
                    } else if (abs(local_pos.y) < epsilon) {
                        hit_info.normal = vec3<f32>(0.0, -1.0, 0.0);
                    } else if (abs(local_pos.y - uniforms.resolution) < epsilon) {
                        hit_info.normal = vec3<f32>(0.0, 1.0, 0.0);
                    } else if (abs(local_pos.z) < epsilon) {
                        hit_info.normal = vec3<f32>(0.0, 0.0, -1.0);
                    } else {
                        hit_info.normal = vec3<f32>(0.0, 0.0, 1.0);
                    }
                }
            }
        } else {
            // Internal node - push children onto stack
            stack_ptr += 1;
            stack[stack_ptr] = left_child;
            stack_ptr += 1;
            stack[stack_ptr] = i32(node.left_right.y);
        }
    }
    
    return hit_info;
}

// === CIE SKY MODEL ===

fn sample_cie_sky(direction: vec3<f32>) -> vec3<f32> {
    // Simplified CIE clear sky model
    // Returns illuminance contribution from sky direction
    
    let elevation = asin(direction.y);
    let gamma = acos(dot(direction, uniforms.sun_direction));
    
    // Perez luminance distribution (simplified)
    let z = max(elevation, 0.01);
    let chi = (4.0 / 9.0 - uniforms.sun_elevation / PI) * (PI - 2.0 * uniforms.sun_elevation);
    
    let f = (1.0 + exp((-0.32 / max(sin(z), 0.01)))) * (1.0 + 2.0 * exp(-3.0 * gamma));
    
    // Scale by diffuse fraction
    let sky_radiance = uniforms.exterior_illuminance * uniforms.diffuse_fraction / PI;
    
    return vec3<f32>(sky_radiance * f);
}

// === COSINE-WEIGHTED HEMISPHERE SAMPLING ===

fn sample_hemisphere(normal: vec3<f32>) -> vec3<f32> {
    // Cosine-weighted sampling
    let r1 = random();
    let r2 = random();
    
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(r2);
    let sin_theta = sqrt(1.0 - r2);
    
    // Local coordinates
    let local_dir = vec3<f32>(
        cos(phi) * sin_theta,
        cos_theta,
        sin(phi) * sin_theta
    );
    
    // Build tangent space
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    
    // Transform to world space
    return normalize(tangent * local_dir.x + normal * local_dir.y + bitangent * local_dir.z);
}

// === PATH TRACING ===

@compute @workgroup_size(8, 8, 1)
fn path_trace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let voxel_x = global_id.x;
    let voxel_y = global_id.y;
    let voxel_z = global_id.z;
    
    if (voxel_x >= uniforms.grid_size.x || voxel_y >= uniforms.grid_size.y || voxel_z >= uniforms.grid_size.z) {
        return;
    }
    
    let voxel_idx = voxel_x + voxel_y * uniforms.grid_size.x + voxel_z * uniforms.grid_size.x * uniforms.grid_size.y;
    
    // Initialize RNG
    init_rng(voxel_idx, uniforms.sample_iteration);
    
    // Start ray from voxel center, shooting upward (for daylight from above)
    let voxel_center = vec3<f32>(
        (f32(voxel_x) + 0.5) * uniforms.resolution,
        (f32(voxel_y) + 0.5) * uniforms.resolution,
        (f32(voxel_z) + 0.5) * uniforms.resolution
    );
    
    // Shoot ray upward (can add random jitter)
    let ray_dir = vec3<f32>(0.0, 1.0, 0.0);
    var ray = Ray(voxel_center, ray_dir);
    
    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    
    // Path tracing loop
    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        let hit = trace_bvh(ray);
        
        if (!hit.hit) {
            // Hit sky - accumulate sky contribution
            radiance += throughput * sample_cie_sky(ray.direction);
            break;
        }
        
        // Material: assume Lambertian with 0.7 reflectance (can be replaced with material buffer)
        let albedo = vec3<f32>(0.7);
        
        // Next Event Estimation: direct sun lighting
        let shadow_ray = Ray(hit.position + hit.normal * EPSILON, uniforms.sun_direction);
        let sun_visible = !trace_bvh(shadow_ray).hit;
        
        if (sun_visible) {
            let n_dot_l = max(dot(hit.normal, uniforms.sun_direction), 0.0);
            radiance += throughput * albedo * (uniforms.sun_intensity / PI) * n_dot_l;
        }
        
        // Russian Roulette termination
        let survival_prob = max(max(throughput.r, throughput.g), throughput.b);
        if (survival_prob < random()) {
            break;
        }
        throughput /= survival_prob;
        
        // Sample next direction
        let next_dir = sample_hemisphere(hit.normal);
        ray = Ray(hit.position + hit.normal * EPSILON, next_dir);
        
        // Update throughput (Lambertian BRDF * cosine / PDF cancels to just albedo)
        throughput *= albedo;
    }
    
    // Progressive accumulation
    let prev_radiance = radiance_output[voxel_idx].rgb;
    let sample_count = f32(uniforms.sample_iteration);
    let new_radiance = (prev_radiance * sample_count + radiance) / (sample_count + 1.0);
    
    radiance_output[voxel_idx] = vec4<f32>(new_radiance, 1.0);
}
