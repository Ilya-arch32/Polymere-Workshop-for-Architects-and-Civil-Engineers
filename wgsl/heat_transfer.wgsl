/**
 * AHI 2.0 Ultimate - Conjugate Heat Transfer (CHT) Solver
 * 
 * Решает уравнение теплопроводности в твердых телах и конвективный теплообмен в жидкостях.
 * Связывает температуру стен с температурой воздуха через граничные условия.
 */

struct Uniforms {
    grid_size: vec3<u32>,       // Offset 0-11 (12 bytes)
    _pad1: u32,                 // Offset 12-15 (4 bytes, aligns next to 16)
    resolution: f32,            // Offset 16
    dt: f32,                    // Offset 20
    alpha_solid: f32,           // Offset 24
    alpha_fluid: f32,           // Offset 28
    h_conv: f32,                // Offset 32
    // Padding 36-47 using array (no 16-byte alignment requirement!)
    _pad2_a: f32,               // Offset 36
    _pad2_b: f32,               // Offset 40
    _pad2_c: f32,               // Offset 44
    // gravity needs 16-byte alignment, starts at offset 48
    gravity: vec3<f32>,         // Offset 48-59 (12 bytes)
    _pad3: f32,                 // Offset 60-63 (4 bytes, aligns to 64)
    beta: f32,                  // Offset 64
    T_ref: f32,                 // Offset 68 - MATCHES JS!
    D_v: f32,                   // Offset 72
    moldRiskThreshold: f32,     // Offset 76
    moldRiskSteps: u32,         // Offset 80
    // Solar data
    solar_irradiance: f32,      // Offset 84
    _pad_solar_1: f32,          // Offset 88
    _pad_solar_2: f32,          // Offset 92
    // solar_direction needs 16-byte alignment, starts at offset 96
    solar_direction: vec3<f32>, // Offset 96-107 (12 bytes + 4 implicit = 112 total struct)
}

// Voxel states (matching VoxelTypes.js)
const VOXEL_EMPTY: u32 = 0u;
const VOXEL_SOLID: u32 = 1u;
const VOXEL_FLUID: u32 = 2u;
const VOXEL_GLASS: u32 = 4u;
const VOXEL_BOUNDARY: u32 = 8u;
// CHT Material-aware states
const VOXEL_WINDOW: u32 = 16u;
const VOXEL_DOOR: u32 = 32u;
const VOXEL_HEAT_SOURCE: u32 = 64u;
const VOXEL_EXTERNAL_WALL: u32 = 128u;
const VOXEL_MOLD_RISK: u32 = 256u; // 0x100 - Флаг риска плесени (ISO 13788)

// Material properties lookup (ASHRAE Handbook — Fundamentals)
// 16 standard building materials matched to MaterialID in VoxelTypes.js
fn get_thermal_conductivity(material_id: u32) -> f32 {
    switch material_id {
        case 0u:  { return 0.026; }   // Air
        case 1u:  { return 1.4; }     // Concrete (Heavy)
        case 2u:  { return 0.15; }    // Wood / Timber
        case 3u:  { return 1.0; }     // Glass (Glazing)
        case 4u:  { return 0.04; }    // Insulation (Generic)
        case 5u:  { return 50.0; }    // Steel
        case 6u:  { return 0.72; }    // Brick (Masonry)
        case 7u:  { return 0.16; }    // Gypsum Board
        case 8u:  { return 0.53; }    // Lightweight Concrete
        case 9u:  { return 0.16; }    // Plasterboard
        case 10u: { return 0.04; }    // Fiberglass
        case 11u: { return 0.038; }   // Mineral Wool
        case 12u: { return 1.3; }     // Ceramic Tile
        case 13u: { return 205.0; }   // Aluminum
        case 14u: { return 2.9; }     // Natural Stone
        case 15u: { return 0.034; }   // XPS / EPS Foam
        default: { return 1.0; }
    }
}

fn get_specific_heat(material_id: u32) -> f32 {
    switch material_id {
        case 0u:  { return 1005.0; }  // Air
        case 1u:  { return 880.0; }   // Concrete (Heavy)
        case 2u:  { return 1700.0; }  // Wood / Timber
        case 3u:  { return 840.0; }   // Glass (Glazing)
        case 4u:  { return 840.0; }   // Insulation (Generic)
        case 5u:  { return 500.0; }   // Steel
        case 6u:  { return 900.0; }   // Brick (Masonry)
        case 7u:  { return 1090.0; }  // Gypsum Board
        case 8u:  { return 1000.0; }  // Lightweight Concrete
        case 9u:  { return 840.0; }   // Plasterboard
        case 10u: { return 840.0; }   // Fiberglass
        case 11u: { return 840.0; }   // Mineral Wool
        case 12u: { return 800.0; }   // Ceramic Tile
        case 13u: { return 900.0; }   // Aluminum
        case 14u: { return 900.0; }   // Natural Stone
        case 15u: { return 1400.0; }  // XPS / EPS Foam
        default: { return 1000.0; }
    }
}

fn get_density(material_id: u32) -> f32 {
    switch material_id {
        case 0u:  { return 1.225; }   // Air
        case 1u:  { return 2400.0; }  // Concrete (Heavy)
        case 2u:  { return 600.0; }   // Wood / Timber
        case 3u:  { return 2500.0; }  // Glass (Glazing)
        case 4u:  { return 40.0; }    // Insulation (Generic)
        case 5u:  { return 7850.0; }  // Steel
        case 6u:  { return 1800.0; }  // Brick (Masonry)
        case 7u:  { return 800.0; }   // Gypsum Board
        case 8u:  { return 1400.0; }  // Lightweight Concrete
        case 9u:  { return 950.0; }   // Plasterboard
        case 10u: { return 12.0; }    // Fiberglass
        case 11u: { return 100.0; }   // Mineral Wool
        case 12u: { return 2000.0; }  // Ceramic Tile
        case 13u: { return 2700.0; }  // Aluminum
        case 14u: { return 2600.0; }  // Natural Stone
        case 15u: { return 35.0; }    // XPS / EPS Foam
        default: { return 1000.0; }
    }
}

// Паропроницаемость материалов (ISO 13788)
// μ - коэффициент сопротивления паропроницанию
fn get_vapor_permeability(material_id: u32) -> f32 {
    switch material_id {
        case 0u: { return 1.0; }     // Air (μ=1)
        case 1u: { return 100.0; }   // Concrete (μ~50-200)
        case 2u: { return 20.0; }    // Wood (μ~5-50)
        case 3u: { return 1000000.0; } // Glass (практически непроницаем)
        case 4u: { return 5.0; }     // Insulation
        case 5u: { return 15.0; }    // Brick
        default: { return 50.0; }
    }
}

// ============================================================================
// U-VALUE (Thermal Transmittance) for building materials W/(m²·K)
// Used for Newton's law of cooling at boundaries
// ============================================================================
fn get_u_value(material_id: u32) -> f32 {
    switch material_id {
        case 0u: { return 0.0; }    // Air - no resistance (interior)
        case 1u: { return 0.5; }    // Concrete wall with insulation ~0.5 W/(m²·K)
        case 2u: { return 1.5; }    // Wood door ~1.5 W/(m²·K)
        case 3u: { return 1.4; }    // Double glazed window ~1.4 W/(m²·K)
        case 4u: { return 0.2; }    // Insulation ~0.2 W/(m²·K)
        case 5u: { return 50.0; }   // Steel (radiator) - high
        case 6u: { return 0.8; }    // Brick ~0.8 W/(m²·K)
        case 99u: { return 0.0; }   // Heat source - N/A
        default: { return 1.0; }
    }
}

// Heat source temperature (Kelvin)
const HEAT_SOURCE_TEMP: f32 = 333.15; // 60°C

// SHGC: Solar Heat Gain Coefficient for double-pane clear glass (ASHRAE 140)
// Fraction of incident solar radiation that enters through the window as heat
const SHGC: f32 = 0.787;

// Floor absorptivity for solar radiation (dark concrete/wood)
const FLOOR_SOLAR_ABSORPTIVITY: f32 = 0.65;

// ============================================================================
// MAGNUS FORMULA - Давление насыщенного пара (ISO 13788)
// ============================================================================

// Формула Магнуса для расчета давления насыщенного водяного пара
// p_sat(T) = 610.94 * exp(17.625 * T_c / (T_c + 243.04)) [Pa]
// где T_c - температура в градусах Цельсия
fn saturated_vapor_pressure(T_kelvin: f32) -> f32 {
    let T_celsius = T_kelvin - 273.15;
    
    // Ограничиваем диапазон для стабильности
    let T_c = clamp(T_celsius, -40.0, 80.0);
    
    // Magnus formula constants (Buck, 1981)
    let a = 17.625;
    let b = 243.04;
    
    return 610.94 * exp(a * T_c / (T_c + b));
}

// Расчет точки росы (Dew Point) по формуле Магнуса
// T_dew = b * [α(T,RH) / (a - α(T,RH))]
// где α(T,RH) = a*T_c/(b+T_c) + ln(RH)
fn dew_point_temperature(T_kelvin: f32, RH: f32) -> f32 {
    let T_celsius = T_kelvin - 273.15;
    let T_c = clamp(T_celsius, -40.0, 80.0);
    let rh = clamp(RH, 0.01, 1.0); // Избегаем ln(0)
    
    let a = 17.625;
    let b = 243.04;
    
    let alpha = a * T_c / (b + T_c) + log(rh);
    let T_dew_c = b * alpha / (a - alpha);
    
    return T_dew_c + 273.15; // Возвращаем в Кельвинах
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> voxel_state: array<f32>; // [state, material, ...]
@group(0) @binding(2) var<storage, read> temperature_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> temperature_out: array<f32>;
@group(0) @binding(4) var<storage, read> velocity: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read_write> heat_flux: array<f32>;

// Humidity and mold risk buffers (group 1) - used by vapor_diffusion_step and calculate_mold_risk
@group(1) @binding(0) var<storage, read> humidity_in: array<f32>;
@group(1) @binding(1) var<storage, read_write> humidity_out: array<f32>;
@group(1) @binding(2) var<storage, read_write> mold_risk: array<u32>;
@group(1) @binding(3) var<storage, read_write> mold_risk_counter: array<u32>;

fn index_1d(x: u32, y: u32, z: u32) -> u32 {
    return x + y * uniforms.grid_size.x + z * uniforms.grid_size.x * uniforms.grid_size.y;
}

// ============================================================================
// Helper functions for material-aware heat transfer
// ============================================================================

// Get U-value at domain boundary by finding neighboring solid material
fn get_boundary_u_value(x: u32, y: u32, z: u32) -> f32 {
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 8u]);
            
            // If neighbor is solid, use its material's U-value
            if ((n_state & VOXEL_SOLID) != 0u) {
                let n_material = u32(voxel_state[n_idx * 8u + 1u]);
                
                // Windows have higher U-value (more heat loss)
                if ((n_state & VOXEL_WINDOW) != 0u) {
                    return 1.4; // Double glazing
                }
                // Doors
                if ((n_state & VOXEL_DOOR) != 0u) {
                    return 1.5;
                }
                // External wall
                if ((n_state & VOXEL_EXTERNAL_WALL) != 0u) {
                    return 0.3; // Well-insulated wall
                }
                return get_u_value(n_material);
            }
        }
    }
    
    // Default: open boundary with high heat transfer
    return 5.0; // High U-value for uninsulated openings
}

// Check if adjacent to heat source and return heat contribution
fn check_adjacent_heat_source(x: u32, y: u32, z: u32, T_current: f32) -> f32 {
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    var heat_gain = 0.0;
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 8u]);
            
            // Adjacent to heat source - receive heat via convection
            if ((n_state & VOXEL_HEAT_SOURCE) != 0u) {
                // Simple model: heat transfer proportional to temperature difference
                // h_conv = 10 W/(m²·K), A = dx², ρcp = 1225 J/(m³·K)
                let h_conv = uniforms.h_conv;
                let dT = h_conv * uniforms.dt * (HEAT_SOURCE_TEMP - T_current) / (1225.0 * uniforms.resolution);
                heat_gain += dT;
            }
        }
    }
    
    return heat_gain;
}

fn is_boundary(x: u32, y: u32, z: u32) -> bool {
    // Check if this voxel is at interface between solid and fluid
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 8u]);
    
    // FIXED: Use bitmask check so voxels with combined flags
    // (e.g. SOLID|WINDOW=17, SOLID|EXTERNAL_WALL=129) are included
    if ((state & VOXEL_SOLID) == 0u) && ((state & VOXEL_FLUID) == 0u) {
        return false;
    }
    
    // Check neighbors
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 8u]);
            
            // Boundary if solid-flagged next to fluid-flagged or vice versa
            if (((state & VOXEL_SOLID) != 0u && (n_state & VOXEL_FLUID) != 0u) ||
                ((state & VOXEL_FLUID) != 0u && (n_state & VOXEL_SOLID) != 0u)) {
                return true;
            }
        }
    }
    
    return false;
}

/**
 * Heat Diffusion in Solids (Fourier's Law)
 */
@compute @workgroup_size(4, 4, 4)
fn diffusion_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 8u]);
    let material = u32(voxel_state[idx * 8u + 1u]);
    
    // Only process solid-flagged voxels for diffusion (fluids handled by convection_step)
    // FIXED: Use bitmask so SOLID|WINDOW (17), SOLID|EXTERNAL_WALL (129) etc. are included
    if ((state & VOXEL_SOLID) == 0u) {
        temperature_out[idx] = temperature_in[idx];
        return;
    }
    
    let T_center = temperature_in[idx];
    let k = get_thermal_conductivity(material);
    let rho = get_density(material);
    let cp = get_specific_heat(material);
    let alpha = k / (rho * cp); // Thermal diffusivity
    
    // Finite difference for heat equation: ∂T/∂t = α∇²T
    // FIXED: Include FLUID neighbors for heat exchange at wall surfaces (convective BC)
    var laplacian = 0.0;
    var count = 0u;
    var convective_heat = 0.0;  // Heat from/to adjacent FLUID
    var conv_count = 0u;
    
    // 6-point stencil - check ALL neighbors
    // FIXED: Use bitmask checks so WINDOW(21), EXTERNAL_WALL(129), DOOR(33) etc. participate
    if (x > 0u) {
        let idx_m = index_1d(x - 1u, y, z);
        let n_state = u32(voxel_state[idx_m * 8u]);
        if ((n_state & VOXEL_SOLID) != 0u) {
            // SOLID-SOLID: pure conduction
            laplacian += temperature_in[idx_m] - T_center;
            count += 1u;
        } else if ((n_state & VOXEL_FLUID) != 0u) {
            // SOLID-FLUID: convective heat transfer (Newton's law)
            let T_fluid = temperature_in[idx_m];
            convective_heat += uniforms.h_conv * (T_fluid - T_center);
            conv_count += 1u;
        }
    }
    if (x < uniforms.grid_size.x - 1u) {
        let idx_p = index_1d(x + 1u, y, z);
        let n_state = u32(voxel_state[idx_p * 8u]);
        if ((n_state & VOXEL_SOLID) != 0u) {
            laplacian += temperature_in[idx_p] - T_center;
            count += 1u;
        } else if ((n_state & VOXEL_FLUID) != 0u) {
            let T_fluid = temperature_in[idx_p];
            convective_heat += uniforms.h_conv * (T_fluid - T_center);
            conv_count += 1u;
        }
    }
    
    if (y > 0u) {
        let idx_m = index_1d(x, y - 1u, z);
        let n_state = u32(voxel_state[idx_m * 8u]);
        if ((n_state & VOXEL_SOLID) != 0u) {
            laplacian += temperature_in[idx_m] - T_center;
            count += 1u;
        } else if ((n_state & VOXEL_FLUID) != 0u) {
            let T_fluid = temperature_in[idx_m];
            convective_heat += uniforms.h_conv * (T_fluid - T_center);
            conv_count += 1u;
        }
    }
    if (y < uniforms.grid_size.y - 1u) {
        let idx_p = index_1d(x, y + 1u, z);
        let n_state = u32(voxel_state[idx_p * 8u]);
        if ((n_state & VOXEL_SOLID) != 0u) {
            laplacian += temperature_in[idx_p] - T_center;
            count += 1u;
        } else if ((n_state & VOXEL_FLUID) != 0u) {
            let T_fluid = temperature_in[idx_p];
            convective_heat += uniforms.h_conv * (T_fluid - T_center);
            conv_count += 1u;
        }
    }
    
    if (z > 0u) {
        let idx_m = index_1d(x, y, z - 1u);
        let n_state = u32(voxel_state[idx_m * 8u]);
        if ((n_state & VOXEL_SOLID) != 0u) {
            laplacian += temperature_in[idx_m] - T_center;
            count += 1u;
        } else if ((n_state & VOXEL_FLUID) != 0u) {
            let T_fluid = temperature_in[idx_m];
            convective_heat += uniforms.h_conv * (T_fluid - T_center);
            conv_count += 1u;
        }
    }
    if (z < uniforms.grid_size.z - 1u) {
        let idx_p = index_1d(x, y, z + 1u);
        let n_state = u32(voxel_state[idx_p * 8u]);
        if ((n_state & VOXEL_SOLID) != 0u) {
            laplacian += temperature_in[idx_p] - T_center;
            count += 1u;
        } else if ((n_state & VOXEL_FLUID) != 0u) {
            let T_fluid = temperature_in[idx_p];
            convective_heat += uniforms.h_conv * (T_fluid - T_center);
            conv_count += 1u;
        }
    }
    
    // Compute temperature change
    var dT = 0.0;
    
    // Conduction term (diffusion through solid)
    if (count > 0u) {
        laplacian = laplacian / (uniforms.resolution * uniforms.resolution);
        dT += alpha * uniforms.dt * laplacian;
    }
    
    // Convection term (heat exchange with adjacent air)
    // dT = h * A * ΔT * dt / (ρ * V * cp)
    // Simplified: dT = h * ΔT * dt / (ρ * dx * cp)
    if (conv_count > 0u) {
        let conv_factor = uniforms.dt / (rho * uniforms.resolution * cp);
        dT += convective_heat * conv_factor;
    }
    
    temperature_out[idx] = T_center + dT;
}

/**
 * Convective Heat Transfer in Fluids
 * Now with material-aware boundaries and heat sources
 */
@compute @workgroup_size(4, 4, 4)
fn convection_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 8u]);
    let material = u32(voxel_state[idx * 8u + 1u]);
    
    // HEAT SOURCE: Maintain constant temperature (e.g., radiators at 60°C)
    if ((state & VOXEL_HEAT_SOURCE) != 0u) {
        temperature_out[idx] = HEAT_SOURCE_TEMP;
        return;
    }
    
    // Only process fluid voxels for convection
    if ((state & VOXEL_FLUID) == 0u) {
        return;
    }
    
    let T_center = temperature_in[idx];
    
    // EXTERNAL BOUNDARY: U-value based heat transfer (Newton's law of cooling)
    // Instead of setting T = T_ref directly, calculate heat flux through envelope
    let at_x_boundary = (x == 0u) || (x >= uniforms.grid_size.x - 1u);
    let at_y_boundary = (y == 0u) || (y >= uniforms.grid_size.y - 1u);
    let at_z_boundary = (z == 0u) || (z >= uniforms.grid_size.z - 1u);
    
    if (at_x_boundary || at_y_boundary || at_z_boundary) {
        // Domain boundary FLUID voxels: set directly to external temperature
        // This matches what diffusion_step does for all boundary voxels
        // Using T_ref directly ensures proper boundary condition
        temperature_out[idx] = uniforms.T_ref;
        return;
    }
    
    // Check if adjacent to heat source - receive heat from nearby radiators
    let heat_from_source = check_adjacent_heat_source(x, y, z, T_center);
    
    let v = velocity[idx];
    
    // Upwind advection scheme
    var dTdx = 0.0;
    var dTdy = 0.0;
    var dTdz = 0.0;
    
    // X direction
    if (v.x > 0.0 && x > 0u) {
        let idx_m = index_1d(x - 1u, y, z);
        dTdx = (T_center - temperature_in[idx_m]) / uniforms.resolution;
    } else if (v.x < 0.0 && x < uniforms.grid_size.x - 1u) {
        let idx_p = index_1d(x + 1u, y, z);
        dTdx = (temperature_in[idx_p] - T_center) / uniforms.resolution;
    }
    
    // Y direction
    if (v.y > 0.0 && y > 0u) {
        let idx_m = index_1d(x, y - 1u, z);
        dTdy = (T_center - temperature_in[idx_m]) / uniforms.resolution;
    } else if (v.y < 0.0 && y < uniforms.grid_size.y - 1u) {
        let idx_p = index_1d(x, y + 1u, z);
        dTdy = (temperature_in[idx_p] - T_center) / uniforms.resolution;
    }
    
    // Z direction
    if (v.z > 0.0 && z > 0u) {
        let idx_m = index_1d(x, y, z - 1u);
        dTdz = (T_center - temperature_in[idx_m]) / uniforms.resolution;
    } else if (v.z < 0.0 && z < uniforms.grid_size.z - 1u) {
        let idx_p = index_1d(x, y, z + 1u);
        dTdz = (temperature_in[idx_p] - T_center) / uniforms.resolution;
    }
    
    // Advection term: -v·∇T
    let advection = -(v.x * dTdx + v.y * dTdy + v.z * dTdz);
    
    // Diffusion (simplified) - ONLY consider FLUID neighbors
    // Heat exchange with SOLID neighbors is handled by boundary_coupling exclusively
    let alpha_air = uniforms.alpha_fluid;
    var laplacian = 0.0;
    var count = 0u;
    
    // Check all 6 neighbors for diffusion (FLUID-to-FLUID only)
    if (x > 0u) {
        let n_idx = index_1d(x - 1u, y, z);
        let n_state = u32(voxel_state[n_idx * 8u]);
        if ((n_state & VOXEL_FLUID) != 0u) {
            laplacian += temperature_in[n_idx] - T_center;
            count += 1u;
        }
    }
    if (x < uniforms.grid_size.x - 1u) {
        let n_idx = index_1d(x + 1u, y, z);
        let n_state = u32(voxel_state[n_idx * 8u]);
        if ((n_state & VOXEL_FLUID) != 0u) {
            laplacian += temperature_in[n_idx] - T_center;
            count += 1u;
        }
    }
    if (y > 0u) {
        let n_idx = index_1d(x, y - 1u, z);
        let n_state = u32(voxel_state[n_idx * 8u]);
        if ((n_state & VOXEL_FLUID) != 0u) {
            laplacian += temperature_in[n_idx] - T_center;
            count += 1u;
        }
    }
    if (y < uniforms.grid_size.y - 1u) {
        let n_idx = index_1d(x, y + 1u, z);
        let n_state = u32(voxel_state[n_idx * 8u]);
        if ((n_state & VOXEL_FLUID) != 0u) {
            laplacian += temperature_in[n_idx] - T_center;
            count += 1u;
        }
    }
    if (z > 0u) {
        let n_idx = index_1d(x, y, z - 1u);
        let n_state = u32(voxel_state[n_idx * 8u]);
        if ((n_state & VOXEL_FLUID) != 0u) {
            laplacian += temperature_in[n_idx] - T_center;
            count += 1u;
        }
    }
    if (z < uniforms.grid_size.z - 1u) {
        let n_idx = index_1d(x, y, z + 1u);
        let n_state = u32(voxel_state[n_idx * 8u]);
        if ((n_state & VOXEL_FLUID) != 0u) {
            laplacian += temperature_in[n_idx] - T_center;
            count += 1u;
        }
    }
    
    if (count > 0u) {
        laplacian = laplacian / (uniforms.resolution * uniforms.resolution);
    }
    
    // Update temperature: ∂T/∂t = -v·∇T + α∇²T + Q_radiator
    temperature_out[idx] = T_center + uniforms.dt * (advection + alpha_air * laplacian) + heat_from_source;
}

/**
 * Boundary Coupling: Heat exchange between solid and fluid
 */
@compute @workgroup_size(4, 4, 4)
fn boundary_coupling(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    // Only process boundary voxels
    if (!is_boundary(x, y, z)) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 8u]);
    
    // Check if this is an EXTERNAL WALL (receives solar radiation)
    let is_external = (state & VOXEL_EXTERNAL_WALL) != 0u;
    
    // Also check if at domain boundary (external envelope)
    let at_x_boundary = (x <= 1u) || (x >= uniforms.grid_size.x - 2u);
    let at_y_boundary = (y <= 1u) || (y >= uniforms.grid_size.y - 2u);
    let at_z_boundary = (z <= 1u) || (z >= uniforms.grid_size.z - 2u);
    let at_domain_edge = at_x_boundary || at_y_boundary || at_z_boundary;
    
    // External surface receives solar radiation
    let receives_solar = is_external || (at_domain_edge && (state & VOXEL_SOLID) != 0u);
    
    // Find adjacent solid/fluid pairs
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    var total_flux = 0.0;
    var flux_count = 0u;
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 8u]);
            
            // FIXED: Heat transfer at solid-fluid interface (bitmask check)
            let this_is_solid = (state & VOXEL_SOLID) != 0u;
            let this_is_fluid = (state & VOXEL_FLUID) != 0u;
            let neigh_is_solid = (n_state & VOXEL_SOLID) != 0u;
            let neigh_is_fluid = (n_state & VOXEL_FLUID) != 0u;
            
            if ((this_is_solid && neigh_is_fluid) ||
                (this_is_fluid && neigh_is_solid)) {
                
                let T_solid = select(temperature_in[idx], temperature_in[n_idx], this_is_fluid);
                let T_fluid = select(temperature_in[n_idx], temperature_in[idx], this_is_fluid);
                
                // Newton's law of cooling: q = h * A * (T_wall - T_fluid)
                // Adjust h based on local velocity (forced convection)
                let v_local = select(velocity[n_idx], velocity[idx], this_is_fluid);
                let v_mag = length(v_local);
                
                // Enhanced heat transfer with flow (simplified correlation)
                let h_effective = uniforms.h_conv * (1.0 + 0.5 * sqrt(v_mag));
                
                let area = uniforms.resolution * uniforms.resolution; // Face area
                var q = h_effective * area * (T_solid - T_fluid);

                // SOLAR RADIATION on EXTERNAL opaque surfaces
                // (Solar through WINDOWS is handled by solar_interior_absorption kernel)
                if (receives_solar && this_is_solid && neigh_is_fluid) {
                    // Skip window voxels - their solar goes THROUGH, not absorbed on surface
                    let is_window_voxel = (state & VOXEL_WINDOW) != 0u;
                    
                    if (!is_window_voxel) {
                        let normal = vec3<f32>(f32(neighbors[i].x), f32(neighbors[i].y), f32(neighbors[i].z));
                        
                        let is_outward_facing = (
                            (neighbors[i].x < 0 && x <= 2u) ||
                            (neighbors[i].x > 0 && x >= uniforms.grid_size.x - 3u) ||
                            (neighbors[i].y < 0 && y <= 2u) ||
                            (neighbors[i].y > 0 && y >= uniforms.grid_size.y - 3u) ||
                            (neighbors[i].z < 0 && z <= 2u) ||
                            (neighbors[i].z > 0 && z >= uniforms.grid_size.z - 3u) ||
                            is_external
                        );
                        
                        if (is_outward_facing) {
                            let sun_dir = normalize(uniforms.solar_direction);
                            let incidence = dot(normal, sun_dir);
                            
                            if (incidence > 0.0) {
                                // Opaque wall absorptivity = 0.6
                                let q_solar = uniforms.solar_irradiance * area * incidence * 0.6;
                                q -= q_solar;
                            }
                        }
                    }
                }
                
                total_flux += q;
                flux_count += 1u;
            }
        }
    }
    
    if (flux_count > 0u) {
        heat_flux[idx] = total_flux / f32(flux_count);
        
        // Apply flux to temperature
        let material = u32(voxel_state[idx * 8u + 1u]);
        let rho = get_density(material);
        let cp = get_specific_heat(material);
        let volume = uniforms.resolution * uniforms.resolution * uniforms.resolution;
        
        // ∆T = Q / (m * cp) = q * dt / (ρ * V * cp)
        let dT = heat_flux[idx] * uniforms.dt / (rho * volume * cp);
        
        if ((state & VOXEL_SOLID) != 0u) {
            temperature_out[idx] -= dT; // Solid loses heat
        } else {
            temperature_out[idx] += dT; // Fluid gains heat
        }
    }
}

/**
 * Compute Buoyancy Force for LBM
 * Returns force vector based on temperature difference
 */
@compute @workgroup_size(4, 4, 4)
fn compute_buoyancy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 8u]);
    
    // Only compute for fluid
    if (state != VOXEL_FLUID) {
        heat_flux[idx] = 0.0; // Reuse buffer for buoyancy magnitude
        return;
    }
    
    let T = temperature_out[idx];
    
    // Boussinesq approximation: F_buoyancy = -ρ₀ * g * β * (T - T_ref)
    // Direction is opposite to gravity
    let buoyancy_magnitude = uniforms.beta * abs(T - uniforms.T_ref);
    
    // Store magnitude for LBM to use
    heat_flux[idx] = buoyancy_magnitude;
}

// ============================================================================
// SOLAR INTERIOR ABSORPTION - Transmitted solar through windows (ASHRAE 140)
// ============================================================================
//
// Column-marching approach: For each interior SOLID voxel, trace backward
// along the sun direction. If the ray passes through a WINDOW voxel before
// exiting the domain, then this voxel receives transmitted solar energy:
//   q_absorbed = SHGC × Irradiance × cos(θ) × absorptivity × A_voxel_face
//   ΔT = q_absorbed × dt / (ρ × V × cp)
//
// This is a simplified model that avoids full GPU ray-tracing while giving
// physically correct total energy input for ASHRAE 140 compliance.
// ============================================================================

@compute @workgroup_size(4, 4, 4)
fn solar_interior_absorption(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    // Skip if no solar irradiance
    if (uniforms.solar_irradiance <= 0.0) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 8u]);
    
    // Only process interior SOLID voxels (potential solar absorption surfaces)
    // These are floor, interior walls, furniture etc. that can absorb transmitted solar
    if ((state & VOXEL_SOLID) == 0u) {
        return;
    }
    
    // Skip external boundary voxels (they get solar via boundary_coupling)
    let at_x_boundary = (x <= 1u) || (x >= uniforms.grid_size.x - 2u);
    let at_y_boundary = (y <= 1u) || (y >= uniforms.grid_size.y - 2u);
    let at_z_boundary = (z <= 1u) || (z >= uniforms.grid_size.z - 2u);
    if (at_x_boundary || at_y_boundary || at_z_boundary) {
        return;
    }
    
    // Skip window voxels themselves (they transmit, not absorb)
    if ((state & VOXEL_WINDOW) != 0u) {
        return;
    }
    
    // Check if this solid voxel has at least one adjacent FLUID neighbor
    // (only exposed surfaces can absorb solar radiation)
    var has_fluid_neighbor = false;
    var fluid_face_count = 0u;
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 8u]);
            
            if ((n_state & VOXEL_FLUID) != 0u) {
                has_fluid_neighbor = true;
                fluid_face_count += 1u;
            }
        }
    }
    
    if (!has_fluid_neighbor) {
        return;
    }
    
    // === COLUMN-MARCH: Trace backward along sun direction ===
    // March from this voxel toward the sun. If we hit a WINDOW voxel
    // before exiting the domain, solar energy is transmitted to us.
    
    let sun_dir = normalize(uniforms.solar_direction);
    
    // We march in the REVERSE sun direction (toward the sun source)
    // Step size = 1 voxel in the dominant axis direction
    let march_dir = sun_dir; // Toward the sun
    
    // Determine step size (march in voxel-space, stepping 1 unit at a time)
    let abs_dir = abs(march_dir);
    let max_component = max(abs_dir.x, max(abs_dir.y, abs_dir.z));
    
    // Skip if sun direction is essentially zero
    if (max_component < 0.001) {
        return;
    }
    
    // Scale step so largest component moves 1 voxel per step
    let step = march_dir / max_component;
    
    var found_window = false;
    var march_pos = vec3<f32>(f32(x), f32(y), f32(z));
    
    // March up to grid diagonal distance (max possible path through domain)
    let max_steps = uniforms.grid_size.x + uniforms.grid_size.y + uniforms.grid_size.z;
    
    for (var s = 1u; s < max_steps; s++) {
        march_pos += step;
        
        let mx = i32(round(march_pos.x));
        let my = i32(round(march_pos.y));
        let mz = i32(round(march_pos.z));
        
        // Check if we've exited the domain
        if (mx < 0 || mx >= i32(uniforms.grid_size.x) ||
            my < 0 || my >= i32(uniforms.grid_size.y) ||
            mz < 0 || mz >= i32(uniforms.grid_size.z)) {
            // Exited domain without hitting a window → no transmitted solar
            break;
        }
        
        let m_idx = index_1d(u32(mx), u32(my), u32(mz));
        let m_state = u32(voxel_state[m_idx * 8u]);
        
        // Found a WINDOW voxel → solar transmits through!
        if ((m_state & VOXEL_WINDOW) != 0u) {
            found_window = true;
            break;
        }
        
        // Hit an opaque SOLID voxel (non-window) → blocked, no solar reaches us
        if ((m_state & VOXEL_SOLID) != 0u) {
            break;
        }
        
        // FLUID voxels are transparent, keep marching
    }
    
    if (!found_window) {
        return;
    }
    
    // === DEPOSIT SOLAR ENERGY ===
    // q_solar = SHGC × Irradiance × cos(incidence) × absorptivity × area
    // For a surface, incidence angle depends on the surface normal vs sun direction.
    // For floor voxels (top face exposed), normal is (0, 0, 1) in Z-up convention.
    // Use the fluid neighbor directions to determine which face is exposed.
    
    // Calculate effective solar flux on this voxel
    // Use surface normal from the fluid-facing direction
    var max_incidence = 0.0;
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 8u]);
            
            if ((n_state & VOXEL_FLUID) != 0u) {
                // Surface normal points toward the fluid (inward)
                // FIXED: Solar travels FROM sun, face lit when normal points TOWARD sun
                // Same convention as boundary_coupling: incidence = dot(normal, sun_dir) > 0
                let face_normal = vec3<f32>(f32(neighbors[i].x), f32(neighbors[i].y), f32(neighbors[i].z));
                let incidence = dot(face_normal, sun_dir);
                max_incidence = max(max_incidence, incidence);
            }
        }
    }
    
    if (max_incidence <= 0.0) {
        return; // Sun doesn't illuminate any exposed face of this voxel
    }
    
    // Transmitted solar flux deposited into this voxel
    let area = uniforms.resolution * uniforms.resolution;
    let q_solar = SHGC * uniforms.solar_irradiance * max_incidence * FLOOR_SOLAR_ABSORPTIVITY * area;
    
    // Temperature rise: ΔT = q × dt / (ρ × V × cp)
    let material = u32(voxel_state[idx * 8u + 1u]);
    let rho = get_density(material);
    let cp = get_specific_heat(material);
    let volume = uniforms.resolution * uniforms.resolution * uniforms.resolution;
    
    let dT_solar = q_solar * uniforms.dt / (rho * volume * cp);
    
    // Add solar heat to the solid voxel temperature
    temperature_out[idx] = temperature_out[idx] + dT_solar;
}

// ============================================================================
// ISO 13788: ДИФФУЗИЯ ВОДЯНОГО ПАРА
// ============================================================================
// Note: humidity_in, humidity_out, mold_risk, mold_risk_counter 
// are declared at the top of the file with other @group(1) bindings

/**
 * Диффузия водяного пара (ISO 13788)
 * 
 * Уравнение диффузии пара аналогично диффузии тепла:
 * ∂φ/∂t = D_eff * ∇²φ
 * 
 * где D_eff = D_v / μ (эффективный коэффициент диффузии с учетом материала)
 * φ - относительная влажность (0.0 - 1.0)
 */
@compute @workgroup_size(4, 4, 4)
fn vapor_diffusion_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 8u]);
    let material = u32(voxel_state[idx * 8u + 1u]);
    
    // Пропускаем пустые воксели
    if (state == VOXEL_EMPTY) {
        humidity_out[idx] = humidity_in[idx];
        return;
    }
    
    let phi_center = humidity_in[idx];
    let mu = get_vapor_permeability(material);
    
    // Эффективный коэффициент диффузии
    let D_eff = uniforms.D_v / mu;
    
    // Лапласиан влажности (6-точечный стенсил)
    var laplacian = 0.0;
    var count = 0u;
    let h2 = uniforms.resolution * uniforms.resolution;
    
    // X direction
    if (x > 0u) {
        let idx_m = index_1d(x - 1u, y, z);
        let state_m = u32(voxel_state[idx_m * 8u]);
        if (state_m != VOXEL_EMPTY) {
            // Учитываем разную паропроницаемость на границе
            let mu_m = get_vapor_permeability(u32(voxel_state[idx_m * 8u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_m); // Гармоническое среднее
            laplacian += D_avg * (humidity_in[idx_m] - phi_center) / h2;
            count += 1u;
        }
    }
    if (x < uniforms.grid_size.x - 1u) {
        let idx_p = index_1d(x + 1u, y, z);
        let state_p = u32(voxel_state[idx_p * 8u]);
        if (state_p != VOXEL_EMPTY) {
            let mu_p = get_vapor_permeability(u32(voxel_state[idx_p * 8u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_p);
            laplacian += D_avg * (humidity_in[idx_p] - phi_center) / h2;
            count += 1u;
        }
    }
    
    // Y direction
    if (y > 0u) {
        let idx_m = index_1d(x, y - 1u, z);
        let state_m = u32(voxel_state[idx_m * 8u]);
        if (state_m != VOXEL_EMPTY) {
            let mu_m = get_vapor_permeability(u32(voxel_state[idx_m * 8u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_m);
            laplacian += D_avg * (humidity_in[idx_m] - phi_center) / h2;
            count += 1u;
        }
    }
    if (y < uniforms.grid_size.y - 1u) {
        let idx_p = index_1d(x, y + 1u, z);
        let state_p = u32(voxel_state[idx_p * 8u]);
        if (state_p != VOXEL_EMPTY) {
            let mu_p = get_vapor_permeability(u32(voxel_state[idx_p * 8u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_p);
            laplacian += D_avg * (humidity_in[idx_p] - phi_center) / h2;
            count += 1u;
        }
    }
    
    // Z direction
    if (z > 0u) {
        let idx_m = index_1d(x, y, z - 1u);
        let state_m = u32(voxel_state[idx_m * 8u]);
        if (state_m != VOXEL_EMPTY) {
            let mu_m = get_vapor_permeability(u32(voxel_state[idx_m * 8u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_m);
            laplacian += D_avg * (humidity_in[idx_m] - phi_center) / h2;
            count += 1u;
        }
    }
    if (z < uniforms.grid_size.z - 1u) {
        let idx_p = index_1d(x, y, z + 1u);
        let state_p = u32(voxel_state[idx_p * 8u]);
        if (state_p != VOXEL_EMPTY) {
            let mu_p = get_vapor_permeability(u32(voxel_state[idx_p * 8u + 1u]));
            let D_avg = 2.0 * uniforms.D_v / (mu + mu_p);
            laplacian += D_avg * (humidity_in[idx_p] - phi_center) / h2;
            count += 1u;
        }
    }
    
    // Обновляем влажность
    if (count > 0u) {
        let new_phi = phi_center + uniforms.dt * laplacian;
        // Ограничиваем в физических пределах [0, 1]
        humidity_out[idx] = clamp(new_phi, 0.0, 1.0);
    } else {
        humidity_out[idx] = phi_center;
    }
}

/**
 * Оценка риска плесени (ISO 13788)
 * 
 * Условия для MOLD_RISK:
 * 1. Температура поверхности < точки росы (T_surface < T_dew)
 * 2. ИЛИ относительная влажность > 80% в течение X шагов
 */
@compute @workgroup_size(4, 4, 4)
fn calculate_mold_risk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let z = gid.z;
    
    if (x >= uniforms.grid_size.x || y >= uniforms.grid_size.y || z >= uniforms.grid_size.z) {
        return;
    }
    
    let idx = index_1d(x, y, z);
    let state = u32(voxel_state[idx * 8u]);
    
    // Проверяем только граничные воксели (твердые рядом с воздухом)
    if (state != VOXEL_SOLID) {
        mold_risk[idx] = 0u;
        return;
    }
    
    // Проверяем, является ли это поверхностью
    var is_surface = false;
    var adjacent_fluid_idx = idx;
    
    let neighbors = array<vec3<i32>, 6>(
        vec3<i32>(-1, 0, 0), vec3<i32>(1, 0, 0),
        vec3<i32>(0, -1, 0), vec3<i32>(0, 1, 0),
        vec3<i32>(0, 0, -1), vec3<i32>(0, 0, 1)
    );
    
    for (var i = 0u; i < 6u; i++) {
        let nx = i32(x) + neighbors[i].x;
        let ny = i32(y) + neighbors[i].y;
        let nz = i32(z) + neighbors[i].z;
        
        if (nx >= 0 && nx < i32(uniforms.grid_size.x) &&
            ny >= 0 && ny < i32(uniforms.grid_size.y) &&
            nz >= 0 && nz < i32(uniforms.grid_size.z)) {
            
            let n_idx = index_1d(u32(nx), u32(ny), u32(nz));
            let n_state = u32(voxel_state[n_idx * 8u]);
            
            if (n_state == VOXEL_FLUID) {
                is_surface = true;
                adjacent_fluid_idx = n_idx;
                break;
            }
        }
    }
    
    if (!is_surface) {
        mold_risk[idx] = 0u;
        return;
    }
    
    // Получаем температуру поверхности и влажность воздуха
    let T_surface = temperature_out[idx];
    let T_air = temperature_out[adjacent_fluid_idx];
    let RH_air = humidity_out[adjacent_fluid_idx];
    
    // Расчет точки росы
    let T_dew = dew_point_temperature(T_air, RH_air);
    
    // Расчет относительной влажности на поверхности
    // RH_surface = p_v_air / p_sat(T_surface)
    // p_v_air = RH_air * p_sat(T_air)
    let p_sat_air = saturated_vapor_pressure(T_air);
    let p_v_air = RH_air * p_sat_air;
    let p_sat_surface = saturated_vapor_pressure(T_surface);
    let RH_surface = p_v_air / p_sat_surface;
    
    // Проверяем условия риска плесени
    var risk_condition = false;
    
    // Условие 1: Температура поверхности ниже точки росы (конденсация)
    if (T_surface < T_dew) {
        risk_condition = true;
    }
    
    // Условие 2: Относительная влажность на поверхности > порога
    if (RH_surface > uniforms.moldRiskThreshold) {
        risk_condition = true;
    }
    
    // Обновляем счетчик
    var counter = mold_risk_counter[idx];
    
    if (risk_condition) {
        counter += 1u;
    } else {
        // Медленно сбрасываем счетчик при улучшении условий
        if (counter > 0u) {
            counter -= 1u;
        }
    }
    
    mold_risk_counter[idx] = counter;
    
    // Помечаем флагом MOLD_RISK если счетчик превышает порог
    if (counter >= uniforms.moldRiskSteps) {
        mold_risk[idx] = VOXEL_MOLD_RISK;
    } else {
        mold_risk[idx] = 0u;
    }
}
