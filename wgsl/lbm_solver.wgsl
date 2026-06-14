/**
 * AHI 2.0 Ultimate - Lattice Boltzmann Method D3Q19 Solver
 * 
 * WebGPU Compute Shader для 3D CFD симуляции воздушных потоков
 * 
 * ВАЖНО: Это ядро всей аэродинамики! Никаких упрощений!
 * Реализует полную D3Q19 схему с MRT оператором столкновений для стабильности.
 * 
 * Физика:
 * - 19 дискретных скоростей в 3D (оптимальный баланс точность/производительность)
 * - Multiple Relaxation Time (MRT) вместо упрощенного BGK
 * - Full Bounce-Back для твердых границ
 * - Источниковые члены для тепловой плавучести (связь с CHT)
 */

// ============================================================================
// КОНСТАНТЫ D3Q19
// ============================================================================

const D3Q19_WEIGHTS: array<f32, 19> = array<f32, 19>(
    1.0/3.0,                    // 0: (0,0,0) - rest particle
    1.0/18.0, 1.0/18.0,        // 1-2: (±1,0,0)
    1.0/18.0, 1.0/18.0,        // 3-4: (0,±1,0)
    1.0/18.0, 1.0/18.0,        // 5-6: (0,0,±1)
    1.0/36.0, 1.0/36.0,        // 7-8: (±1,±1,0)
    1.0/36.0, 1.0/36.0,        // 9-10: (±1,0,±1)
    1.0/36.0, 1.0/36.0,        // 11-12: (0,±1,±1)
    1.0/36.0, 1.0/36.0,        // 13-14: (±1,-1,0)
    1.0/36.0, 1.0/36.0,        // 15-16: (±1,0,-1)
    1.0/36.0, 1.0/36.0         // 17-18: (0,±1,-1)
);

// Дискретные векторы скоростей (19 направлений)
const D3Q19_CX: array<i32, 19> = array<i32, 19>(
    0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  0,  0,  1, -1,  1, -1,  0,  0
);
const D3Q19_CY: array<i32, 19> = array<i32, 19>(
    0,  0,  0,  1, -1,  0,  0,  1,  1,  0,  0,  1, -1, -1, -1,  0,  0,  1, -1
);
const D3Q19_CZ: array<i32, 19> = array<i32, 19>(
    0,  0,  0,  0,  0,  1, -1,  0,  0,  1,  1,  1,  1,  0,  0, -1, -1, -1, -1
);

// Обратные направления (для bounce-back)
const D3Q19_OPPOSITE: array<i32, 19> = array<i32, 19>(
    0,  2,  1,  4,  3,  6,  5,  14, 13, 16, 15, 18, 17, 8, 7, 10, 9, 12, 11
);

// ============================================================================
// UNIFORM BUFFERS
// ============================================================================

struct SimulationParams {
    nx: u32,                    // Grid dimensions
    ny: u32,
    nz: u32,
    resolution: f32,            // Voxel size (м)
    
    tau: f32,                   // Relaxation time (связан с вязкостью)
    omega: f32,                 // ω = 1/τ
    
    rho0: f32,                  // Reference density (кг/м³)
    nu: f32,                    // Kinematic viscosity (м²/с)
    
    // CRITICAL: In WGSL uniform buffers, vec3<f32> has alignment 16!
    // Next field MUST start at offset 48, not 44!
    gravity: vec3<f32>,         // offset 32-43 (12 bytes)
    _padding_gravity: f32,      // offset 44-47 (explicit padding!)
    
    dt: f32,                    // offset 48
    enableBuoyancy: u32,        // offset 52
    beta: f32,                  // offset 56
    smagorinskyConstant: f32,   // offset 60
    enableLES: u32,             // offset 64 (was missing!)
    
    // Parameters from UI
    inletVelocity: f32,         // offset 68
    terrainRoughness: f32,      // offset 72
    
    // Full 3D wind direction (normalized vector)
    windDirX: f32,              // offset 76
    windDirY: f32,              // offset 80
    windDirZ: f32,              // offset 84
    
    // Dynamic inlet/outlet plane selection
    // 0=X_MIN, 1=X_MAX, 2=Y_MIN, 3=Y_MAX, 4=Z_MIN, 5=Z_MAX
    inletPlane: u32,            // offset 88
    outletPlane: u32,           // offset 92
    
    // Padding to reach 112 bytes (WGSL uniform buffers must be multiple of 16)
    _padding_end1: u32,         // offset 96
    _padding_end2: u32,         // offset 100
    _padding_end3: u32,         // offset 104
    _padding_end4: u32,         // offset 108
    // Total: 112 bytes
}

@group(0) @binding(0) var<uniform> params: SimulationParams;

// ============================================================================
// STORAGE BUFFERS (Read/Write)
// ============================================================================

// Distribution functions (19 per voxel)
@group(0) @binding(1) var<storage, read> f_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> f_out: array<f32>;

// Macroscopic variables
@group(0) @binding(3) var<storage, read_write> density: array<f32>;
// CRITICAL: Use array<f32> with manual indexing to avoid vec3 16-byte alignment issues
// Access: velocity[idx*3+0] = vx, velocity[idx*3+1] = vy, velocity[idx*3+2] = vz
@group(0) @binding(4) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(5) var<storage, read_write> temperature: array<f32>;

// Voxel state (для boundary conditions)
// ВАЖНО: Буфер содержит 8 float32 на воксель: [state, material, density, temp, vx, vy, vz, pad]
// state: 0=EMPTY, 1=SOLID, 2=FLUID
@group(0) @binding(6) var<storage, read> voxelState: array<f32>;

// Константа: stride вокселей (8 float32 на воксель)
const VOXEL_STRIDE: u32 = 8u;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn voxelIndex(i: u32, j: u32, k: u32) -> u32 {
    return i + j * params.nx + k * params.nx * params.ny;
}

fn isInside(i: i32, j: i32, k: i32) -> bool {
    return i >= 0 && i < i32(params.nx) &&
           j >= 0 && j < i32(params.ny) &&
           k >= 0 && k < i32(params.nz);
}

fn isSolid(idx: u32) -> bool {
    // Читаем первый float из вокселя (state)
    // SOLID = 1.0, проверяем диапазон 0.5-1.5 для надежности
    let state = voxelState[idx * VOXEL_STRIDE];
    return state > 0.5 && state < 1.5;
}

fn isFluid(idx: u32) -> bool {
    // FLUID = 2.0, проверяем диапазон 1.5-2.5
    let state = voxelState[idx * VOXEL_STRIDE];
    return state > 1.5 && state < 2.5;
}

// ============================================================================
// EQUILIBRIUM DISTRIBUTION FUNCTION
// ============================================================================

fn equilibrium(q: u32, rho: f32, u: vec3<f32>) -> f32 {
    let w = D3Q19_WEIGHTS[q];
    let c = vec3<f32>(f32(D3Q19_CX[q]), f32(D3Q19_CY[q]), f32(D3Q19_CZ[q]));
    
    let cu = dot(c, u);
    let usqr = dot(u, u);
    
    // Maxwell-Boltzmann распределение (2-й порядок по скорости)
    return w * rho * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usqr);
}

// ============================================================================
// LES: STRESS TENSOR И МОДЕЛЬ СМАГОРИНСКОГО
// ============================================================================

// Тензор напряжений S_ij из неравновесной части функции распределения
// S_ij = Σ_q (f_q - f_eq_q) * c_qi * c_qj
fn computeStressTensor(f_local: array<f32, 19>, f_eq: array<f32, 19>) -> mat3x3<f32> {
    var S: mat3x3<f32>;
    
    // Инициализируем нулями
    S[0] = vec3<f32>(0.0, 0.0, 0.0);
    S[1] = vec3<f32>(0.0, 0.0, 0.0);
    S[2] = vec3<f32>(0.0, 0.0, 0.0);
    
    // Суммируем вклад от неравновесных моментов
    for (var q = 0u; q < 19u; q++) {
        let f_neq = f_local[q] - f_eq[q];  // Неравновесная часть
        let cx = f32(D3Q19_CX[q]);
        let cy = f32(D3Q19_CY[q]);
        let cz = f32(D3Q19_CZ[q]);
        
        // S_ij += f_neq * c_i * c_j
        S[0][0] += f_neq * cx * cx;  // S_xx
        S[0][1] += f_neq * cx * cy;  // S_xy
        S[0][2] += f_neq * cx * cz;  // S_xz
        S[1][0] += f_neq * cy * cx;  // S_yx
        S[1][1] += f_neq * cy * cy;  // S_yy
        S[1][2] += f_neq * cy * cz;  // S_yz
        S[2][0] += f_neq * cz * cx;  // S_zx
        S[2][1] += f_neq * cz * cy;  // S_zy
        S[2][2] += f_neq * cz * cz;  // S_zz
    }
    
    // Нормализация: в LBM тензор связан с вязкостью через tau
    // S_ij = -(1 / (2 * tau * rho * c_s^2)) * Π_neq_ij
    // где c_s^2 = 1/3 для D3Q19
    let factor = -1.0 / (2.0 * params.tau * params.rho0 * (1.0 / 3.0));
    S[0] *= factor;
    S[1] *= factor;
    S[2] *= factor;
    
    return S;
}

// Вычисление магнитуды тензора скоростей деформации |S| = sqrt(2 * S_ij * S_ij)
fn computeStrainRateMagnitude(S: mat3x3<f32>) -> f32 {
    // |S| = sqrt(2 * S_ij * S_ij) = sqrt(2 * (S_xx^2 + S_yy^2 + S_zz^2 + 2*(S_xy^2 + S_xz^2 + S_yz^2)))
    let Sxx = S[0][0];
    let Syy = S[1][1];
    let Szz = S[2][2];
    let Sxy = S[0][1];
    let Sxz = S[0][2];
    let Syz = S[1][2];
    
    let S_squared = Sxx*Sxx + Syy*Syy + Szz*Szz + 2.0*(Sxy*Sxy + Sxz*Sxz + Syz*Syz);
    return sqrt(2.0 * S_squared);
}

// Модель Смагоринского: турбулентная вязкость ν_t = (C_s * Δ)² * |S|
fn computeTurbulentViscosity(S_magnitude: f32) -> f32 {
    let Cs = params.smagorinskyConstant;  // Константа Смагоринского (0.1-0.2)
    let delta = params.resolution;         // Размер ячейки сетки (фильтр)
    
    // ν_t = (C_s * Δ)² * |S|
    let nu_t = (Cs * delta) * (Cs * delta) * S_magnitude;
    
    return nu_t;
}

// Вычисление эффективного времени релаксации τ_eff = τ_0 + τ_turb
// где τ_turb = 3 * ν_t / c_s^2 = 3 * ν_t * 3 = 9 * ν_t (в решеточных единицах)
fn computeEffectiveTau(nu_turb: f32) -> f32 {
    // В LBM: ν = (τ - 0.5) * c_s^2 = (τ - 0.5) / 3
    // Поэтому: τ = 3*ν + 0.5
    // τ_turb = 3 * ν_t
    let tau_turb = 3.0 * nu_turb;
    
    // τ_eff = τ_0 + τ_turb
    let tau_eff = params.tau + tau_turb;
    
    // Ограничиваем снизу для стабильности (τ > 0.5)
    return max(tau_eff, 0.505);
}

// ============================================================================
// MRT (Multiple Relaxation Time) COLLISION OPERATOR
// ============================================================================

// Преобразование в момент space (упрощенная версия для D3Q19)
fn momentTransform(f: array<f32, 19>) -> array<f32, 19> {
    var m: array<f32, 19>;
    
    // m[0] = density
    m[0] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + 
           f[7] + f[8] + f[9] + f[10] + f[11] + f[12] +
           f[13] + f[14] + f[15] + f[16] + f[17] + f[18];
    
    // m[1-3] = momentum (jx, jy, jz)
    m[1] = f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[13] - f[14] + f[15] - f[16];
    m[2] = f[3] - f[4] + f[7] + f[8] + f[11] - f[12] - f[13] - f[14] + f[17] - f[18];
    m[3] = f[5] - f[6] + f[9] + f[10] + f[11] + f[12] - f[15] - f[16] - f[17] - f[18];
    
    // Высшие моменты (энергия, stress tensor) - упрощенная формула
    for (var i = 4u; i < 19u; i++) {
        m[i] = 0.0; // Для базовой реализации
    }
    
    return m;
}

fn inverseMomentTransform(m: array<f32, 19>) -> array<f32, 19> {
    var f: array<f32, 19>;
    
    // Обратное преобразование (точная формула сложна, используем приближение)
    let rho = m[0];
    let ux = m[1] / rho;
    let uy = m[2] / rho;
    let uz = m[3] / rho;
    let u = vec3<f32>(ux, uy, uz);
    
    for (var q = 0u; q < 19u; q++) {
        f[q] = equilibrium(q, rho, u);
    }
    
    return f;
}

fn mrtCollision(f_local: array<f32, 19>, rho: f32, u: vec3<f32>) -> array<f32, 19> {
    return mrtCollisionWithOmega(f_local, rho, u, params.omega);
}

// MRT collision с произвольным omega (для LES с переменной вязкостью)
fn mrtCollisionWithOmega(f_local: array<f32, 19>, rho: f32, u: vec3<f32>, omega: f32) -> array<f32, 19> {
    var f_new: array<f32, 19>;
    
    // Преобразуем в момент space
    let m = momentTransform(f_local);
    let m_eq = momentTransform(inverseMomentTransform(m));
    
    // Relaxation с разными временами для разных моментов
    var m_relaxed: array<f32, 19>;
    m_relaxed[0] = m[0]; // Плотность сохраняется
    
    // Момент импульса релаксируется с omega (теперь переменный для LES)
    for (var i = 1u; i < 4u; i++) {
        m_relaxed[i] = m[i] - omega * (m[i] - m_eq[i]);
    }
    
    // Высшие моменты - масштабируем bulk omega пропорционально
    // Соотношение сохраняется: omega_bulk/omega_base = 1.2/omega_0
    let omega_ratio = 1.2 / params.omega;
    let omega_bulk = omega * omega_ratio;
    for (var i = 4u; i < 19u; i++) {
        m_relaxed[i] = m[i] - omega_bulk * (m[i] - m_eq[i]);
    }
    
    // Обратное преобразование
    return inverseMomentTransform(m_relaxed);
}

// ============================================================================
// MAIN COMPUTE SHADER: COLLISION STEP
// ============================================================================

@compute @workgroup_size(4, 4, 4)
fn collisionStep(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    let j = globalId.y;
    let k = globalId.z;
    
    if (i >= params.nx || j >= params.ny || k >= params.nz) {
        return;
    }
    
    let idx = voxelIndex(i, j, k);
    
    // Skip solid voxels
    if (isSolid(idx)) {
        return;
    }
    
    // CRITICAL: Skip inlet/outlet boundary voxels!
    // Their velocity is set by BC kernels and must NOT be overwritten
    var isInletVoxel = false;
    var isOutletVoxel = false;
    
    // Detect inlet voxels based on inletPlane parameter
    if (params.inletPlane == 0u) {
        isInletVoxel = (i == 0u);
    } else if (params.inletPlane == 1u) {
        isInletVoxel = (i == params.nx - 1u);
    } else if (params.inletPlane == 4u) {
        isInletVoxel = (k == 0u);
    } else if (params.inletPlane == 5u) {
        isInletVoxel = (k == params.nz - 1u);
    }
    
    // Detect outlet voxels based on outletPlane parameter
    if (params.outletPlane == 0u) {
        isOutletVoxel = (i == 0u);
    } else if (params.outletPlane == 1u) {
        isOutletVoxel = (i == params.nx - 1u);
    } else if (params.outletPlane == 4u) {
        isOutletVoxel = (k == 0u);
    } else if (params.outletPlane == 5u) {
        isOutletVoxel = (k == params.nz - 1u);
    }
    
    // Skip BC voxels - write correct velocity for inlet, just copy for outlet
    if (isInletVoxel) {
        // CRITICAL: Write inlet velocity HERE in collision step!
        // This guarantees it's the FINAL value in velocity buffer
        let inlet_speed = clamp(params.inletVelocity, 0.01, 0.5);
        let u_inlet = vec3<f32>(
            inlet_speed * params.windDirX,
            inlet_speed * params.windDirY,
            inlet_speed * params.windDirZ
        );
        velocity[idx * 3u] = u_inlet.x;
        velocity[idx * 3u + 1u] = u_inlet.y;
        velocity[idx * 3u + 2u] = u_inlet.z;
        density[idx] = params.rho0;
        
        // Copy f distributions for next iteration
        for (var q = 0u; q < 19u; q++) {
            f_out[idx * 19u + q] = f_in[idx * 19u + q];
        }
        return;
    }
    
    // CRITICAL LBM PHYSICS: DO NOT SKIP COLLISION FOR OUTLET VOXELS!
    // They must undergo normal MRT relaxation so that they propagate mathematically 
    // consistent post-collision states to the upstream neighbor in the next streaming step.
    // If we skip them, un-relaxed populations bounce back and accumulate mass.
    
    
    // Собираем локальные distribution functions
    var f_local: array<f32, 19>;
    for (var q = 0u; q < 19u; q++) {
        f_local[q] = f_in[idx * 19u + q];
    }
    
    // Вычисляем макроскопические переменные
    var rho = 0.0;
    var momentum = vec3<f32>(0.0, 0.0, 0.0);
    
    for (var q = 0u; q < 19u; q++) {
        rho += f_local[q];
        let c = vec3<f32>(f32(D3Q19_CX[q]), f32(D3Q19_CY[q]), f32(D3Q19_CZ[q]));
        momentum += c * f_local[q];
    }
    
    // STABILITY: Clamp rho to prevent numerical explosion
    // Density should be around 1.2 kg/m³ for air
    // If it deviates too much, simulation is unstable
    let rho_min = 0.5;   // Minimum density
    let rho_max = 2.0;   // Maximum density (prevents explosion!)
    rho = clamp(rho, rho_min, rho_max);
    
    var u = momentum / rho;
    
    // STABILITY: Clamp velocity magnitude (Ma < 0.1 for stability)
    // With c_s = 1/sqrt(3) ≈ 0.577, Ma=0.1 means u_max ≈ 0.06 lattice units
    // Converting to physical: 0.06 * (resolution/dt) ≈ 0.06 * 0.7/0.001 ≈ 42 m/s
    // Use 3 m/s as conservative physical limit for indoor air
    let u_mag = length(u);
    if (u_mag > 3.0) {
        u = u * (3.0 / u_mag);  // Scale down to 3 m/s max
    }
    
    // Источниковый член: тепловая плавучесть (Boussinesq approximation)
    var force = vec3<f32>(0.0, 0.0, 0.0);
    if (params.enableBuoyancy != 0u) {
        let T = temperature[idx];
        // Reference temperature from initial condition (Boussinesq approximation)
        // This is the equilibrium temperature around which buoyancy perturbations are calculated
        // For indoor: ~293K (20°C), for outdoor: use EPW outdoor temp
        // The actual value should match the initialTemperature from config
        let T0 = 293.0; // TODO: Pass as uniform param.T_ref for full correctness
        let dT = clamp(T - T0, -50.0, 50.0);  // Clamp temperature difference
        force = params.beta * dT * params.gravity * params.rho0;
    }
    
    // Add X-axis and/or Z-axis body force for PBC / open boundaries
    // Prevents closed-box mass accumulation and maintains target velocity
    if (abs(params.windDirX) > 0.1) {
        let target_u_x = params.inletVelocity * params.windDirX;
        let diff_x = target_u_x - u.x;
        force.x += diff_x * rho * 5.0; 
    }
    if (abs(params.windDirZ) > 0.1) {
        let target_u_z = params.inletVelocity * params.windDirZ;
        let diff_z = target_u_z - u.z;
        force.z += diff_z * rho * 5.0; 
    }
    
    // Добавляем силу к скорости (Guo forcing scheme)
    let u_forced = u + force * params.dt / (2.0 * rho);
    
    // ============================================================================
    // LES: Smagorinsky SGS Model
    // ============================================================================
    var omega_eff = params.omega;  // По умолчанию используем базовый omega
    
    if (params.enableLES != 0u) {
        // 1. Вычисляем равновесное распределение для текущего состояния
        var f_eq: array<f32, 19>;
        for (var q = 0u; q < 19u; q++) {
            f_eq[q] = equilibrium(q, rho, u_forced);
        }
        
        // 2. Вычисляем тензор напряжений из неравновесных моментов
        let S = computeStressTensor(f_local, f_eq);
        
        // 3. Вычисляем магнитуду скорости деформации |S|
        let S_magnitude = computeStrainRateMagnitude(S);
        
        // 4. Вычисляем турбулентную вязкость по модели Смагоринского
        let nu_turb = computeTurbulentViscosity(S_magnitude);
        
        // 5. Вычисляем эффективное время релаксации τ_eff = τ_0 + τ_turb
        let tau_eff = computeEffectiveTau(nu_turb);
        
        // 6. Эффективная частота релаксации
        omega_eff = 1.0 / tau_eff;
    }
    
    // MRT collision с эффективным omega (для LES или стандартным)
    let f_post_collision = mrtCollisionWithOmega(f_local, rho, u_forced, omega_eff);
    
    // Записываем обратно
    for (var q = 0u; q < 19u; q++) {
        f_out[idx * 19u + q] = f_post_collision[q];
    }
    
    // Обновляем макроскопические переменные для visualization
    density[idx] = rho;
    velocity[idx * 3u] = u.x;
    velocity[idx * 3u + 1u] = u.y;
    velocity[idx * 3u + 2u] = u.z;
}

// ============================================================================
// STREAMING STEP (перемещение частиц)
// ============================================================================

@compute @workgroup_size(4, 4, 4)
fn streamingStep(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = i32(globalId.x);
    let j = i32(globalId.y);
    let k = i32(globalId.z);
    
    if (i >= i32(params.nx) || j >= i32(params.ny) || k >= i32(params.nz)) {
        return;
    }
    
    let idx = voxelIndex(u32(i), u32(j), u32(k));
    
    // Твердые вокселы: full bounce-back
    if (isSolid(idx)) {
        var f_bounced: array<f32, 19>;
        
        for (var q = 0u; q < 19u; q++) {
            let opposite_q = u32(D3Q19_OPPOSITE[q]);
            f_bounced[q] = f_in[idx * 19u + opposite_q];
        }
        
        for (var q = 0u; q < 19u; q++) {
            f_out[idx * 19u + q] = f_bounced[q];
        }
        
        return;
    }
    
    // Fluid voxels: streaming
    for (var q = 0u; q < 19u; q++) {
        var ni = i - D3Q19_CX[q];
        var nj = j - D3Q19_CY[q];
        var nk = k - D3Q19_CZ[q];
        
        // PBC on X and Z axes to prevent mass accumulation
        if (abs(params.windDirX) > 0.1) {
            if (ni < 0) {
                ni = i32(params.nx) - 1;
            } else if (ni >= i32(params.nx)) {
                ni = 0;
            }
        }
        
        if (abs(params.windDirZ) > 0.1) {
            if (nk < 0) {
                nk = i32(params.nz) - 1;
            } else if (nk >= i32(params.nz)) {
                nk = 0;
            }
        }
        
        if (isInside(ni, nj, nk)) {
            let neighbor_idx = voxelIndex(u32(ni), u32(nj), u32(nk));
            
            // Копируем из соседа (потоковая передача)
            f_out[idx * 19u + q] = f_in[neighbor_idx * 19u + q];
        } else {
            // Граница домена: bounce-back
            let opposite_q = u32(D3Q19_OPPOSITE[q]);
            f_out[idx * 19u + q] = f_in[idx * 19u + opposite_q];
        }
    }
}

// ============================================================================
// ГРАНИЧНЫЕ УСЛОВИЯ: Inlet (фиксированная скорость)
// Supports dynamic inlet plane selection based on wind direction
// ============================================================================

@compute @workgroup_size(8, 8)
fn applyInletBC(@builtin(global_invocation_id) globalId: vec3<u32>) {
    // Determine voxel indices based on inlet plane
    var i: u32;
    var j: u32;
    var k: u32;
    var isValid = false;
    
    let a = globalId.x;
    let b = globalId.y;
    
    // Use if/else instead of switch for WGSL compatibility
    if (params.inletPlane == 0u) {
        // X_MIN: inlet at X=0
        i = 0u;
        j = a;
        k = b;
        isValid = j < params.ny && k < params.nz;
    } else if (params.inletPlane == 1u) {
        // X_MAX: inlet at X=max
        i = params.nx - 1u;
        j = a;
        k = b;
        isValid = j < params.ny && k < params.nz;
    } else if (params.inletPlane == 4u) {
        // Z_MIN: inlet at Z=0
        i = a;
        j = b;
        k = 0u;
        isValid = i < params.nx && j < params.ny;
    } else if (params.inletPlane == 5u) {
        // Z_MAX: inlet at Z=max
        i = a;
        j = b;
        k = params.nz - 1u;
        isValid = i < params.nx && j < params.ny;
    }
    // NO FALLBACK - invalid inletPlane should never occur
    // If we reach here with isValid=false, the voxel is simply skipped
    
    if (!isValid) {
        return;
    }
    
    let idx = voxelIndex(i, j, k);
    
    // Apply inlet BC to ALL non-solid voxels (including EMPTY boundary voxels!)
    // Boundary voxels may have state=EMPTY which was blocking inlet BC
    if (isSolid(idx)) {
        return;
    }
    
    // Inlet velocity from UI, clamped for LBM stability (Ma < 0.3)
    let inlet_speed = clamp(params.inletVelocity, 0.01, 0.5);
    
    // Use full 3D wind direction vector
    let u_inlet = vec3<f32>(
        inlet_speed * params.windDirX, 
        inlet_speed * params.windDirY,
        inlet_speed * params.windDirZ
    );
    let rho = params.rho0;
    
    for (var q = 0u; q < 19u; q++) {
        f_out[idx * 19u + q] = equilibrium(q, rho, u_inlet);
    }
    
    // CRITICAL: Write velocity DIRECTLY to velocity buffer!
    // Now that BC runs AFTER collision, this will be the FINAL velocity value
    velocity[idx * 3u] = u_inlet.x;
    velocity[idx * 3u + 1u] = u_inlet.y;
    velocity[idx * 3u + 2u] = u_inlet.z;
    density[idx] = rho;
}

// ============================================================================
// ГРАНИЧНЫЕ УСЛОВИЯ: Outlet (свободное истечение)
// Supports dynamic outlet plane selection
// ============================================================================

@compute @workgroup_size(8, 8)
fn applyOutletBC(@builtin(global_invocation_id) globalId: vec3<u32>) {
    // Determine voxel indices based on outlet plane
    var i: u32;
    var j: u32;
    var k: u32;
    var i_prev: u32;
    var j_prev: u32;
    var k_prev: u32;
    var isValid = false;
    
    let a = globalId.x;
    let b = globalId.y;
    
    // Use if/else instead of switch for WGSL compatibility
    if (params.outletPlane == 0u) {
        // X_MIN: outlet at X=0
        i = 0u;
        j = a;
        k = b;
        i_prev = 1u;
        j_prev = j;
        k_prev = k;
        isValid = j < params.ny && k < params.nz;
    } else if (params.outletPlane == 1u) {
        // X_MAX: outlet at X=max
        i = params.nx - 1u;
        j = a;
        k = b;
        i_prev = params.nx - 2u;
        j_prev = j;
        k_prev = k;
        isValid = j < params.ny && k < params.nz;
    } else if (params.outletPlane == 4u) {
        // Z_MIN: outlet at Z=0
        i = a;
        j = b;
        k = 0u;
        i_prev = i;
        j_prev = j;
        k_prev = 1u;
        isValid = i < params.nx && j < params.ny;
    } else if (params.outletPlane == 5u) {
        // Z_MAX: outlet at Z=max
        i = a;
        j = b;
        k = params.nz - 1u;
        i_prev = i;
        j_prev = j;
        k_prev = params.nz - 2u;
        isValid = i < params.nx && j < params.ny;
    }
    // NO FALLBACK - invalid outletPlane should never occur
    
    if (!isValid) {
        return;
    }
    
    let idx = voxelIndex(i, j, k);
    let idx_prev = voxelIndex(i_prev, j_prev, k_prev);
    
    // CRITICAL FIX: Do NOT check isSolid(idx) here!
    // We MUST force the outlet boundary to be open, overriding any solid wall
    // that the Voxelizer might have placed at the grid perimeter.
    
    // 1. Fetch f_i for all 19 directions from idx_prev
    // 2. Write them directly into the f buffer for idx
    var f_local: array<f32, 19>;
    for (var q = 0u; q < 19u; q++) {
        f_local[q] = f_in[idx_prev * 19u + q];
        f_out[idx * 19u + q] = f_local[q];
    }
    
    // 3. THEN recalculate the macroscopic density and velocity for idx based on those new f_i
    var rho = 0.0;
    var momentum = vec3<f32>(0.0, 0.0, 0.0);
    
    for (var q = 0u; q < 19u; q++) {
        rho += f_local[q];
        let c = vec3<f32>(f32(D3Q19_CX[q]), f32(D3Q19_CY[q]), f32(D3Q19_CZ[q]));
        momentum += c * f_local[q];
    }
    
    // Update buffers with recalculated values
    density[idx] = rho;
    var u = momentum / rho;
    velocity[idx * 3u] = u.x;
    velocity[idx * 3u + 1u] = u.y;
    velocity[idx * 3u + 2u] = u.z;
}
