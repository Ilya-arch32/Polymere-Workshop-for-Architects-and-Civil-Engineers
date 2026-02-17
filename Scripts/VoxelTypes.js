/**
 * AHI 2.0 Ultimate - Core Voxel Data Structures
 * Lead Architect: Systems Design Team
 *
 * Унифицированные типы для воксельной топологии - основа всей физики
 */
/**
 * Битовые маски состояния вокселя (для эффективного хранения в GPU buffers)
 */
export var VoxelState;
(function (VoxelState) {
    VoxelState[VoxelState["EMPTY"] = 0] = "EMPTY";
    VoxelState[VoxelState["SOLID"] = 1] = "SOLID";
    VoxelState[VoxelState["FLUID"] = 2] = "FLUID";
    VoxelState[VoxelState["GLASS"] = 4] = "GLASS";
    VoxelState[VoxelState["BOUNDARY"] = 8] = "BOUNDARY";
    // CHT Material-Aware states
    VoxelState[VoxelState["WINDOW"] = 16] = "WINDOW";
    VoxelState[VoxelState["DOOR"] = 32] = "DOOR";
    VoxelState[VoxelState["HEAT_SOURCE"] = 64] = "HEAT_SOURCE";
    VoxelState[VoxelState["EXTERNAL_WALL"] = 128] = "EXTERNAL_WALL";
})(VoxelState || (VoxelState = {}));
/**
 * Идентификаторы материалов (расширяемый enum)
 */
export var MaterialID;
(function (MaterialID) {
    MaterialID[MaterialID["AIR"] = 0] = "AIR";
    MaterialID[MaterialID["CONCRETE"] = 1] = "CONCRETE";
    MaterialID[MaterialID["WOOD"] = 2] = "WOOD";
    MaterialID[MaterialID["GLASS"] = 3] = "GLASS";
    MaterialID[MaterialID["INSULATION"] = 4] = "INSULATION";
    MaterialID[MaterialID["STEEL"] = 5] = "STEEL";
    MaterialID[MaterialID["BRICK"] = 6] = "BRICK";
    MaterialID[MaterialID["GYPSUM"] = 7] = "GYPSUM";
    // Special materials
    MaterialID[MaterialID["HEAT_SOURCE"] = 99] = "HEAT_SOURCE";
})(MaterialID || (MaterialID = {}));
/**
 * Константы физических моделей
 */
export const PHYSICS_CONSTANTS = {
    // LBM D3Q19
    LBM_RELAXATION_TIME: 0.6, // τ для BGK оператора
    LBM_LATTICE_SPEED: 0.1, // Скорость решетки (м/с)
    // CHT
    STEFAN_BOLTZMANN: 5.670374419e-8, // Вт/(м²·К⁴)
    AIR_DENSITY: 1.225, // кг/м³ при 15°C
    AIR_SPECIFIC_HEAT: 1005, // Дж/(кг·К)
    // Comfort
    PMV_METABOLIC_RATE: 1.2, // met (сидячая работа)
    PPD_THRESHOLD: 10, // % недовольных (целевой комфорт)
    // Spectral optics
    SPECTRAL_BINS: 16, // Дискретизация спектра
    WAVELENGTH_MIN: 380, // нм
    WAVELENGTH_MAX: 780, // нм
    // Neuroaesthetics (Research-based optimal values)
    FRACTAL_OPTIMAL_D: 1.4, // Оптимальная фрактальная размерность (Taylor et al.)
    FRACTAL_TOLERANCE: 0.2, // D ∈ [1.2, 1.6] = биофильный диапазон
    ENTROPY_OPTIMAL_MIN: 4.0, // Минимальная энтропия (бит) - не скучно
    ENTROPY_OPTIMAL_MAX: 6.0, // Максимальная энтропия (бит) - не хаос
    ISOVIST_EYE_HEIGHT: 1.5, // Высота глаз для расчета изовистов (м)
};
/**
 * Утилита для расчета индекса вокселя в линейном массиве
 */
export function voxelIndex(i, j, k, nx, ny) {
    return i + j * nx + k * nx * ny;
}
/**
 * Утилита для обратного преобразования индекса в координаты
 */
export function indexToCoords(index, nx, ny) {
    const k = Math.floor(index / (nx * ny));
    const remainder = index % (nx * ny);
    const j = Math.floor(remainder / nx);
    const i = remainder % nx;
    return [i, j, k];
}
