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
    // Extended materials (ASHRAE Handbook — Fundamentals)
    MaterialID[MaterialID["LIGHTWEIGHT_CONCRETE"] = 8] = "LIGHTWEIGHT_CONCRETE";
    MaterialID[MaterialID["PLASTERBOARD"] = 9] = "PLASTERBOARD";
    MaterialID[MaterialID["FIBERGLASS"] = 10] = "FIBERGLASS";
    MaterialID[MaterialID["MINERAL_WOOL"] = 11] = "MINERAL_WOOL";
    MaterialID[MaterialID["CERAMIC_TILE"] = 12] = "CERAMIC_TILE";
    MaterialID[MaterialID["ALUMINUM"] = 13] = "ALUMINUM";
    MaterialID[MaterialID["STONE"] = 14] = "STONE";
    MaterialID[MaterialID["XPS_FOAM"] = 15] = "XPS_FOAM";
    // Special materials
    MaterialID[MaterialID["HEAT_SOURCE"] = 99] = "HEAT_SOURCE";
})(MaterialID || (MaterialID = {}));

/**
 * ASHRAE Handbook — Fundamentals: Standard Material Properties
 * Each entry: { name, rho (kg/m³), cp (J/kg·K), k (W/m·K), keywords (EN/RU) }
 * keywords used for Tier 2 string-matching fallback when IFC has no thermal props
 */
export const ASHRAE_MATERIALS_DB = {
    0: {
        name: 'Air', rho: 1.225, cp: 1005, k: 0.026,
        keywords: []
    },
    1: {
        name: 'Concrete (Heavy)', rho: 2400, cp: 880, k: 1.4,
        keywords: ['concrete', 'бетон', 'cement', 'цемент', 'железобетон']
    },
    2: {
        name: 'Wood / Timber', rho: 600, cp: 1700, k: 0.15,
        keywords: ['wood', 'timber', 'дерев', 'древес', 'lumber', 'фанер', 'plywood']
    },
    3: {
        name: 'Glass (Glazing)', rho: 2500, cp: 840, k: 1.0,
        keywords: ['glass', 'стекл', 'стекло', 'glazing', 'glazed', 'остеклен',
            'витраж', 'витрин', 'storefront', 'curtain', 'panel', 'панель',
            'skylight', 'transparent', 'window', 'окн']
    },
    4: {
        name: 'Insulation (Generic)', rho: 40, cp: 840, k: 0.04,
        keywords: ['insulation', 'insul', 'утеплит', 'теплоизоляц', 'пенопласт']
    },
    5: {
        name: 'Steel', rho: 7850, cp: 500, k: 50.0,
        keywords: ['steel', 'metal', 'сталь', 'металл', 'iron', 'железо']
    },
    6: {
        name: 'Brick (Masonry)', rho: 1800, cp: 900, k: 0.72,
        keywords: ['brick', 'masonry', 'кирпич', 'кладк', 'керамзит']
    },
    7: {
        name: 'Gypsum Board', rho: 800, cp: 1090, k: 0.16,
        keywords: ['gypsum', 'plaster', 'гипс', 'штукатур', 'drywall', 'гкл', 'gyproc']
    },
    8: {
        name: 'Lightweight Concrete', rho: 1400, cp: 1000, k: 0.53,
        keywords: ['lightweight', 'пенобетон', 'газобетон', 'газосиликат', 'aerated', 'autoclaved']
    },
    9: {
        name: 'Plasterboard', rho: 950, cp: 840, k: 0.16,
        keywords: ['plasterboard', 'гипсокартон', 'sheetrock', 'цсп']
    },
    10: {
        name: 'Fiberglass', rho: 12, cp: 840, k: 0.04,
        keywords: ['fiberglass', 'стекловолокн', 'стекловат', 'glasswool', 'glass wool']
    },
    11: {
        name: 'Mineral Wool', rho: 100, cp: 840, k: 0.038,
        keywords: ['mineral', 'минерал', 'rockwool', 'rock wool', 'минват', 'базальт', 'basalt']
    },
    12: {
        name: 'Ceramic Tile', rho: 2000, cp: 800, k: 1.3,
        keywords: ['ceramic', 'tile', 'плитк', 'керами', 'фаянс', 'porcelain']
    },
    13: {
        name: 'Aluminum', rho: 2700, cp: 900, k: 205.0,
        keywords: ['aluminum', 'aluminium', 'алюмин']
    },
    14: {
        name: 'Natural Stone', rho: 2600, cp: 900, k: 2.9,
        keywords: ['stone', 'granite', 'marble', 'камень', 'гранит', 'мрамор', 'limestone', 'известняк']
    },
    15: {
        name: 'XPS / EPS Foam', rho: 35, cp: 1400, k: 0.034,
        keywords: ['xps', 'eps', 'foam', 'пеноплекс', 'экструд', 'полистирол', 'styrofoam']
    },
    99: {
        name: 'Heat Source', rho: 7850, cp: 500, k: 50.0,
        keywords: ['radiator', 'heater', 'радиатор', 'отопл', 'батарея']
    }
};
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
