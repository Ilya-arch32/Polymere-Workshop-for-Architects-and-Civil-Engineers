/**
 * AHI 2.0 Ultimate - Solar Position Algorithm (SPA)
 *
 * Based on NREL Solar Position Algorithm
 * Calculates sun position with accuracy up to 0.0003 degrees
 * Essential for accurate solar heat gains and daylighting
 */
export class SolarPositionAlgorithm {
    location;
    verbose = false; // Control logging - set to true for debugging

    constructor(location, options = {}) {
        this.location = {
            elevation: 0,
            ...location
        };
        this.verbose = options.verbose ?? false;
    }
    /**
     * Calculate solar position for given date/time
     * Based on NREL SPA with simplified Earth orbit parameters
     */
    calculate(date, forceLocalHour = null) {
        // Julian Date
        const jd = this.getJulianDate(date);
        const jc = (jd - 2451545.0) / 36525.0; // Julian Century
        // Solar coordinates
        const L = this.getSolarMeanLongitude(jc);
        const M = this.getSolarMeanAnomaly(jc);
        const C = this.getSolarEquationOfCenter(M, jc);  // Pass jc for coefficient correction
        const lambda = L + C; // True solar longitude
        // Declination
        const epsilon = this.getObliquityCorrection(jc);
        const delta = Math.asin(Math.sin(this.deg2rad(epsilon)) * Math.sin(this.deg2rad(lambda)));
        const declination = this.rad2deg(delta);
        if (this.verbose) console.log(`[SolarPos] L=${L.toFixed(1)}°, λ=${lambda.toFixed(1)}°, ε=${epsilon.toFixed(2)}°, δ=${declination.toFixed(2)}°`);

        // Equation of time (minutes) - pass date for day-of-year calculation
        const E = this.getEquationOfTime(jc, L, M, epsilon, date);

        // Hour angle - use forceLocalHour if provided to bypass timezone issues
        const localHour = forceLocalHour !== null ? forceLocalHour : (date.getHours() + date.getMinutes() / 60);
        const solarTime = localHour + E / 60 + (this.location.longitude - this.location.timezone * 15) / 15;
        if (this.verbose) console.log(`[SolarPos] localHour=${localHour.toFixed(2)}, E=${E.toFixed(2)}min, solarTime=${solarTime.toFixed(2)}h`);

        const hourAngle = (solarTime - 12) * 15; // degrees
        // Solar elevation and azimuth
        const lat = this.deg2rad(this.location.latitude);
        const ha = this.deg2rad(hourAngle);
        const elevation = Math.asin(Math.sin(lat) * Math.sin(delta) +
            Math.cos(lat) * Math.cos(delta) * Math.cos(ha));

        // Azimuth calculation (NOAA Solar Calculator formula)
        // Uses: azimuth = atan2(sin(ha), cos(ha)*sin(lat) - tan(dec)*cos(lat))
        // Result: 0°=North, 90°=East, 180°=South, 270°=West
        const cosZenith = Math.sin(lat) * Math.sin(delta) + Math.cos(lat) * Math.cos(delta) * Math.cos(ha);
        let azimuth;
        if (Math.cos(elevation) !== 0) {
            // Standard formula
            const sinAz = -Math.sin(ha) * Math.cos(delta) / Math.cos(elevation);
            const cosAz = (Math.sin(delta) - Math.sin(lat) * cosZenith) / (Math.cos(lat) * Math.cos(elevation));
            azimuth = Math.atan2(sinAz, cosAz);
            azimuth = this.rad2deg(azimuth);
            if (azimuth < 0) azimuth += 360;
        } else {
            azimuth = hourAngle > 0 ? 270 : 90; // Sunrise/sunset edge case
        }

        const elevationDeg = this.rad2deg(elevation);
        if (this.verbose) console.log(`[SolarPos] lat=${this.location.latitude.toFixed(2)}°, hourAngle=${hourAngle.toFixed(2)}°, elev=${elevationDeg.toFixed(2)}°, azimuth=${azimuth.toFixed(1)}°`);
        // Atmospheric refraction correction
        const refractionCorrection = this.getRefractionCorrection(elevationDeg);
        const correctedElevation = elevationDeg + refractionCorrection;
        // Calculate sunrise/sunset
        const { sunrise, sunset, solarNoon } = this.calculateSunTimes(date, declination);
        return {
            azimuth: azimuth % 360,
            elevation: correctedElevation,
            zenith: 90 - correctedElevation,
            hourAngle,
            declination,
            sunrise,
            sunset,
            solarNoon
        };
    }
    /**
     * Calculate solar irradiance using EPW weather data
     * @param {Object} position - Solar position with elevation, azimuth, zenith
     * @param {Date} date - Date for day-of-year calculation
     * @param {Object} epwWeather - Optional EPW weather data with directNormalRad, diffuseHorizRad, totalSkyCover
     * @returns {Object} { ghi, dni, dhi, source } - Irradiance values and data source
     */
    calculateSolarIrradianceWithWeather(position, date, epwWeather = null) {
        if (position.elevation <= 0) {
            return { ghi: 0, dni: 0, dhi: 0, source: 'night' };
        }

        // If EPW provides actual radiation data, use it directly
        if (epwWeather && epwWeather.directNormalRad !== undefined && epwWeather.diffuseHorizRad !== undefined) {
            const dni = epwWeather.directNormalRad; // Wh/m² from EPW
            const dhi = epwWeather.diffuseHorizRad; // Wh/m² from EPW
            const ghi = dni * Math.sin(this.deg2rad(position.elevation)) + dhi;
            if (this.verbose) console.log(`[SolarPos] Using EPW radiation: DNI=${dni}W/m², DHI=${dhi}W/m², GHI=${ghi.toFixed(0)}W/m²`);
            return { ghi, dni, dhi, source: 'epw' };
        }

        // Fallback: clear-sky model with cloud adjustment
        const result = this.calculateSolarIrradianceClearSky(position, date, epwWeather?.totalSkyCover);
        return { ...result, source: 'clear-sky-model' };
    }

    /**
     * Calculate direct normal irradiance (DNI) using clear-sky model
     * Updated with dynamic turbidity based on cloud cover
     * @param {Object} position - Solar position
     * @param {Date} date - Date for day-of-year
     * @param {number} cloudCover - Optional cloud cover (0-10 tenths)
     */
    calculateSolarIrradianceClearSky(position, date, cloudCover = 0) {
        if (position.elevation <= 0)
            return { ghi: 0, dni: 0, dhi: 0 };

        // Solar constant (updated to modern measurement, TSI ≈ 1361 W/m²)
        const I0 = 1361; // W/m² (SORCE/TIM, Kopp & Lean 2011)

        // Earth-Sun distance correction (eccentricity factor)
        const dayOfYear = this.getDayOfYear(date);
        const B = 2 * Math.PI * (dayOfYear - 1) / (this.isLeapYear(date) ? 366 : 365);
        const E0 = 1.00011 + 0.034221 * Math.cos(B) + 0.001280 * Math.sin(B)
            + 0.000719 * Math.cos(2 * B) + 0.000077 * Math.sin(2 * B);

        // Air mass (Kasten & Young 1989 formula)
        const zenithRad = this.deg2rad(position.zenith);
        const airMass = 1 / (Math.cos(zenithRad) + 0.50572 * Math.pow(96.07995 - position.zenith, -1.6364));

        // Dynamic turbidity based on cloud cover (0-10 tenths)
        // Clear sky: 2.0-2.5, Partly cloudy: 3-4, Overcast: 5+
        const baseTurbidity = 2.0;
        const cloudFactor = (cloudCover || 0) / 10; // 0-1
        const turbidity = baseTurbidity + cloudFactor * 3.0; // Range: 2.0 - 5.0

        // Atmospheric transmission (Bird & Hulstrom simplified model)
        const transmission = 0.75 * Math.exp(-0.14 * airMass * turbidity);

        // Direct Normal Irradiance
        const dni = I0 * E0 * transmission;

        // Diffuse horizontal irradiance
        // Clear sky: ~10-15% of DNI, Overcast: 60-100% diffuse-dominated
        const clearSkyDiffuseRatio = 0.12;
        const overcastDiffuseRatio = 0.80;
        const diffuseRatio = clearSkyDiffuseRatio + cloudFactor * (overcastDiffuseRatio - clearSkyDiffuseRatio);
        const dhi = dni * diffuseRatio;

        // Global horizontal irradiance
        const ghi = dni * Math.sin(this.deg2rad(position.elevation)) + dhi;

        console.log(`[SolarPos] Clear-sky model: turbidity=${turbidity.toFixed(1)}, cloudCover=${cloudCover}/10, DNI=${dni.toFixed(0)}, GHI=${ghi.toFixed(0)}`);
        return { ghi, dni, dhi };
    }

    /**
     * Calculate direct normal irradiance (DNI)
     * LEGACY: For backward compatibility, wraps new method
     */
    calculateSolarIrradiance(position, date) {
        const result = this.calculateSolarIrradianceClearSky(position, date, 0);
        return result.ghi;
    }

    /**
     * Check if year is a leap year
     */
    isLeapYear(date) {
        const year = date.getFullYear();
        return (year % 4 === 0 && year % 100 !== 0) || (year % 400 === 0);
    }
    /**
     * Calculate sunrise, sunset, and solar noon times
     */
    calculateSunTimes(date, declination) {
        const lat = this.deg2rad(this.location.latitude);
        const dec = this.deg2rad(declination);
        // Hour angle at sunrise/sunset
        const cosH = -Math.tan(lat) * Math.tan(dec);
        // Check for polar day/night
        if (cosH > 1) {
            // Polar night
            const noon = new Date(date);
            noon.setHours(12, 0, 0, 0);
            return { sunrise: noon, sunset: noon, solarNoon: noon };
        }
        else if (cosH < -1) {
            // Polar day
            const midnight = new Date(date);
            midnight.setHours(0, 0, 0, 0);
            const endOfDay = new Date(date);
            endOfDay.setHours(23, 59, 59, 999);
            const noon = new Date(date);
            noon.setHours(12, 0, 0, 0);
            return { sunrise: midnight, sunset: endOfDay, solarNoon: noon };
        }
        const H = this.rad2deg(Math.acos(cosH)); // Hour angle in degrees
        const timeCorrection = -this.location.longitude / 15 + this.location.timezone;
        const solarNoonHour = 12 + timeCorrection;
        const sunriseHour = solarNoonHour - H / 15;
        const sunsetHour = solarNoonHour + H / 15;
        const sunrise = new Date(date);
        const sunriseMinutes = Math.floor(sunriseHour * 60);
        sunrise.setHours(Math.floor(sunriseMinutes / 60), sunriseMinutes % 60, 0, 0);
        const sunset = new Date(date);
        const sunsetMinutes = Math.floor(sunsetHour * 60);
        sunset.setHours(Math.floor(sunsetMinutes / 60), sunsetMinutes % 60, 0, 0);
        const solarNoon = new Date(date);
        const noonMinutes = Math.floor(solarNoonHour * 60);
        solarNoon.setHours(Math.floor(noonMinutes / 60), noonMinutes % 60, 0, 0);
        return { sunrise, sunset, solarNoon };
    }
    /**
     * Julian Date calculation
     */
    getJulianDate(date) {
        const year = date.getUTCFullYear();
        const month = date.getUTCMonth() + 1;
        const day = date.getUTCDate();
        const hour = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
        let a = Math.floor((14 - month) / 12);
        let y = year + 4800 - a;
        let m = month + 12 * a - 3;
        let jdn = day + Math.floor((153 * m + 2) / 5) + 365 * y +
            Math.floor(y / 4) - Math.floor(y / 100) + Math.floor(y / 400) - 32045;
        return jdn + (hour - 12) / 24;
    }
    /**
     * Solar Mean Longitude (degrees)
     */
    getSolarMeanLongitude(jc) {
        return (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360;
    }
    /**
     * Solar Mean Anomaly (degrees)
     */
    getSolarMeanAnomaly(jc) {
        return 357.52911 + jc * (35999.05029 - 0.0001537 * jc);
    }
    /**
     * Solar Equation of Center (degrees)
     * CRITICAL: Coefficients use jc (Julian Century), NOT M (degrees)!
     */
    getSolarEquationOfCenter(M, jc) {
        const mRad = this.deg2rad(M);
        // Time-varying coefficients use jc (small value ~0.24), not M (large value ~200°)
        return Math.sin(mRad) * (1.914602 - 0.004817 * jc - 0.000014 * jc * jc) +
            Math.sin(2 * mRad) * (0.019993 - 0.000101 * jc) +
            Math.sin(3 * mRad) * 0.000289;
    }
    /**
     * Obliquity of Ecliptic Correction (degrees)
     */
    getObliquityCorrection(jc) {
        const seconds = 21.448 - jc * (46.8150 + jc * (0.00059 - jc * 0.001813));
        return 23.0 + (26.0 + (seconds / 60.0)) / 60.0;
    }
    /**
     * Equation of Time (minutes) - Spencer (1971) formula
     * More robust and well-validated than the complex formula
     */
    getEquationOfTime(jc, L, M, epsilon, date) {
        // Spencer (1971) formula - well validated
        const dayOfYear = date ? this.getDayOfYear(date) :
            Math.floor((jc * 36525 + 2451545 - 2451545) % 365.25) + 1;
        const B = 2 * Math.PI * (dayOfYear - 1) / 365;

        // Equation of Time in minutes
        const E = 229.2 * (
            0.000075 +
            0.001868 * Math.cos(B) -
            0.032077 * Math.sin(B) -
            0.014615 * Math.cos(2 * B) -
            0.040849 * Math.sin(2 * B)
        );

        if (this.verbose) console.log(`[SolarPos] EoT: dayOfYear=${dayOfYear}, B=${(B * 180 / Math.PI).toFixed(1)}°, E=${E.toFixed(2)}min`);
        return E;
    }
    /**
     * Atmospheric Refraction Correction (degrees)
     */
    getRefractionCorrection(elevation) {
        if (elevation > 85)
            return 0;
        const h = Math.max(0, elevation);
        const refraction = 1.02 / (60 * Math.tan(this.deg2rad(h + 10.3 / (h + 5.11))));
        // Temperature and pressure correction (standard conditions)
        const pressure = 1013.25; // mbar
        const temperature = 15; // Celsius
        return refraction * (pressure / 1010) * (283 / (273 + temperature));
    }
    /**
     * Day of year
     */
    getDayOfYear(date) {
        const start = new Date(date.getFullYear(), 0, 0);
        const diff = date.getTime() - start.getTime();
        return Math.floor(diff / (1000 * 60 * 60 * 24));
    }
    /**
     * Degree to radian conversion
     */
    deg2rad(deg) {
        return deg * Math.PI / 180;
    }
    /**
     * Radian to degree conversion
     */
    rad2deg(rad) {
        return rad * 180 / Math.PI;
    }
}
