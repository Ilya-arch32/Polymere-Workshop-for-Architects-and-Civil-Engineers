/**
 * AI Model Rate Limiter
 * Per-model request caps with 5-hour cooldown, enforced via Firebase Firestore.
 * 
 * Firestore path: users/{uid}/ai_usage/{rateLimitModelKey}
 * Document shape: { count: number, cooldown_until: Timestamp | null }
 */

// =========================================================================
// TIER LIMITS CONFIGURATION
// =========================================================================
// Maps tier → rateLimitKey → message cap before 5h cooldown.
// Infinity = unlimited. 0 = model not available for that tier.
const AI_TIER_LIMITS = {
    Free: {
        trinity_large_preview: Infinity,
        gemini_3_flash: 0,
        claude_sonnet_4_6: 0,
        gemini_3_1_pro: 0,
        claude_opus_4_6: 0
    },
    Lite: {
        trinity_large_preview: 50,
        gemini_3_flash: 15,
        claude_sonnet_4_6: 0,
        gemini_3_1_pro: 0,
        claude_opus_4_6: 0
    },
    Pro: {
        trinity_large_preview: 200,
        gemini_3_flash: 50,
        claude_sonnet_4_6: 10,
        gemini_3_1_pro: 2,
        claude_opus_4_6: 0
    },
    Master: {
        trinity_large_preview: Infinity,
        gemini_3_flash: 100,
        claude_sonnet_4_6: 50,
        gemini_3_1_pro: 25,
        claude_opus_4_6: 10
    }
};

// Cooldown duration in milliseconds (5 hours)
const COOLDOWN_MS = 5 * 60 * 60 * 1000;

// =========================================================================
// MODEL ID MAPPING
// =========================================================================
// Maps the OpenRouter model ID (used in AI_MODEL global) to a rate-limit key.
// "Trinity Large Preview" is the branded default, separate from "Gemini 3 Flash".
const OPENROUTER_TO_RATELIMIT = {
    'google/gemini-3-flash': 'trinity_large_preview',  // default branded model
    'gemini-3-flash': 'gemini_3_flash',         // separate Gemini 3 Flash model
    'google/gemini-3.1-pro': 'gemini_3_1_pro',
    'anthropic/claude-sonnet-4.6': 'claude_sonnet_4_6',
    'anthropic/claude-opus-4.6': 'claude_opus_4_6'
};

// Display names for UI
const RATELIMIT_MODEL_LABELS = {
    trinity_large_preview: 'Trinity Large Preview',
    gemini_3_flash: 'Gemini 3 Flash',
    gemini_3_1_pro: 'Gemini 3.1 Pro',
    claude_sonnet_4_6: 'Claude Sonnet 4.6',
    claude_opus_4_6: 'Claude Opus 4.6'
};

// =========================================================================
// CORE: CHECK AND CONSUME AI TOKEN
// =========================================================================
/**
 * Checks if user can send a message with the given model, and consumes a token.
 * 
 * @param {string} uid - Firebase user ID
 * @param {string} userTier - 'Free' | 'Lite' | 'Pro' | 'Master'
 * @param {string} openRouterModelId - e.g. 'google/gemini-3-flash'
 * @returns {Promise<{allowed: boolean, remaining?: number, cooldownUntil?: Date, reason?: string}>}
 */
async function checkAndConsumeAIToken(uid, userTier, openRouterModelId) {
    const fs = window.firestore;
    const db = window.firebaseDb;

    if (!fs || !db) {
        return { allowed: false, reason: 'FIREBASE_NOT_READY' };
    }

    // Map OpenRouter model ID to rate-limit key
    const rateLimitKey = OPENROUTER_TO_RATELIMIT[openRouterModelId];
    if (!rateLimitKey) {
        console.error(`[RateLimiter] Unknown model ID: ${openRouterModelId}`);
        return { allowed: false, reason: 'UNKNOWN_MODEL' };
    }

    // Get tier limits
    const tierLimits = AI_TIER_LIMITS[userTier];
    if (!tierLimits) {
        console.error(`[RateLimiter] Unknown tier: ${userTier}`);
        return { allowed: false, reason: 'UNKNOWN_TIER' };
    }

    const limit = tierLimits[rateLimitKey];

    // Limit is 0 or undefined → model not available
    if (limit === undefined || limit === 0) {
        return { allowed: false, reason: 'MODEL_NOT_AVAILABLE' };
    }

    // Unlimited
    if (limit === Infinity) {
        return { allowed: true, remaining: Infinity };
    }

    // Fetch current usage from Firestore
    const usageDocRef = fs.doc(db, 'users', uid, 'ai_usage', rateLimitKey);
    let usageSnap;
    try {
        usageSnap = await fs.getDoc(usageDocRef);
    } catch (err) {
        console.error(`[RateLimiter] Firestore read error:`, err);
        return { allowed: false, reason: 'FIRESTORE_ERROR' };
    }

    const now = new Date();
    let count = 0;
    let cooldownUntil = null;

    if (usageSnap.exists()) {
        const data = usageSnap.data();
        count = data.count || 0;
        cooldownUntil = data.cooldown_until ? data.cooldown_until.toDate() : null;
    }

    // COOLDOWN CHECK: if cooldown is active and hasn't expired
    if (cooldownUntil && now < cooldownUntil) {
        const remaining = 0;
        console.log(`[RateLimiter] BLOCKED — ${rateLimitKey} on cooldown until ${cooldownUntil.toLocaleTimeString()}`);
        return {
            allowed: false,
            reason: 'COOLDOWN',
            remaining,
            cooldownUntil
        };
    }

    // RESET CHECK: if cooldown expired, reset counter
    if (cooldownUntil && now >= cooldownUntil) {
        count = 0;
        cooldownUntil = null;
    }

    // INCREMENT
    count++;

    // Build Firestore update payload
    const updatePayload = {
        count: count,
        cooldown_until: null,
        last_used: fs.Timestamp.now()
    };

    // CAP CHECK: if count >= limit → set cooldown (but allow THIS message)
    if (count >= limit) {
        const cooldownEnd = new Date(now.getTime() + COOLDOWN_MS);
        updatePayload.cooldown_until = fs.Timestamp.fromDate(cooldownEnd);
        console.log(`[RateLimiter] CAP HIT — ${rateLimitKey}: ${count}/${limit}. Cooldown until ${cooldownEnd.toLocaleTimeString()}`);
    }

    // Write to Firestore
    try {
        await fs.setDoc(usageDocRef, updatePayload);
    } catch (err) {
        console.error(`[RateLimiter] Firestore write error:`, err);
        return { allowed: false, reason: 'FIRESTORE_ERROR' };
    }

    const remaining = Math.max(0, limit - count);
    console.log(`[RateLimiter] ALLOWED — ${rateLimitKey}: ${count}/${limit} (${remaining} remaining)`);

    return {
        allowed: true,
        remaining,
        count,
        limit
    };
}

// =========================================================================
// GET MODEL USAGE STATUS (for dropdown rendering)
// =========================================================================
/**
 * Returns usage status for ALL models for the user's tier.
 * Used to render the model dropdown with remaining counts / cooldown info.
 * 
 * @param {string} uid - Firebase user ID
 * @param {string} userTier - 'Free' | 'Lite' | 'Pro' | 'Master'
 * @returns {Promise<Object>} Map of rateLimitKey → { available, limit, count, remaining, cooldownUntil, label }
 */
async function getModelUsageStatus(uid, userTier) {
    const fs = window.firestore;
    const db = window.firebaseDb;
    const result = {};

    const tierLimits = AI_TIER_LIMITS[userTier] || AI_TIER_LIMITS['Free'];
    const now = new Date();

    for (const [rateLimitKey, limit] of Object.entries(tierLimits)) {
        const status = {
            label: RATELIMIT_MODEL_LABELS[rateLimitKey] || rateLimitKey,
            limit: limit,
            available: limit > 0,
            count: 0,
            remaining: limit,
            cooldownUntil: null,
            onCooldown: false
        };

        if (limit === 0) {
            status.remaining = 0;
            result[rateLimitKey] = status;
            continue;
        }

        if (limit === Infinity) {
            status.remaining = Infinity;
            result[rateLimitKey] = status;
            continue;
        }

        // Read Firestore usage doc
        if (fs && db) {
            try {
                const usageDocRef = fs.doc(db, 'users', uid, 'ai_usage', rateLimitKey);
                const snap = await fs.getDoc(usageDocRef);
                if (snap.exists()) {
                    const data = snap.data();
                    let count = data.count || 0;
                    let cooldownUntil = data.cooldown_until ? data.cooldown_until.toDate() : null;

                    // If cooldown expired, reset
                    if (cooldownUntil && now >= cooldownUntil) {
                        count = 0;
                        cooldownUntil = null;
                    }

                    status.count = count;
                    status.remaining = Math.max(0, limit - count);
                    status.cooldownUntil = cooldownUntil;
                    status.onCooldown = !!(cooldownUntil && now < cooldownUntil);
                }
            } catch (err) {
                console.error(`[RateLimiter] Error reading usage for ${rateLimitKey}:`, err);
            }
        }

        result[rateLimitKey] = status;
    }

    return result;
}

// =========================================================================
// COOLDOWN TOAST UI
// =========================================================================
/**
 * Shows a dismissible toast notification for cooldown.
 * @param {string} modelLabel - Display name of the model
 * @param {string} unlockTime - Formatted time string (e.g. "17:21")
 */
function showAICooldownToast(modelLabel, unlockTime) {
    // Remove any existing toast
    const existing = document.getElementById('ai-cooldown-toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.id = 'ai-cooldown-toast';
    toast.style.cssText = `
        position: fixed;
        bottom: 80px;
        right: 24px;
        background: rgba(26, 26, 26, 0.95);
        color: #fff;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 13px;
        font-family: 'Manrope', sans-serif;
        z-index: 100000;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        animation: slideInToast 0.3s ease-out;
        max-width: 400px;
    `;
    toast.innerHTML = `
        <span class="material-icons" style="font-size: 20px; color: #ff9800;">schedule</span>
        <span>Limit reached for <strong>${modelLabel}</strong>. Unlocks at <strong>${unlockTime}</strong>.</span>
        <button onclick="this.parentElement.remove()" style="background:none;border:none;color:#888;cursor:pointer;padding:0;margin-left:8px;">
            <span class="material-icons" style="font-size:16px;">close</span>
        </button>
    `;

    // Add animation keyframes if not already present
    if (!document.getElementById('ai-toast-styles')) {
        const style = document.createElement('style');
        style.id = 'ai-toast-styles';
        style.textContent = `
            @keyframes slideInToast {
                from { transform: translateX(100px); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(toast);

    // Auto-dismiss after 6 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.transition = 'opacity 0.3s, transform 0.3s';
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100px)';
            setTimeout(() => toast.remove(), 300);
        }
    }, 6000);
}

// =========================================================================
// EXPOSE TO WINDOW
// =========================================================================
window.checkAndConsumeAIToken = checkAndConsumeAIToken;
window.getModelUsageStatus = getModelUsageStatus;
window.showAICooldownToast = showAICooldownToast;
window.AI_TIER_LIMITS = AI_TIER_LIMITS;
window.OPENROUTER_TO_RATELIMIT = OPENROUTER_TO_RATELIMIT;
window.RATELIMIT_MODEL_LABELS = RATELIMIT_MODEL_LABELS;
