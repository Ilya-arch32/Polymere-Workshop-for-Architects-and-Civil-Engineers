/**
 * ProjectManager.js
 * Manages multiple projects, persistence, and state switching.
 * V3 Refactor: Isolated Storage
 */

export class ProjectManager {
    constructor() {
        this.projects = []; // Metadata only
        this.activeProject = null; // Full project object

        // Base Keys
        this.BASE_KEY_META = 'neighbly_v3_meta';
        this.KEY_CURRENT_ID = 'neighbly_v3_current_id';
        this.PREFIX_PROJ = 'neighbly_proj_';

        // API Base URL (hardcoded for now, or detect dev/prod)
        // If we are on localhost/127.0.0.1, assume backend is at :8000 if not same origin
        this.API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://localhost:8000'
            : ''; // Relative path for production

        // Current User Context (null = guest)
        this.currentUserId = null;

        // Dynamic Key (derived from user)
        this.currentMetaKey = this.BASE_KEY_META;

        // Legacy Key for migration
        this.KEY_LEGACY = 'neighbly_projects';

        this.autosaveInterval = null;

        // Current User Context (null = guest)
        this.currentUserId = null;

        // Dynamic Keys (derived from user)
        this.currentMetaKey = this.BASE_KEY_META;
        this.currentDescripKey = 'neighbly_proj_desc_'; // Not used yet but good for future
        this.currentProjPrefix = this.PREFIX_PROJ; // Will become neighbly_proj_{uid}_


        // Default empty state
        this.defaultState = {
            name: 'New Project',
            created: Date.now(),
            lastModified: Date.now(),
            data: {
                ifcMetadata: null,
                epwFile: null,
                params: {},
                results: null,
                cuts: [],
                excludedElements: [],
                chatHistory: [],
                windowLayout: {}
            }
        };
    }

    // Switch context when user logs in/out
    // Switch context when user logs in/out
    setUserId(uid) {
        if (this.currentUserId === uid) return; // No change

        console.log(`[ProjectManager] 👤 Switching User Context: ${this.currentUserId} -> ${uid}`);
        this.currentUserId = uid;

        // Update Metadata Key & Project Prefix
        if (uid) {
            this.currentMetaKey = `${this.BASE_KEY_META}_${uid}`;
            this.currentProjPrefix = `${this.PREFIX_PROJ}${uid}_`;
        } else {
            this.currentMetaKey = this.BASE_KEY_META; // Guest uses default key
            this.currentProjPrefix = this.PREFIX_PROJ;
        }

        // CRITICAL: Clear current state from memory to prevent "ghost" data
        this.projects = [];
        this.activeProject = null;

        // Reload everything for the new context
        this.init();
    }

    // Initialize: Migration -> Load Meta -> Load Current
    async init() {
        console.log(`[ProjectManager] 📂 Initializing for context: ${this.currentUserId || 'Guest'}`);

        // 1. Check for legacy migration (only for guest/default mainly, but good to keep)
        if (!this.currentUserId) {
            this.checkAndMigrate();
        }

        // 2. Load Metadata (from localStorage first)
        this.loadMetadata();

        // 3. KEY FIX: If logged in, ALWAYS fetch project list from SERVER to ensure sync
        if (this.currentUserId) {
            console.log('[ProjectManager] 📡 Fetching project list from server (Source of Truth)...');
            await this.syncProjectListFromServer();
        }

        // 4. Load Current Project
        const currentIdKey = this.currentUserId ? `${this.KEY_CURRENT_ID}_${this.currentUserId}` : this.KEY_CURRENT_ID;
        let currentId = localStorage.getItem(currentIdKey);

        // If no currentId saved but we have projects, pick the most recent
        if (!currentId && this.projects.length > 0) {
            const sorted = [...this.projects].sort((a, b) => b.lastModified - a.lastModified);
            currentId = sorted[0].id;
            localStorage.setItem(currentIdKey, currentId);
            console.log(`   ⮑ No saved current ID. Defaulting to most recent: ${currentId}`);
        }

        if (currentId && this.getProjectMetadata(currentId)) {
            console.log(`   ⮑ Loading active project: ${currentId}`);
            await this.loadProjectIntoMemory(currentId);
        } else if (currentId && this.currentUserId) {
            // ID exists but not in metadata - try loading from server directly
            console.log(`   ⮑ Project ${currentId} not in metadata. Trying server directly...`);
            await this.loadProjectIntoMemory(currentId);
        } else {
            console.log('   ⮑ No active project found for this user. Checking list...');
            if (this.projects.length > 0) {
                await this.setCurrentProject(this.projects[0].id);
            } else {
                console.log('   ⮑ Creating default project for new/empty user context');
                await this.createProject(`New Project (${this.currentUserId ? 'Cloud' : 'Local'})`, true);
            }
        }

        this.startAutosave();

        console.log(`[ProjectManager] ✅ Init complete. Active: ${this.activeProject?.name || 'NONE'}`);
    }

    // Fetch project list from server (when localStorage is empty or for sync)
    async syncProjectListFromServer() {
        try {
            const headers = await this.getAuthHeaders();
            const response = await fetch(`${this.API_BASE}/api/projects`, {
                method: 'GET',
                headers: headers
            });

            if (!response.ok) {
                console.warn('[ProjectManager] ⚠️ Failed to fetch project list from server:', response.status);
                return;
            }

            const data = await response.json();
            if (data.projects && data.projects.length > 0) {
                console.log(`[ProjectManager] 📡 Found ${data.projects.length} projects on server`);

                // Merge server projects with local projects (server wins on conflict/update)
                // We want to preserve local-only projects that haven't been synced yet?
                // For now, let's assume Server is the Master list for the logged-in user.

                const serverProjects = data.projects;

                // Update local metadata & Sync to Firestore
                // PERF: Parallelize Firestore syncs to reduce wait time
                const syncPromises = [];
                for (const serverProj of serverProjects) {
                    const existingIdx = this.projects.findIndex(p => p.id === serverProj.id);
                    if (existingIdx !== -1) {
                        this.projects[existingIdx] = { ...this.projects[existingIdx], ...serverProj };
                    } else {
                        this.projects.push(serverProj);
                    }

                    // ALSO SYNC TO FIRESTORE (Metadata Only)
                    syncPromises.push(this.syncToFirebase(serverProj));
                }

                await Promise.all(syncPromises);

                this.saveMetadata(); // Cache to localStorage

                // If we have projects but no current ID, set it
                const currentIdKey = `${this.KEY_CURRENT_ID}_${this.currentUserId}`;
                if (!localStorage.getItem(currentIdKey)) {
                    // Set current to most recent
                    const sorted = [...this.projects].sort((a, b) => b.lastModified - a.lastModified);
                    localStorage.setItem(currentIdKey, sorted[0].id);
                }

                // Trigger UI refresh if global function exists
                if (window.loadUserProjects) {
                    console.log('[ProjectManager] 🔄 Refreshing UI Project List...');
                    // Small delay to ensure Firestore writes have propagated (optimistic)
                    await new Promise(r => setTimeout(r, 500));
                    window.loadUserProjects(this.currentUserId);
                }

            } else {
                console.log('[ProjectManager] 📡 No projects found on server');
            }
        } catch (e) {
            console.error('[ProjectManager] ❌ syncProjectListFromServer error:', e);
        }
    }

    // Helper: Get all projects metadata
    getAllProjects() {
        return this.projects || [];
    }

    // Helper: Get project metadata by ID
    getProjectMetadata(id) {
        return this.projects.find(p => p.id === id);
    }

    // Helper: Get current active project
    getCurrentProject() {
        return this.activeProject;
    }

    checkAndMigrate() {
        const legacyData = localStorage.getItem(this.KEY_LEGACY);
        if (!legacyData) return;

        console.log('📦 Found legacy project data. Migrating to V3...');
        try {
            const parsed = JSON.parse(legacyData);
            const legacyProjects = parsed.projects || [];
            const legacyCurrent = parsed.currentProjectId;

            if (legacyProjects.length === 0) {
                localStorage.removeItem(this.KEY_LEGACY);
                return;
            }

            const newMeta = [];

            // Split and save each project
            legacyProjects.forEach(p => {
                // Ensure it has an ID
                if (!p.id) p.id = 'proj_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

                // Save Data using current prefix (which is default/guest at this point of migration usually)
                const fullProject = {
                    id: p.id,
                    name: p.name,
                    created: p.created,
                    lastModified: p.lastModified,
                    data: p.data || JSON.parse(JSON.stringify(this.defaultState.data))
                };
                localStorage.setItem(this.currentProjPrefix + p.id, JSON.stringify(fullProject));

                // Add to Meta
                newMeta.push({
                    id: p.id,
                    name: p.name,
                    created: p.created,
                    lastModified: p.lastModified
                });
                console.log(`   ✓ Migrated: ${p.name} (${p.id})`);
            });

            // Save Meta
            localStorage.setItem(this.KEY_META, JSON.stringify(newMeta));

            // Set Current
            if (legacyCurrent) {
                localStorage.setItem(this.KEY_CURRENT_ID, legacyCurrent);
            }

            // Rename legacy key to back it up (or delete strictly)
            // We'll rename it to avoid re-migration but keep a safety backup
            localStorage.setItem(this.KEY_LEGACY + '_backup', legacyData);
            localStorage.removeItem(this.KEY_LEGACY);

            console.log('   ✨ Migration complete.');

        } catch (e) {
            console.error('   ✗ Migration failed:', e);
        }
    }

    loadMetadata() {
        const json = localStorage.getItem(this.currentMetaKey);
        if (json) {
            try {
                this.projects = JSON.parse(json);
                // Sort by last modified desc
                this.projects.sort((a, b) => b.lastModified - a.lastModified);
            } catch (e) {
                console.error('Failed to parse project metadata', e);
                this.projects = [];
            }
        } else {
            this.projects = [];
        }
    }



    // --- SERVER API INTEGRATION ---

    async getAuthHeaders() {
        if (!window.firebaseAuth || !window.firebaseAuth.currentUser) return {};
        try {
            const token = await window.firebaseAuth.currentUser.getIdToken();
            return {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
                'X-User-ID': window.firebaseAuth.currentUser.uid // Redundant but helpful for debug/validation
            };
        } catch (e) {
            console.error("Failed to get ID token", e);
            return {};
        }
    }

    async saveToServer(project) {
        if (!this.currentUserId) return false;

        console.log(`[ProjectManager] ☁️ Uploading to Server: ${this.API_BASE}/api/projects/${project.id}`);
        try {
            const headers = await this.getAuthHeaders();
            const response = await fetch(`${this.API_BASE}/api/projects/${project.id}`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(project)
            });

            const sizeBytes = new TextEncoder().encode(JSON.stringify(project)).length;
            const sizeMB = (sizeBytes / (1024 * 1024)).toFixed(2);
            console.log(`[ProjectManager] 📦 Payload Size: ${sizeMB} MB`);

            if (!response.ok) {
                if (response.status === 413) {
                    console.warn('[ProjectManager] ⚠️ Payload too large (413). Retrying without heavy data...');
                    // Clone and strip heavy data
                    const slimProject = JSON.parse(JSON.stringify(project));
                    if (slimProject.data) {
                        slimProject.data.epwData = null; // Strip EPW Data
                        if (slimProject.data.results) {
                            slimProject.data.results._displayOnly = true; // Ensure flag
                            // Note: We already strip raw results in preview.html, but maybe not enough?
                        }
                    }

                    // Retry Upload
                    const retryResponse = await fetch(`${this.API_BASE}/api/projects/${project.id}`, {
                        method: 'POST',
                        headers: headers,
                        body: JSON.stringify(slimProject)
                    });

                    if (!retryResponse.ok) {
                        const err = await retryResponse.json();
                        throw new Error(`Retry failed: ${err.detail || retryResponse.statusText}`);
                    }
                    console.log(`[ProjectManager] ✅ Server Upload Success (Slim Version)`);
                    return true;
                }

                const err = await response.json();
                throw new Error(err.detail || 'Server upload failed');
            }
            console.log(`[ProjectManager] ✅ Server Upload Success`);
            return true;
        } catch (e) {
            console.error(`[ProjectManager] ❌ Server Upload Error:`, e);
            return false;
        }
    }

    async loadFromServer(projectId) {
        if (!this.currentUserId) return null;

        console.log(`[ProjectManager] ☁️ Downloading from Server: ${projectId}`);
        try {
            const headers = await this.getAuthHeaders();
            const response = await fetch(`${this.API_BASE}/api/projects/${projectId}`, {
                method: 'GET',
                headers: headers
            });

            if (!response.ok) {
                if (response.status === 404) return null;
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            console.log(`[ProjectManager] ✅ Server Download Success`);
            return data;
        } catch (e) {
            console.error(`[ProjectManager] ❌ Server Download Error:`, e);
            return null;
        }
    }

    async deleteFromServer(projectId) {
        if (!this.currentUserId) return false;

        console.log(`[ProjectManager] ☁️ Deleting from Server: ${projectId}`);
        try {
            const headers = await this.getAuthHeaders();
            const response = await fetch(`${this.API_BASE}/api/projects/${projectId}`, {
                method: 'DELETE',
                headers: headers
            });

            if (!response.ok) throw new Error('Delete failed');
            return true;
        } catch (e) {
            console.error(`[ProjectManager] ❌ Server Delete Error:`, e);
            return false;
        }
    }

    // Load full project data into memory (this.activeProject)
    async loadProjectIntoMemory(id) {
        console.log(`[ProjectManager] 📥 Loading project into memory: ${id}`);

        let project = null;

        // 1. If Logged In -> Try Cache First, then Server (SOURCE OF TRUTH)
        if (this.currentUserId) {
            // Check Local Cache Validity
            let useCache = false;
            const key = this.currentProjPrefix + id;
            try {
                const cachedJson = localStorage.getItem(key);
                if (cachedJson) {
                    const cachedProj = JSON.parse(cachedJson);
                    const serverMeta = this.getProjectMetadata(id);

                    // If we have server metadata, compare timestamps
                    if (serverMeta) {
                        // Trust cache if it's same or newer than server
                        // Note: serverMeta.lastModified comes from the sync we just did
                        if (cachedProj.lastModified >= serverMeta.lastModified) {
                            console.log(`[ProjectManager] ⚡ Project ${id} is up-to-date in cache. Using local.`);
                            project = cachedProj;
                            useCache = true;
                        } else {
                            console.log(`[ProjectManager] ⬇️ Cache stale (Local: ${cachedProj.lastModified} < Server: ${serverMeta.lastModified}). Downloading...`);
                        }
                    } else {
                        // Metadata missing? Safest to download.
                        console.warn(`[ProjectManager] ⚠️ Metadata missing for ${id}. Downloading...`);
                    }
                }
            } catch (e) {
                console.warn('[ProjectManager] Cache check failed:', e);
            }

            if (!useCache) {
                project = await this.loadFromServer(id);
                if (project) {
                    console.log('[ProjectManager] ✅ Loaded from SERVER');
                    // Cache to localStorage for faster next load
                    try {
                        localStorage.setItem(key, JSON.stringify(project));
                    } catch (e) {
                        console.warn('[ProjectManager] ⚠️ localStorage cache failed (size?):', e.message);
                    }
                }
            }
        }

        // 2. Fallback to LocalStorage (Guest or Server miss)
        if (!project) {
            const key = this.currentProjPrefix + id;
            const json = localStorage.getItem(key);
            if (json) {
                try {
                    project = JSON.parse(json);
                    console.log('[ProjectManager] ⚠️ Loaded from localStorage (fallback)');
                } catch (e) { console.error(e); }
            }
        }

        // 3. Nothing found
        if (!project) {
            console.warn(`[ProjectManager] ⚠️ Project data not found anywhere: ${id}`);
            return false;
        }

        // 4. Basic validation
        if (!project.id || !project.data) {
            console.warn(`[ProjectManager] ⚠️ Invalid project structure (missing .id or .data)`);
            return false;
        }

        // 5. Set active
        this.activeProject = project;
        if (typeof window.currentProjectId !== 'undefined') {
            window.currentProjectId = id;
        }

        // 6. Dispatch Event for UI to update
        console.log(`[ProjectManager] 📢 Dispatching project-loaded event: "${project.name}"`);
        window.dispatchEvent(new CustomEvent('neighbly-project-loaded', { detail: this.activeProject }));

        return true;
    }

    saveMetadata() {
        localStorage.setItem(this.currentMetaKey, JSON.stringify(this.projects));
    }

    // Helper: Sync to Firebase (Metadata ONLY now, since Server has full data)
    async syncToFirebase(project) {
        const auth = window.firebaseAuth;
        const db = window.firestore;
        const appDb = window.firebaseDb;

        if (!auth || !auth.currentUser || !db || !appDb) return;
        if (this.currentUserId && this.currentUserId !== auth.currentUser.uid) return;

        // Define valid list check...
        const isValid = this.projects.some(p => p.id === project.id);
        if (!isValid) return;

        try {
            const user = auth.currentUser;
            const projectRef = db.doc(appDb, "users", user.uid, "projects", project.id);

            // Save METADATA ONLY to Firestore (Catalog)
            const payload = {
                id: project.id,
                name: project.name,
                lastModified: project.lastModified,
                created: project.created,
                synced: true,
                serverStored: true // Flag to indicate data is on VPS
            };

            await db.setDoc(projectRef, payload, { merge: true });
            console.log(`[ProjectManager] ✅ Firestore Metadata Sync Success`);
        } catch (e) {
            console.error(`[ProjectManager] ❌ Firestore Sync Failed:`, e);
        }
    }

    // Create a new project
    async createProject(name, isInitial = false) {
        const id = 'proj_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const now = Date.now();

        // 1. Create full project object
        const newProject = {
            id: id,
            name: name || `Project ${this.projects.length + 1}`,
            created: now,
            lastModified: now,
            data: JSON.parse(JSON.stringify(this.defaultState.data))
        };

        // 2. Save (LocalStorage as cache + Server)
        try {
            // Write to LS for immediate responsiveness/cache
            localStorage.setItem(this.currentProjPrefix + id, JSON.stringify(newProject));

            // Upload to Server
            if (this.currentUserId) {
                await this.saveToServer(newProject);
            }

        } catch (e) {
            console.error('Storage full?', e);
            alert('Cannot create project: Storage full or Server Error.');
            return null;
        }

        // 3. Update Metadata
        const meta = {
            id: id,
            name: newProject.name,
            created: now,
            lastModified: now
        };
        this.projects.unshift(meta); // Add to top
        this.saveMetadata(); // Local Metadata

        // 4. Firestore Sync
        if (this.currentUserId) {
            await this.syncToFirebase(newProject);
        }

        console.log(`   + Created project: ${newProject.name} (${id})`);

        if (isInitial) {
            // We need to set it properly, but setCurrentProject now is async-ish for loading
            // But we already have it in memory here?
            // To be safe and consistent with new load logic:
            this.activeProject = newProject;
            this.setCurrentProject(id); // This will re-trigger load but that's okay, or we optimize
        }

        return id;
    }

    async deleteProject(id) {
        // Prevent deleting the last project
        if (this.projects.length <= 1) {
            console.warn('   ⚠️ Cannot delete the last project.');
            return false;
        }

        // 0. Pre-emptive active clear
        const currentId = localStorage.getItem(this.KEY_CURRENT_ID);
        if ((this.activeProject && this.activeProject.id === id) || currentId === id) {
            this.activeProject = null;
            localStorage.removeItem(this.KEY_CURRENT_ID);
        }

        // 1. Remove from Meta
        const index = this.projects.findIndex(p => p.id === id);
        if (index === -1) return false;
        this.projects.splice(index, 1);
        this.saveMetadata();

        // 2. Remove File (Local)
        localStorage.removeItem(this.currentProjPrefix + id);

        // 3. SERVER DELETE
        if (this.currentUserId) {
            await this.deleteFromServer(id);
            await this.syncDeleteToFirebase(id); // Also cleanup Firestore
        }

        return true;
    }

    async syncDeleteToFirebase(projectId) {
        if (!window.firebaseAuth || !window.firebaseAuth.currentUser || !window.firestore) return;

        const user = window.firebaseAuth.currentUser;
        console.log(`[ProjectManager] 🗑️ Syncing delete to Firebase: ${projectId}`);

        try {
            await window.firestore.deleteDoc(window.firestore.doc(window.firebaseDb, "users", user.uid, "projects", projectId));
            console.log(`[ProjectManager] ✅ Cloud Delete Success`);
        } catch (e) {
            console.error(`[ProjectManager] ❌ Cloud Delete Failed:`, e);
        }
    }

    // Update project metadata (name only for now)
    updateProject(id, updates) {
        // ... (Keep existing metadata logic)
        const meta = this.projects.find(p => p.id === id);
        if (!meta) return false;

        if (updates.name) {
            meta.name = updates.name;
            meta.lastModified = Date.now();
        }
        this.saveMetadata();

        // 2. Update active project if it matches
        if (this.activeProject && this.activeProject.id === id) {
            if (updates.name) {
                this.activeProject.name = updates.name;
                this.activeProject.lastModified = Date.now();
            }
            // Trigger a save to persist name change to server
            // We can't call saveProjectState because that needs a dataCollector.
            // We just save the active object directly.
            if (this.currentUserId) {
                this.saveToServer(this.activeProject);
            }
            localStorage.setItem(this.currentProjPrefix + id, JSON.stringify(this.activeProject));
        }
        // ...

        // Update Firestore Meta
        // We really should trigger a sync here too
        if (this.activeProject && this.activeProject.id === id) {
            this.syncToFirebase(this.activeProject);
        }

        return true;
    }

    // ... public API getters ...

    // Called BEFORE switching OUT of a project
    async saveProjectState(id, dataCollector) {
        console.log(`[ProjectManager] 💾 saveProjectState called for ${id}`);
        // We only save the ACTIVE project
        if (!this.activeProject || this.activeProject.id !== id) {
            console.warn(`Attempted to save inactive project ${id} (Active: ${this.activeProject?.id})`);
            return false;
        }

        // dataCollector returns current state object from UI
        let currentState;
        try {
            currentState = dataCollector();
        } catch (e) {
            console.error('❌ Data collector failed:', e);
            return false;
        }

        // Merge
        this.activeProject.data = { ...this.activeProject.data, ...currentState };
        this.activeProject.lastModified = Date.now();

        // 1. Save File
        try {
            // Cache locally for speed
            const key = this.currentProjPrefix + id;
            try {
                localStorage.setItem(key, JSON.stringify(this.activeProject));
            } catch (lsErr) {
                console.warn('[ProjectManager] localStorage cache failed:', lsErr.message);
            }

            // UPLOAD TO SERVER (await to ensure it's saved)
            if (this.currentUserId) {
                await this.saveToServer(this.activeProject);
            }

            // 2. Update Metadata
            const meta = this.projects.find(p => p.id === id);
            if (meta) {
                meta.lastModified = this.activeProject.lastModified;
                this.saveMetadata();
            }

            // 3. Firestore (Meta only)
            if (this.currentUserId) {
                this.syncToFirebase(this.activeProject);
            }

            return true;
        } catch (e) {
            console.error('[ProjectManager] ❌ Save failed:', e);
            return false;
        }
    }

    async setCurrentProject(id) {
        // ...
        // Verify it exists in meta
        if (!this.getProjectMetadata(id)) {
            console.warn(`setCurrentProject: Project ${id} not found in metadata`);
            // We might want to allow loading even if not in meta (e.g. from deep link)?
            // For now adhere to strict list.
        }

        // Update localStorage pointer
        const currentIdKey = this.currentUserId ? `${this.KEY_CURRENT_ID}_${this.currentUserId}` : this.KEY_CURRENT_ID;
        localStorage.setItem(currentIdKey, id);

        // Load Mem
        // Note: loadProjectIntoMemory is now Async
        const loadSuccess = await this.loadProjectIntoMemory(id);
        if (!loadSuccess) {
            console.error(`setCurrentProject: Failed to load project ${id}`);
            return false;
        }

        console.log(`   ✅ setCurrentProject: Now active: ${this.activeProject?.name || id}`);
        return true;
    }

    // Legacy property compatibility
    get currentProjectId() {
        const currentIdKey = this.currentUserId ? `${this.KEY_CURRENT_ID}_${this.currentUserId}` : this.KEY_CURRENT_ID;
        return localStorage.getItem(currentIdKey);
    }

    set currentProjectId(val) {
        const currentIdKey = this.currentUserId ? `${this.KEY_CURRENT_ID}_${this.currentUserId}` : this.KEY_CURRENT_ID;
        if (val) localStorage.setItem(currentIdKey, val);
        else localStorage.removeItem(currentIdKey);
    }

    startAutosave() {
        if (this.autosaveInterval) clearInterval(this.autosaveInterval);
        console.log('[ProjectManager] ⏱️ Autosave timer started (60s)');
        this.autosaveInterval = setInterval(() => {
            // Dispatch event for UI to handle data collection and saving
            window.dispatchEvent(new CustomEvent('neighbly-autosave'));
        }, 60000);
    }
}
