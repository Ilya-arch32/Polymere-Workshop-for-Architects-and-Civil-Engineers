/**
 * UI & Auth Logic for Neighbly
 * Handles Login, Logout, Main Menu Toggling, and Project List
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements - Updated IDs for new menu structure
    const authBtn = document.getElementById('auth-btn-menu');
    const authBtnText = document.getElementById('auth-btn-text');
    const userNameText = document.getElementById('user-info-display-menu');
    // const userEmailText = document.getElementById('user-email-text'); // No longer used in compact view
    // const userAvatarImg = document.getElementById('user-avatar-img'); // Avatar is in header, handled separately if possible

    // Validating presence
    if (!authBtn) console.warn('Auth button not found');

    // --- AUTHENTICATION ---

    // Auth State Listener
    if (window.onAuthStateChanged && window.firebaseAuth) {
        window.onAuthStateChanged(window.firebaseAuth, (user) => {
            const pm = window.projectManager;

            if (user) {
                // User is signed in

                document.body.classList.remove('guest-mode'); // Hide guest overlay, show workspace
                if (userNameText) userNameText.textContent = user.displayName || user.email;
                if (authBtnText) authBtnText.textContent = 'Sign Out';

                // Update header avatar
                const headerAvatar = document.querySelector('.user-profile .avatar-circle');
                if (headerAvatar && user.photoURL) {
                    headerAvatar.innerHTML = `<img src="${user.photoURL}" style="width: 100%; height: 100%; object-fit: cover;">`;
                }

                // Switch Project Context
                // Switch Project Context (Robust)
                const syncUserToPM = (uid) => {
                    if (window.projectManager) {
                        window.projectManager.setUserId(uid);
                    } else {
                        window.addEventListener('AHIModulesReady', () => {
                            if (window.projectManager) window.projectManager.setUserId(uid);
                        }, { once: true });
                    }
                };
                syncUserToPM(user.uid);

                // SECURITY: If we are on a specific project URL but just switched user context,
                // we might be looking at the previous user's project ID in the URL.
                // It's safer to clear it and let the PM load the last active project for THIS user.
                const urlParams = new URLSearchParams(window.location.search);
                if (urlParams.has('project')) {
                    // Check if this project actually belongs to the user (or is public?)
                    // For now, simple strict isolation: reload to default view for this user
                    // unless we want to support deep linking (which needs more checks).
                    const pid = urlParams.get('project');

                    // Optimization: If the project exists in the NEW user's local storage, we can keep it?
                    // But we don't know yet because PM just switched. 
                    // Let's just reload to be safe and clean.

                    // Actually, let's just clear it to force a clean "dashboard" or "last active" load
                    window.history.replaceState({}, document.title, window.location.pathname);
                    window.location.reload();
                    return;
                }

                loadUserProjects(user.uid);

            } else {
                // User is signed out

                document.body.classList.add('guest-mode'); // Show guest overlay, hide workspace
                if (userNameText) userNameText.textContent = 'Guest User';
                if (authBtnText) authBtnText.textContent = 'Sign In';

                const headerAvatar = document.querySelector('.user-profile .avatar-circle');
                if (headerAvatar) {
                    headerAvatar.innerHTML = `<span class="material-icons">person</span>`;
                }

                // Switch Project Context to Guest
                if (pm) {
                    pm.setUserId(null);
                }

                // Clear Projects List UI
                const projectList = document.getElementById('submenu-project-list');
                if (projectList) {
                    projectList.innerHTML = '<div style="text-align: center; color: #999; font-size: 12px; padding: 10px;">Sign in to see projects</div>';
                }

                // SECURITY / ISOLATION:
                // If we are on a specific project URL, we must reload to clear state and ensure
                // we aren't showing a user's project to a guest (or vice versa).
                const urlParams = new URLSearchParams(window.location.search);
                if (urlParams.has('project')) {

                    window.history.replaceState({}, document.title, window.location.pathname);
                    window.location.reload();
                }
            }
        });
    }

    // Auth Button Click
    if (authBtn) {
        authBtn.addEventListener('click', async (e) => {
            e.stopPropagation(); // Prevent menu close
            const user = window.firebaseAuth.currentUser;
            if (user) {
                // Sign Out
                try {
                    await window.signOut(window.firebaseAuth);
                } catch (error) {
                    console.error('Sign Out Error', error);
                    alert('Error signing out.');
                }
            } else {
                // Sign In
                try {
                    await window.signInWithPopup(window.firebaseAuth, window.googleProvider);
                } catch (error) {
                    console.error('Sign In Error', error);
                }
            }
        });
    }

    // --- FIRESTORE PROJECT STORAGE ---

    async function loadUserProjects(userId) {
        if (!userId) return;

        const projectList = document.getElementById('submenu-project-list');
        if (!projectList) return;

        projectList.innerHTML = '<div style="text-align: center; color: #999; font-size: 12px; padding:10px;">Loading...</div>';

        try {
            const projectsRef = window.firestore.collection(window.firebaseDb, "users", userId, "projects");
            const q = window.firestore.query(projectsRef); // Get all projects for user
            const querySnapshot = await window.firestore.getDocs(q);

            if (querySnapshot.empty) {
                projectList.innerHTML = '<div style="text-align: center; color: #999; font-size: 12px; padding:10px;">No projects found. Create one!</div>';
                return;
            }

            projectList.innerHTML = '';
            querySnapshot.forEach((doc) => {
                const data = doc.data();
                const pid = doc.id;

                const item = document.createElement('div');
                item.className = 'submenu-project-item';

                // Format date matches preview.html
                const date = new Date(data.lastModified || data.created || Date.now());
                const dateStr = date.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' }) + ' ' + date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

                item.innerHTML = `
                    <span class="submenu-project-name">${data.name || 'Untitled'}</span>
                    <span class="submenu-project-time">${dateStr}</span>
                `;

                // Load Project Click
                item.addEventListener('click', async (e) => {


                    if (window.projectManager) {
                        try {
                            // 1. Ensure Metadata is in ProjectManager (in case it's a new device)
                            let meta = window.projectManager.getProjectMetadata(pid);
                            if (!meta) {

                                // data is the doc.data() from the list loop above
                                const newMeta = {
                                    id: pid,
                                    name: data.name,
                                    created: data.created,
                                    lastModified: data.lastModified
                                };
                                window.projectManager.projects.unshift(newMeta);
                                window.projectManager.saveMetadata();
                            }

                            // 2. Delegate Loading to ProjectManager (Handles Server vs LocalStorage)
                            const success = await window.projectManager.setCurrentProject(pid);

                            if (success) {
                                location.reload();
                            } else {
                                alert('Failed to load project data. It might be missing from both Server and LocalStorage.');
                            }

                        } catch (err) {
                            console.error('Failed to load project:', err);
                            alert('Error loading project: ' + err.message);
                        }
                    }
                });

                projectList.appendChild(item);
            });

        } catch (error) {
            console.error("Error loading projects:", error);
            projectList.innerHTML = '<div style="text-align: center; color: red; font-size: 12px;">Error loading projects</div>';
        }
    }

    // Expose loadUserProjects globally
    window.loadUserProjects = loadUserProjects;

    // Expose handleNewProject
    window.handleNewProject = async () => {
        if (window.projectManager) {
            const newId = await window.projectManager.createProject('New Project', true);
            if (newId) {
                location.reload();
            }
        } else {
            console.error('ProjectManager not available');
        }
    };
});
