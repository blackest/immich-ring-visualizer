/**
 * initguing.js
 * Handles the "Void" state: no active character project.
 * Responsibility: Render empty shell and listen for project creation.
 */

(function() {
    "use strict";

    const gui = {
        mainPlaceholder: document.getElementById("ng-main-placeholder"),
        stageWrap: document.getElementById("ng-stage-wrap"),
        sidebar: document.getElementById("ng-sidebar"),
        leftRailEmpty: document.getElementById("ng-leftrail-empty"),
        leftRailBody: document.getElementById("ng-leftrail-body"),
        newProjectBtn: document.getElementById("ng-new-project"),
        taskButtons: Array.from(document.querySelectorAll(".ng-task-btn")),
        
        // The "Void" render
        renderVoid() {
            if (this.mainPlaceholder) {
                this.mainPlaceholder.style.display = "";
                this.mainPlaceholder.textContent = "No project open. Press + to start one.";
            }
            if (this.stageWrap) this.stageWrap.style.display = "none";
            if (this.sidebar) this.sidebar.style.display = "none";
            if (this.leftRailEmpty) this.leftRailEmpty.style.display = "block";
            if (this.leftRailBody) this.leftRailBody.style.display = "none";
            
            // Disable task buttons when no character exists
            this.taskButtons.forEach(btn => {
                btn.disabled = true;
                btn.classList.remove("ng-task-active");
            });
        }
    };

    // Wire the "+" button to the ProjectManager handover
    if (gui.newProjectBtn) {
        gui.newProjectBtn.addEventListener("click", () => {
            if (window.ProjectManager && typeof window.ProjectManager.createProject === "function") {
                window.ProjectManager.createProject();
            } else {
                console.error("ProjectManager not found in global scope.");
            }
        });
    }

    // Expose to window so ProjectManager can trigger Void state if all projects are closed
    window.initGuiNG = gui;

    // Initial call to ensure the void is rendered on load -- but only if
    // ProjectManager (which runs its own loadState()+render() earlier in
    // the script load order, in bootstrapWiringNG.js) didn't already
    // restore a project from localStorage. Calling this unconditionally
    // stomped the restored project's render on every page refresh.
    if (!window.ProjectManager || !window.ProjectManager.projects || window.ProjectManager.projects.length === 0) {
        gui.renderVoid();
    }
})();
