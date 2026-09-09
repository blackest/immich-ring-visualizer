/**
 * initchar.js
 * Handles the "Birth" sequence: Character Project instantiation and initial canvas setup.
 * Responsibility: Set up the empty state for a new project (sidebar + stage) and handle naming.
 */

(function() {
    "use strict";

    const charInit = {
        // DOM Refs for the a-priori shell
        stageWrap: document.getElementById("ng-stage-wrap"),
        sidebar: document.getElementById("ng-sidebar"),
        leftRailEmpty: document.getElementById("ng-leftrail-empty"),
        leftRailBody: document.getElementById("ng-leftrail-body"),
        mainPlaceholder: document.getElementById("ng-main-placeholder"),

        renderBlankCanvas(project) {
            if (this.mainPlaceholder) {
                this.mainPlaceholder.style.display = "none";
            }
            if (this.stageWrap) this.stageWrap.style.display = "";
            if (this.sidebar) this.sidebar.style.display = "";
            if (this.leftRailEmpty) this.leftRailEmpty.style.display = "none";
            if (this.leftRailBody) this.leftRailBody.style.display = "";

            if (this.mainPlaceholder) {
                this.mainPlaceholder.textContent = `Pick Video, Immich, or Folder / Zip below to get started with "${project.name}".`;
                this.mainPlaceholder.style.display = "";
            }
        },

        wireNamingFlow(project) {
            const tabsEl = document.getElementById("ng-tabs");
            if (!tabsEl) return;

            const projectTab = Array.from(tabsEl.querySelectorAll(".ng-tab"))
                .find(t => t.dataset.id === project.id);

            if (projectTab) {
                const label = projectTab.querySelector(".ng-tab-label");
                if (label) {
                    label.addEventListener("dblclick", (e) => {
                        e.stopPropagation();
                        label.contentEditable = "true";
                        label.focus();
                        document.execCommand("selectAll", false, null);
                    });
                    label.addEventListener("blur", () => {
                        label.contentEditable = "false";
                        if (window.ProjectManager) {
                            window.ProjectManager.renameProject(project.id, label.textContent);
                        }
                    });
                }
            }
        }
    };

    const originalCreate = window.ProjectManager?.createProject;
    if (originalCreate) {
        window.ProjectManager.createProject = function(name) {
            const project = originalCreate.apply(this, arguments);
            const newProj = this.getActive();
            if (newProj) {
                charInit.renderBlankCanvas(newProj);
                charInit.wireNamingFlow(newProj);
            }
        };
    }

    window.initCharNG = charInit;
})();
