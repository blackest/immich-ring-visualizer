(function () {
  "use strict";

  // This is the active runtime bootstrap.
  // appNG.js remains a legacy stub; bootstrapNG.js is the real app entry.
  window.AppNG = window.AppNG || {};

  window.AppNG.bootstrap = window.AppNG.bootstrap || {
    started: false,
    start() {
      if (this.started) return;
      this.started = true;

      if (window.AppNG.project && window.AppNG.project.ProjectManager) {
        window.AppNG.manager = new window.AppNG.project.ProjectManager();
        window.AppNG.projectManager = window.AppNG.manager;
        window.ProjectManager = window.AppNG.manager;

        const existing = window.AppNG.manager.loadState();

        if (!existing.length) {
          const defaultProject = new window.AppNG.project.CharacterProject(
            "project-1",
            "Character 1",
          );
          window.AppNG.manager.addProject(defaultProject);
        } else {
          existing.forEach((project) =>
            window.AppNG.manager.addProject(project),
          );
        }

        if (
          window.AppNG.render &&
          typeof window.AppNG.render.wireNGGlobalControls === "function"
        ) {
          window.AppNG.render.wireNGGlobalControls();
        }

        if (
          window.AppNG.render &&
          typeof window.AppNG.render.wireAppShellNG === "function"
        ) {
          window.AppNG.render.wireAppShellNG();
        }

        if (
          window.AppNG.export &&
          typeof window.AppNG.export.wireExportSettingsNG === "function"
        ) {
          window.AppNG.export.wireExportSettingsNG();
        }

        window.AppNG.manager.render();
      }
    },
  };

  window.AppNG.bootstrap.start();
})();
