content = """# Refactoring Plan: Monolith to Modular Architecture

This document outlines the architectural refactoring plan to break down the legacy monolith into a clean, modular structure. The primary objective is to separate state management from UI rendering and organize code by responsibility and dependency direction rather than blindly reproducing the old file boundaries.

---

## Architectural Goals

- **Modular Boundaries:** Aim for 8–10 reasonably substantial modules rather than dozens of tiny files or a few massive monoliths.
- **Mechanical First, Architectural Second:** Move existing code first, make it work, test it, and _then_ improve the interfaces. This avoids simultaneously refactoring architecture and changing behavior.
- **Boring Entry Point:** Keep `appNG.js` minimal so it serves purely as an orchestrator and entry point.

---

## Target File Structure

```text
static/
├── appNG.js              ← Small bootstrap/orchestrator
│
├── ng-project.js         ← CharacterProject state container
├── ng-project-manager.js ← ProjectManager orchestration
│
├── ng-render.js          ← Shared shell & DOM rendering
├── ng-task-ui.js         ← Footer / tasks
│
├── ng-video.js           ← Video ingestion, playback, & analysis
├── ng-immich.js          ← Immich ingestion, search, & analysis
├── ng-folder.js          ← Folder / zip ingestion
│
├── ng-ring.js            ← Ring construction & rendering
├── ng-selection.js       ← Selection, export, & selection UI
├── ng-pose.js            ← Pose picker & sorting
│
└── ng-generate.js        ← Character-sheet generation
```
