# Refactoring Plan: Monolith to Modular Architecture

This document outlines the architectural refactoring plan to break down the legacy monolith into a clean, modular structure. The primary objective is to separate state management from UI rendering and organize code by responsibility and dependency direction rather than blindly reproducing the old file boundaries.

---

## Architectural Goals

* **Modular Boundaries:** Aim for 8–10 reasonably substantial modules rather than dozens of tiny files or a few massive monoliths.
* **Mechanical First, Architectural Second:** Move existing code first, make it work, test it, and *then* improve the interfaces. This avoids simultaneously refactoring architecture and changing behavior.
* **Boring Entry Point:** Keep `appNG.js` minimal so it serves purely as an orchestrator and entry point.

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

---

## Step-by-Step Extraction Order

To minimize the risk of breaking functionality during the refactoring process, execute the extractions in the following sequence:

### 1. Extract `CharacterProject` (`ng-project.js`)
* **Role:** The heart of the new architecture. 
* **Responsibilities:** Owns all project state (Video state, Immich state, ring state, selections, job state, etc.). 
* **Key Rule:** `CharacterProject` owns state, but it **does not** own UI rendering.

### 2. Extract `ProjectManager` (`ng-project-manager.js`)
* **Role:** High-level state and project collection coordination.
* **Responsibilities:** Managing `projects`, `activeId`, `getActive()`, `setTask()`, `switchProject()`, `createProject()`, `deleteProject()`, and top-level `render()`.
* **Design Shift:** Moves away from massive embedded rendering blocks toward clean delegation (e.g., setting a task and calling `this.render()`).

### 3. Extract Rendering (`ng-render.js` & `ng-task-ui.js`)
* **Role:** Centralizing DOM output and UI shell generation.
* **Responsibilities:** Moving layout functions like `renderTabs()`, `renderBottomBar()`, `renderLeftRail()`, `renderMain()`, and `renderStage()` alongside associated DOM helpers.

### 4. Split Input Sources (`ng-video.js`, `ng-immich.js`, `ng-folder.js`)
* **Role:** Isolating data ingestion pathways.
* **Responsibilities:** Handling source-specific ingestion logic while passing references to the active `CharacterProject` rather than relying on hidden global states (e.g., replacing global variables like `lastVideoRingState` with clean properties on the project instance).

### 5. Separate Ring, Selection, and Pose (`ng-ring.js`, `ng-selection.js`, `ng-pose.js`)
* **Role:** Decoupling distinct operational pipeline stages.
* **Responsibilities:** Managing the logical flow from analysis results down to ring construction, selection handling, and pose sorting/viewing.

### 6. Extract Generation (`ng-generate.js`)
* **Role:** Handling final character-sheet generation tasks.
* **Responsibilities:** Acting as the behavioral specification (informed by legacy implementations like `phosphene-sheet.js`) that takes the active `CharacterProject`, runs the generation job, and attaches output to `project.generation` without maintaining its own pile of global state.

### 7. Refine the Orchestrator (`appNG.js`)
* **Role:** Application bootstrap.
* **Responsibilities:** Initializing the DOM, instantiating the `ProjectManager`, wiring modules, restoring projects, and triggering the initial render.
