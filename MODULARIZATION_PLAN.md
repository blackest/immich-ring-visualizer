# Modularization Plan for `appNG.js`

## Goal

Reduce the size and complexity of the `appNG.js` monolith by splitting it into concern-based modules. The primary objective is to move from a single large file to a modular architecture where state is centralized and feature implementations are decoupled from the main orchestrator.

## Proposed Module Structure

| File                    | Responsibility                                                                                                                       |
| :---------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| `appNG.js`              | **Bootstrap & Orchestrator**: Entry point, DOM initialization, wiring modules, and initial project restoration.                      |
| `ng-project.js`         | **Core State**: Definition of the `CharacterProject` class. Owns all project-specific state (video, Immich, ring, selections, jobs). |
| `ng-project-manager.js` | **Orchestration**: Management of multiple projects, active project tracking, and high-level task switching.                          |
| `ng-render.js`          | **Shell Rendering**: Logic for rendering the shared application shell (tabs, bottom bar, left rail, main stage).                     |
| `ng-task-ui.js`         | **Task Interface**: Footer and task-bar UI logic.                                                                                    |
| `ng-video.js`           | **Video Pipeline**: Ingestion, playback, and analysis of video files.                                                                |
| `ng-immich.js`          | **Immich Integration**: Immich ingestion, search, and analysis.                                                                      |
| `ng-folder.js`          | **Local File Ingestion**: Folder and ZIP file loading and analysis.                                                                  |
| `ng-ring.js`            | **Ring Logic**: Ring construction and visual rendering.                                                                              |
| `ng-selection.js`       | **Curation UI**: Selection management and export pipeline.                                                                           |
| `ng-pose.js`            | **Pose Analysis**: Pose picker and sorting logic.                                                                                    |
| `ng-generate.js`        | **Generation**: Character sheet generation workflow.                                                                                 |

## Execution Strategy (Phased Approach)

To avoid breaking the application, the extraction will follow a "mechanical first, architectural second" approach: **Move existing code $\rightarrow$ make it work $\rightarrow$ test $\rightarrow$ improve interfaces.**

### Phase 1: Core State Extraction

**Target:** `ng-project.js`

- Extract the `CharacterProject` class and all its associated state.
- **Principle:** `CharacterProject` owns the state; it does not own the UI rendering.

### Phase 2: Project Management Extraction

**Target:** `ng-project-manager.js`

- Extract `ProjectManager` logic: `projects` list, `activeId`, `switchProject()`, `createProject()`, etc.
- Transition `ProjectManager` toward pure orchestration rather than containing rendering logic.

### Phase 3: Rendering Logic Extraction

**Target:** `ng-render.js`

- Move all top-level rendering functions: `renderTabs()`, `renderBottomBar()`, `renderLeftRail()`, `renderMain()`, `renderStage()`.
- Establish a clear flow: `ProjectManager` $\rightarrow$ `ng-render.js`.

### Phase 4: Input Source Separation

**Targets:** `ng-video.js`, `ng-immich.js`, `ng-folder.js`

- Separate ingestion logic by source.
- Replace hidden globals (e.g., `lastVideoRingState`) with references to the active `CharacterProject` instance.

### Phase 5: Ring and Selection Decoupling

**Targets:** `ng-ring.js`, `ng-selection.js`, `ng-pose.js`

- Separate the distinct operations of analysis results $\rightarrow$ ring $\rightarrow$ selection $\rightarrow$ pose analysis.

### Phase 6: Generation Workflow Extraction

**Target:** `ng-generate.js`

- Extract the character sheet generation logic.
- Ensure `ng-generate.js` operates on the active `CharacterProject` rather than maintaining its own global state.

## Core Architecture Principles

1. **Centralized State**: All durable and session state for a character lives within the `CharacterProject` instance.
2. **Boring Entry Point**: `appNG.js` should eventually serve only as the wiring and bootstrap layer.
3. **Avoid Over-Fragmentation**: Aim for 8–10 substantial modules rather than dozens of tiny files.
4. **Dependency Direction**: Feature modules depend on the `CharacterProject` state, but the state does not depend on the feature modules.
