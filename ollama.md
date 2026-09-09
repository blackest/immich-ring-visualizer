# Immich Ring Visualizer: Ollama Implementation Overview

## Overview

This document provides an overview of the Immich Ring Visualizer project, comparing the older implementation with the new Next Generation (NG) version, particularly focusing on the character generation functionality.

## Older Implementation (Phosphene)

### Key Features

- Character sheet generation using HiDream engine
- HTTP proxy to separate Phosphene process (127.0.0.1:8198)
- Blocking HTTP requests for generation duration
- Single-threaded job processing with blocking behavior
- URL routes under `/api/phosphene/...`

### Core Components

- `routes/phosphene.py` - HTTP layer for character generation
- `character_sheet.py` - Character sheet management
- `hidream_engine.py` - HiDream engine integration
- `sheet_jobs.py` - Background job management
- `shot_presets.py` - Preset configurations

### Key Endpoints

- `/api/phosphene/status` - Health check for HiDream lab
- `/api/phosphene/presets` - Available shot presets and styles
- `/api/phosphene/characters` - Create draft characters
- `/api/phosphene/characters/<character_id>/sheet/generate` - Generate character sheet
- `/api/phosphene/sheet-jobs/<job_id>` - Poll for job status

### Characteristics

- **Blocking behavior**: Long-running generation jobs block the HTTP connection
- **Process separation**: Uses separate Phosphene process for generation
- **Synchronous requests**: Each request waits for completion before returning
- **Limited concurrency**: Only one job can be actively processed at a time

## Next Generation (NG) Implementation

### Key Features

- Non-blocking HTTP requests
- Background job processing with FIFO queue
- Multiple concurrent jobs without blocking
- URL routes under `/api/ng/...` for new functionality
- Enhanced character sheet management with improved architecture

### Core Components

- `routes/generateNG.py` - HTTP layer for character generation (NG)
- `character_sheetNG.py` - Enhanced character sheet management (NG)
- `hidream_engineNG.py` - Enhanced HiDream engine integration (NG)
- `sheet_jobsNG.py` - Enhanced background job management (NG)
- `shot_presetsNG.py` - Enhanced preset configurations (NG)

### Key Endpoints

- `/api/ng/generate/status` - Health check for HiDream lab
- `/api/ng/generate/presets` - Available shot presets and styles
- `/api/ng/generate/characters` - Create draft characters
- `/api/ng/generate/sheet-from-asset` - Generate sheet from Immich asset
- `/api/ng/generate/sheet-from-upload` - Generate sheet from local file
- `/api/ng/generate/characters/<character_id>/sheet/generate` - Generate character sheet
- `/api/ng/generate/sheet-jobs/<job_id>` - Poll for job status

### Characteristics

- **Non-blocking behavior**: Requests return immediately with job ID
- **Concurrent processing**: Multiple jobs can run simultaneously
- **Queue management**: FIFO queue prevents busy errors for concurrent requests
- **Enhanced features**: Better error handling and state management
- **Improved architecture**: More modular and maintainable structure

## Key Differences Between Versions

### Architecture

| Aspect           | Old (Phosphene)            | NG                      |
| ---------------- | -------------------------- | ----------------------- |
| Process Model    | Separate Phosphene process | In-process execution    |
| Request Handling | Blocking                   | Non-blocking            |
| Job Management   | Synchronous                | Asynchronous with queue |
| Concurrency      | Limited                    | Enhanced                |
| Error Handling   | Basic                      | Improved                |

### Functionality

| Feature             | Old                                | NG                                  |
| ------------------- | ---------------------------------- | ----------------------------------- |
| Character Creation  | Basic draft character registration | Enhanced with better error handling |
| Generation Jobs     | Blocking requests                  | Non-blocking background jobs        |
| Concurrent Requests | Limited                            | Full support                        |
| Job Queueing        | Not supported                      | FIFO queue                          |
| State Management    | Basic                              | Enhanced with better tracking       |

### File Structure

- **Old**: `routes/phosphene.py`, `character_sheet.py`, `hidream_engine.py`
- **NG**: `routes/generateNG.py`, `character_sheetNG.py`, `hidream_engineNG.py`

## Migration Strategy

To port functionality from the older version to the newer version:

1. **Character Generation Routes**:
   - Use `generateNG_bp` blueprint instead of `phosphene_bp`
   - Migrate endpoints like `/api/phosphene/characters` to `/api/ng/generate/characters`

2. **Job Management**:
   - Use `sheet_jobsNG.py` instead of `sheet_jobs.py`
   - Implement background job processing with proper queue management

3. **Data Handling**:
   - Use `character_sheetNG.py` for enhanced character sheet management
   - Implement proper error handling for file operations

4. **Configuration**:
   - Use `configNG.py` instead of `config.py` for the NG version
   - Ensure proper database connections and connection pooling

## Implementation Notes

The NG version maintains compatibility with existing functionality while providing enhanced features:

1. **Backward Compatibility**: The original routes are preserved for existing integrations
2. **Enhanced Performance**: Non-blocking requests and concurrent job processing
3. **Better Error Handling**: More granular error responses and state management
4. **Improved Scalability**: Support for multiple concurrent generation jobs

This approach allows for gradual migration of functionality without breaking existing implementations.
