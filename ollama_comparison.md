# Ollama vs NG Implementation Comparison

This document compares the older immich-ring-visualizer project with the newer NG (Next Generation) version, focusing on the character system implementation changes and backend wiring.

## Overview

The NG version introduces a more robust architecture with non-blocking HTTP requests, background job processing, and improved queue management. The core change involves moving from a simple implementation to a class-based system for character handling.

## Character System Implementation

### Old Implementation (immich-ring-visualizer)

In the older version:

- Character system was implemented with a simpler approach
- Direct function calls for character generation
- No background job processing
- Synchronous execution of generation tasks

### NG Implementation

In the NG version:

- Character system is implemented with class-based instances
- Uses a background job processing system with FIFO queue management
- Non-blocking HTTP requests with immediate response and polling
- Proper serialization of GPU usage through background worker threads

## Architecture Differences

### Old Architecture

- Direct function calls for generation
- Synchronous processing
- No queue management
- Blocking HTTP requests

### NG Architecture

- Uses `sheet_jobsNG.py` for job queue management
- Implements background worker threads for processing
- Non-blocking HTTP requests with immediate response
- FIFO queue with job status polling
- Proper locking mechanisms to prevent concurrent renders

## Key Features of NG Implementation

### 1. Background Job Processing

```python
# In sheet_jobsNG.py
def start_job_ng(character_id, shot_keys, trigger):
    # Validates inputs and enqueues job
    # Uses queue.Queue for thread-safe job queuing
    # Starts worker thread if needed
```

### 2. Job Status Polling

```python
# In generateNG.py
@app.route('/api/ng/generate/sheet-jobs/<job_id>')
def sheet_job_status_route_ng(job_id):
    # Returns job status including per-shot status
    # Allows client to poll for completion
```

### 3. Non-blocking Requests

```python
# In generateNG.py
@app.route('/api/ng/generate/characters/<character_id>/sheet/generate', methods=['POST'])
def generate_character_sheet_route_ng(character_id):
    # Immediately returns job info without blocking
    # Job is queued for background processing
```

## Backend Wiring

The NG backend is fully wired with:

- Proper route blueprints in `generateNG.py`
- Background job management in `sheet_jobsNG.py`
- FIFO queue system with proper locking mechanisms
- Thread-safe job processing
- Job status tracking and polling endpoints

## Implementation Details

### Queue Management

- Uses Python's `queue.Queue` for thread-safe job queuing
- Maintains job status in memory (`_JOBS` dictionary)
- Implements FIFO ordering with `_MAX_JOBS_KEPT = 20` (ring buffer)
- Single background worker thread processes jobs one at a time

### Worker Thread

- `_worker_loop()` continuously processes jobs from the queue
- Each job is executed by calling its target function
- Proper error handling with exception catching and error storage
- Concurrent job execution is prevented through locking

### Job Lifecycle

1. Jobs start with status "queued"
2. Worker thread transitions to "dispatched" and "rendering"
3. Jobs complete with "completed" or "failed" status
4. Job status includes per-shot status inferred from disk

## Benefits of NG Implementation

1. **Non-blocking Requests**: HTTP responses are immediate, allowing for better user experience
2. **Background Processing**: Heavy computation happens in background without blocking the server
3. **Queue Management**: Proper handling of concurrent requests with FIFO ordering
4. **Thread Safety**: Concurrent job processing is prevented through locking mechanisms
5. **Scalability**: Can handle multiple concurrent requests efficiently
6. **Status Tracking**: Real-time job status monitoring through polling endpoints

## Migration Notes

When porting from older to NG versions:

1. Routes will need to be updated to use new endpoints
2. Background job processing requires proper queue management
3. Client-side polling logic needs to be implemented
4. Character system now uses class-based instances instead of simple functions
5. Error handling and status tracking are more robust in NG version
