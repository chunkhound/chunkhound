# Embedding Failure Handling Plan

## Overview
This plan addresses the infinite loop issue in `generate_missing_embeddings` where failed chunks are repeatedly retrieved and processed, preventing completion. The solution implements comprehensive error handling with proper classification, retry logic, and progress tracking.

**Note**: Backwards compatibility is not required - this is a breaking change that fixes a critical bug.

## Requirements
- **Permanent failures**: Chunks that fail permanently (e.g., oversized) should be excluded from future processing for the same provider/model combination
- **Transient failures**: Network errors, temporary service unavailability should trigger retries with backoff, but abort after threshold
- **Provider/Model changes**: When switching providers or models, previous embeddings become invalid
- **Progress tracking**: Only count successfully processed chunks
- **Error reporting**: Accumulate samples of each error type and report to logs
- **Testing**: Include test that simulates errors to verify flow works correctly

## Implementation Steps

### 1. Database Schema Enhancement
**Goal**: Add embedding status tracking that handles provider/model invalidation

**Changes**:
- Add `embedding_status` field to chunks table with values: `'pending'`, `'success'`, `'failed'`, `'permanent_failure'`
- existing `embedding_signature` field is used to track which provider/model combo was used
- Modify `get_chunks_without_embeddings_paginated` to:
  - Exclude chunks with `permanent_failure` status for current provider/model
  - Include chunks with `success` status if provider/model changed (invalidate old embeddings)
  - Prioritize chunks that haven't been attempted yet
- Update LanceDB query to use `embedding_signature` (no fallback needed as it works reliably):
  - Current: `(embedding_signature IS NULL OR embedding_signature != '{target_sig}')`
  - New: `(embedding_signature IS NULL OR embedding_signature != '{target_sig}') AND embedding_status != 'permanent_failure'`
- Ensure indexes on new field (`embedding_status`) for query performance

**Database Migration**:
- Add new columns with appropriate defaults
- Migrate existing chunks to have `embedding_status = 'success'` if they have embeddings
- Handle both DuckDB and LanceDB providers

### 2. Enhanced Error Classification
**Goal**: Categorize errors as recoverable vs permanent

**Implementation**:
- Create `EmbeddingError` enum/classification system:
  - `PERMANENT`: Oversized chunks, invalid content, unsupported encoding
  - `TRANSIENT`: Network timeouts, rate limits, temporary service unavailability
  - `BATCH_RECOVERABLE`: Token limits, partial batch failures

- Modify `_generate_embeddings_in_batches` to classify exceptions
- Add error counters and sample collection for logging

### 3. Batch-Level Retry with Granular Failure Handling
**Goal**: Handle mixed success/failure batches and implement retry logic

**Changes**:
- Modify batch processing to track per-chunk success/failure
- For partial batch failures:
  - Successfully embedded chunks: mark as `success`
  - Failed chunks: classify error and handle appropriately
- Implement retry queue for transient failures with exponential backoff
- Add retry attempt limits per chunk (max 3 attempts for transient errors)
- For permanent failures: mark chunks immediately and exclude from future processing

### 4. Improved Progress Tracking
**Goal**: Only count successfully processed chunks in progress

**Changes**:
- Modify progress advancement to only occur when embeddings are actually inserted
- Track separate counters for: attempted, successful, failed, permanent_failed
- Update progress bar to show accurate completion percentage
- Add detailed progress logging with error statistics

### 5. Error Reporting and Logging
**Goal**: Provide visibility into failure patterns

**Implementation**:
- Add configurable error sample limit (default: 5 samples per error type)
- Report errors at the end of each embedding batch
- After reaching sample limit, summarize remaining errors by type and count
- Example format: "embedding batch processed x/y chunks. success: [a] failed [b] failure reason breakdown: 1. error type 1: [c]. error type 2: [d]"
- Include error counts and examples in return statistics
- Reset error tracking between executions

### 6. Flow Control for Transient Errors
**Goal**: Handle sequences of transient errors appropriately

**Implementation**:
- Add configuration for transient error thresholds:
  - `max_consecutive_transient_failures` (default: 5)
  - `transient_error_window_seconds` (default: 300)
- Track consecutive transient failure count within time window
- If consecutive failures exceed threshold, abort the flow
- If a retry succeeds, reset the consecutive failure counter
- Log warnings when approaching failure threshold

### 7. Update Database Provider Interface
**Goal**: Support status tracking across providers

**Changes**:
- Add method to `DatabaseProvider` interface:
  - `invalidate_embeddings_by_provider_model(current_provider, current_model)` - removes all embeddings except those matching the current provider/model combination

- Implement in both DuckDB and LanceDB providers
- Use this method at the start of embedding flow to clean up old embeddings
- Call `optimize_tables()` after the invalidation to reclaim space
- Ensure atomic status updates with embedding insertions during batch operations

### 8. Modify Embedding Service Logic
**Goal**: Integrate error handling into the main flow

**Changes**:
- Update `generate_missing_embeddings` to handle status tracking
- Modify `_generate_embeddings_in_batches` to return detailed results
- Add error classification and handling logic
- Update progress tracking integration

### 9. Create Database Cleanup Script
**Goal**: Provide manual script to clear old embeddings before using new flow

**Implementation**:
- Create script in `scripts/` directory: `clear_legacy_embeddings.py`
- Script removes all embeddings from database
- User runs this manually before testing new flow
- Include safety checks and confirmation prompts

### 10. Update Tests
**Goal**: Create comprehensive error simulation test first, then update existing tests

**Changes**:
- **FIRST**: Create `test_embedding_flow_with_simulated_failures` test that reproduces current bug
  - Mock provider to simulate both permanent and transient error types:
    - **Permanent failures**: Oversized chunks, invalid content, unsupported encoding
    - **Transient failures**: Network timeouts, rate limits, temporary service unavailability
  - Verify current behavior: infinite loop on permanent failures, improper handling of transient failures
  - This test should FAIL initially, demonstrating the bug
- **THEN**: After implementing the design, update existing embedding tests to expect new status fields
- **THEN**: Update end-to-end tests to handle new flow behavior
- **FINALLY**: Verify the error simulation test now PASSES with the new implementation

## Execution Order
1. **PRIORITY 1: Create error simulation test** (part of step 10) - this should reproduce the current infinite loop bug
2. Execute implementation steps 1-9 sequentially
3. **Run the error simulation test** - it should now pass with the new implementation
4. Update remaining existing tests to work with new behavior

## Success Criteria
- **Error simulation test initially fails** (reproduces current infinite loop bug with both permanent and transient failures)
- **After implementation, error simulation test passes** (bug is fixed for both error types)
- No infinite loops when chunks fail permanently
- Transient errors trigger appropriate retries with exponential backoff and abort threshold
- Progress tracking accurately reflects successful processing only
- Provider/model changes properly invalidate old embeddings
- Error reporting provides useful debugging information with error samples for all error types
- All existing tests pass with new behavior
- Database cleanup script works correctly

## Implementation Status

### ✅ COMPLETED: All Steps Successfully Implemented

#### Step-by-Step Implementation Summary
1. **Database Schema Enhancement**: Added `embedding_status` field to chunks table with statuses 'pending', 'success', 'failed', 'permanent_failure'. Updated `get_chunks_without_embeddings_paginated` to exclude permanent failures and handle provider/model changes. Implemented in both DuckDB and LanceDB providers with appropriate indexes and migrations.

2. **Enhanced Error Classification**: Created `EmbeddingError` classification system in `chunkhound/core/exceptions/embedding_error_classification.py` with categories PERMANENT, TRANSIENT, and BATCH_RECOVERABLE. Integrated into batch processing for exception classification.

3. **Batch-Level Retry with Granular Failure Handling**: Modified batch processing to track per-chunk outcomes. Implemented retry logic with exponential backoff for transient failures (max 3 attempts), and immediate marking for permanent failures.

4. **Improved Progress Tracking**: Updated progress to advance only on successful embeddings. Added detailed counters and logging for attempted, successful, failed, and permanent failed chunks.

5. **Error Reporting and Logging**: Implemented error sample collection (5 per type) with detailed reporting at batch end, including counts and examples.

6. **Flow Control for Transient Errors**: Added thresholds for consecutive transient failures (5 max in 300s window), with abort logic and counter reset on success.

7. **Update Database Provider Interface**: Added `invalidate_embeddings_by_provider_model` method to interface, implemented in both providers. Called at flow start with table optimization.

8. **Modify Embedding Service Logic**: Updated `generate_missing_embeddings` and batch methods to integrate status tracking, error handling, and progress updates.

9. **Create Database Cleanup Script**: Developed `scripts/clear_legacy_embeddings.py` for manual embedding removal with safety prompts.

10. **Update Tests**: Created `test_embedding_flow_with_simulated_failures` to simulate and verify error handling. Updated existing tests for new status fields and behavior.

#### Key Deviations from Original Plan
No significant deviations from the original plan. All steps were implemented as specified.

#### Critical Bug Fix: Embedding Status Determination
**Issue Identified**: The `embedding_status` field was being hardcoded to "success" in the database providers' `insert_embeddings_batch` method, rather than being determined by the embedding process logic. This violated separation of concerns and prevented proper status tracking.

**Fix Implemented**:
- **Modified Embedding Service**: Updated `_generate_embeddings_in_batches` to include `"status": "success"` in embeddings_data for successful chunks, removing redundant `update_chunk_status("success")` calls
- **Updated Database Providers**:
  - **DuckDB Provider**: Modified `_executor_insert_embeddings_batch` to require and use status from embeddings_data with validation
  - **LanceDB Provider**: Updated to use `emb_data.get("status")` instead of hardcoded "success"
- **Maintained Performance**: Failed chunks still get status updates via separate `update_chunk_status` calls (correct, as they don't go through batch insert), while successful chunks get status set efficiently through batch operations

#### Additional Features Implemented
- Enhanced error classification with support for additional provider-specific error types for better granularity.
- Improved logging with detailed error statistics and progress reporting.

#### Testing & Validation Results
The error simulation test (`test_embedding_flow_with_simulated_failures`) initially failed, reproducing the infinite loop bug with permanent and transient failures. After implementation, the test passes, confirming the fix. All existing embedding-related tests have been updated to work with the new status fields and flow behavior, and they pass successfully.

**Specific Status Fix Validation**:
- ✅ `test_embedding_flow_with_simulated_failures` **PASSES** - confirms embedding status is properly determined by embedding process
- ✅ All embedding tests pass - no regression in functionality
- ✅ Error classification tests pass - status handling works correctly
- ✅ Code imports successfully - no syntax or import issues

#### Production Readiness
The embedding failure handling solution is fully implemented, thoroughly tested, and ready for production deployment. The fix eliminates the infinite loop issue and provides robust error handling for both permanent and transient failures, ensuring reliable embedding generation.