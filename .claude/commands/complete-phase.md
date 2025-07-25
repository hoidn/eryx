# Command: /complete-phase [optional-phase-number] [optional-git-ref] <initiative-path>

**Goal:** Manage the end-of-phase transition using a formal review cycle. This command now supports optional overrides for the phase number and the `git diff` baseline, making it a highly versatile review tool.

**Usage:**
-   **Automatic (Standard):** `/complete-phase plans/active/my-initiative`
    -   *Completes the current phase from `PROJECT_STATUS.md`, diffing against the `Last Phase Commit Hash`.*
-   **Manual Phase:** `/complete-phase 2 plans/active/my-initiative`
    -   *Completes Phase 2, diffing against its default baseline.*
-   **Manual Baseline (Powerful):** `/complete-phase plans/active/my-initiative main`
    -   *Completes the current phase, but generates a diff against the `main` branch.*
-   **Fully Manual:** `/complete-phase 2 plans/active/my-initiative abc123f`
    -   *Completes Phase 2, generating a diff against the specific commit `abc123f`.*

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**You MUST operate in one of two distinct modes. You are not allowed to mix them.**

**Mode 1: Request Review (Default)**
*   **Trigger:** No `review_phase_N.md` file exists for the target phase.
*   **Action:** You MUST parse all arguments, determine the correct diff baseline (from state files or user override), generate a `review_request_phase_N.md` file containing the `git diff`, and then HALT.

**Mode 2: Process Review**
*   **Trigger:** A `review_phase_N.md` file EXISTS for the target phase.
*   **Action:** You MUST read the review, parse the `VERDICT`, and then either commit the changes (on `ACCEPT`) or report the required fixes (on `REJECT`). The `[optional-git-ref]` argument has no effect in this mode.

**DO NOT:**
-   ❌ Commit any code without a `VERDICT: ACCEPT` from a review file.
-   ❌ Generate a new review request if a review file already exists.

---

## 🤖 **CONTEXT: YOU ARE CLAUDE CODE**

You are Claude Code, an autonomous agent. You will execute the Git and file commands below to manage the phase completion and review process. You will handle all steps without human intervention.

---

## 📋 **YOUR EXECUTION WORKFLOW**

### Step 1: Parse Arguments & Determine Mode

This step uses a robust parsing strategy to handle flexible arguments and determine the correct `PHASE_NUMBER` and `DIFF_BASE`.

```bash
# --- Argument Parsing Logic ---
PHASE_NUMBER=""
GIT_REF_OVERRIDE=""
INITIATIVE_PATH=""

# The last argument is always the path.
INITIATIVE_PATH="${@: -1}" 

# Verify the initiative path exists before proceeding.
if [ ! -d "$INITIATIVE_PATH" ]; then
    echo "❌ ERROR: Initiative path '$INITIATIVE_PATH' not found."
    exit 1
fi

# Process optional arguments (phase number and git ref) by looping
# through all arguments except the last one (the path).
for arg in "${@:1:$#-1}"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        PHASE_NUMBER="$arg"
    elif git rev-parse --verify "$arg" >/dev/null 2>&1; then
        GIT_REF_OVERRIDE="$arg"
    else
        echo "⚠️ Warning: Ignoring unrecognized argument '$arg'. It is not a valid phase number or git ref."
    fi
done

# --- Determine Final Phase Number ---
if [ -z "$PHASE_NUMBER" ]; then
    echo "ℹ️ No phase number provided. Auto-detecting from PROJECT_STATUS.md..."
    PHASE_NUMBER=$(grep 'Current Phase:' PROJECT_STATUS.md | sed 's/.*Phase \([0-9]*\).*/\1/')
    if ! [[ "$PHASE_NUMBER" =~ ^[0-9]+$ ]]; then
        echo "❌ ERROR: Could not auto-detect phase number from PROJECT_STATUS.md."
        exit 1
    fi
    echo "✅ Auto-detected current phase as: Phase $PHASE_NUMBER"
else
    echo "✅ Using explicitly provided phase number: $PHASE_NUMBER"
fi

# --- Determine Mode ---
if [ -f "$INITIATIVE_PATH/review_phase_${PHASE_NUMBER}.md" ]; then
    echo "✅ Review file found. Proceeding in 'Process Review' mode."
    # Proceed to Mode 2
else
    echo "ℹ️ No review file found. Proceeding in 'Request Review' mode."
    # Proceed to Mode 1
fi
```

---

### **MODE 1: REQUEST REVIEW**

#### Step 1.1: Determine Diff Baseline and Generate Diff

```bash
# --- Determine Final Diff Baseline ---
DIFF_BASE=""
BASELINE_SOURCE_MSG="" # For logging in the review request

if [ -n "$GIT_REF_OVERRIDE" ]; then
    DIFF_BASE="$GIT_REF_OVERRIDE"
    BASELINE_SOURCE_MSG="Override provided by user: '$GIT_REF_OVERRIDE'"
    echo "✅ Using provided git ref override as diff baseline: $DIFF_BASE"
else
    echo "ℹ️ No git ref override provided. Using default from implementation.md..."
    DIFF_BASE=$(grep 'Last Phase Commit Hash:' "$INITIATIVE_PATH/implementation.md" | awk '{print $4}')
    BASELINE_SOURCE_MSG="Default from implementation.md: '$DIFF_BASE'"
    if [ -z "$DIFF_BASE" ]; then
        echo "❌ ERROR: Could not determine default diff baseline from '$INITIATIVE_PATH/implementation.md'."
        exit 1
    fi
    echo "✅ Using default diff baseline: $DIFF_BASE"
fi

# --- Generate the Diff ---
mkdir -p ./tmp
git diff "${DIFF_BASE}"..HEAD > ./tmp/phase_diff.txt
```

#### Step 1.2: Generate Review Request File & Halt
-   Create a new file: `$INITIATIVE_PATH/review_request_phase_${PHASE_NUMBER}.md`.
-   Populate it using the "REVIEW REQUEST TEMPLATE" below, including the `BASELINE_SOURCE_MSG`.
-   Inform the user that the review request is ready and instruct them to run `/review-phase-gemini` or perform a manual review.
-   **HALT.** Your task for this run is complete.

---

### **MODE 2: PROCESS REVIEW**

*(This mode is unaffected by the `[optional-git-ref]` argument)*

#### Step 2.1: Read and Parse Review File
-   Read the file `$INITIATIVE_PATH/review_phase_${PHASE_NUMBER}.md`.
-   Parse the `VERDICT: [ACCEPT|REJECT]`.

#### Step 2.2: Conditional Execution
-   If `VERDICT: ACCEPT`, execute the `git add`, `git commit`, and state update sequence.
-   If `VERDICT: REJECT`, report the required fixes to the user and halt.

---

## 템플릿 & 가이드라인 (Templates & Guidelines)

### **REVIEW REQUEST TEMPLATE (Revised)**
*This is the content for the agent-generated `review_request_phase_N.md`.*
```markdown
# Review Request: Phase <N> - <Phase Name>

**Initiative:** <Initiative Name>
**Generated:** <YYYY-MM-DD HH:MM:SS>

## Instructions for Reviewer

1.  Analyze the planning documents and the code changes (`git diff`) below.
2.  Create a new file named `review_phase_N.md` in this same directory (`<path>/`).
3.  In your review file, you **MUST** provide a clear verdict on a single line: `VERDICT: ACCEPT` or `VERDICT: REJECT`.
4.  If rejecting, you **MUST** provide a list of specific, actionable fixes under a "Required Fixes" heading.

---
## 1. Planning Documents

### R&D Plan (`plan.md`)
<The full content of plan.md is embedded here>

### Implementation Plan (`implementation.md`)
<The full content of implementation.md is embedded here>

### Phase Checklist (`phase_N_checklist.md`)
<The full content of the current phase_N_checklist.md is embedded here>

---
## 2. Code Changes for This Phase

**This diff shows changes between the specified baseline and the current HEAD.**

**Baseline Used:** <Value of $BASELINE_SOURCE_MSG from logic above>
**Current Branch:** <current feature branch name>

```diff
<The full output of the 'git diff' command is embedded here>
```
```

---

## 📊 **SAMPLE INTERACTION (With Override)**

```
User: /complete-phase plans/active/hotfix-auth-module main

You: "No review file found. Proceeding in 'Request Review' mode.
       Auto-detected current phase as: Phase 3
       Using provided git ref override as diff baseline: main"

     [You execute 'git diff main..HEAD', then generate 'review_request_phase_3.md']

You: "✅ Review request for Phase 3 has been generated at:
       `plans/active/hotfix-auth-module/review_request_phase_3.md`
       
       The diff was generated against the 'main' branch as requested.
       
       Please have it reviewed. Once the review is complete, run `/review-phase-gemini` or create the review file manually, then run this command again to process it."
```
