### **Prompt: Generating a High-Quality Documentation Plan**

**Your Role:** You are an expert Staff Software Engineer and AI Agent Orchestrator. Your current task is to analyze a codebase and create a detailed, self-contained implementation checklist for a comprehensive documentation initiative. Your primary deliverable is the **checklist itself**, not the documentation.

**User's Goal:**
> "I want every `.py` module (not counting scripts) to have a docstring <10 percent of the module size that documents the module's public interface: i.e., gives sufficient information on how that module is used / is to be used in other parts of the code or as a public api."

Your task is to break this high-level goal into a robust, executable plan that another AI agent (or a team of developers) can follow to produce high-quality, consistent documentation.

---

### **Your Mandatory Workflow**

You must follow this three-step process to generate the final implementation checklist.

#### **Step 1: Codebase Analysis & Dependency Mapping (The "Why")**

Before you can plan, you must understand. You need to determine the codebase's architecture, identify its core components, and map out their interdependencies. This is the most critical step.

1.  **Define the Scope:**
    *   First, clearly define what constitutes a "module" that needs documentation versus a "script" that does not. Create a precise rule (e.g., "all `.py` files in the `src/` or `lib/` directory, excluding `__init__.py`").
    *   Use a `find` command to generate a definitive list of all files that fall within this scope. This list will become the basis of your project plan.

2.  **Map the Dependencies:**
    *   You must determine how these modules connect. The goal is to understand, for each module, "who uses it?" and "what does it use?". This defines its public interface and its role.
    *   **Primary Method (Static Analysis):** Use a tool like `pydeps` to generate a complete dependency graph of the entire library. This provides a high-level, visual overview of the architecture and a detailed, machine-readable report of every import relationship.
        ```bash
        # Example commands for pydeps
        pip install pydeps
        pydeps <library_name> --cluster -o dependency_graph.svg
        pydeps <library_name> --no-output --show-deps > dependency_report.txt
        ```
    *   **Secondary Method (Targeted Inspection):** Use command-line tools like `grep` or `rg` (ripgrep) to perform targeted lookups. This is essential for quickly verifying the consumers of a specific function or class.
        ```bash
        # Example: Find all modules that import the 'loader' module
        grep -r "from my_lib import loader" my_lib/
        ```
    *   **Synthesize Findings:** After your analysis, you should be able to answer these questions for any given module:
        *   What is its primary purpose? (e.g., "Data transformation," "Physics simulation," "Core utilities").
        *   What are its key public functions/classes? (i.e., those imported by other modules).
        *   What are its primary consumers? (i.e., the most important modules that depend on it).

#### **Step 2: Plan the Execution Strategy (The "How")**

Now that you understand the codebase, you must design a plan to document it efficiently and consistently.

1.  **Adopt a Sub-Agent Strategy:** For a task like this (documenting N independent modules), the best approach is to break it down into parallelizable sub-tasks. Your implementation plan should be structured around a main "Orchestrator Agent" that manages specialized "Authoring Sub-Agents."

2.  **Design the Checklist Structure:** Your checklist should be a phased plan.
    *   **Phase 1: Analysis & Scoping:** This phase codifies the dependency mapping work you just completed into executable steps. It creates the foundational artifacts for the project (e.g., the list of modules to document, the dependency reports).
    *   **Phase 2: Implementation (Orchestration):** This phase details the process of spawning a sub-agent for each module. The core of this phase is the set of instructions you will provide to each sub-agent.
    *   **Phase 3: Verification & Consistency:** This is the crucial quality assurance step. It must include automated checks for completeness, a final "peer review" pass to ensure consistency, and automated style linting.

3.  **Draft the Sub-Agent Instructions:** This is the most important part of your plan. You must create a detailed prompt that will be given to each "Authoring Sub-Agent." This prompt must be so clear and prescriptive that it guarantees a consistent, high-quality output, regardless of which agent executes it. It must force the sub-agent to:
    *   Analyze the provided dependency context.
    *   Focus on the public interface and the module's role.
    *   Explain the *effect* of key parameters.
    *   Provide a realistic, multi-step usage example.
    *   Adhere to all constraints (e.g., the 10% size limit).

#### **Step 3: Draft the Final Implementation Checklist**

Finally, synthesize all of the above into a single, self-contained markdown file. This file is your final deliverable.

*   Use the **"Example Agent Implementation Checklist"** provided below as a template.
*   The checklist must be clear, detailed, and contain all the necessary commands and instructions for another agent to execute the entire documentation initiative from start to finish.
*   Your instructions to the sub-agents, including the hardened docstring template, must be embedded directly within the main checklist.

---

### **Example Agent Implementation Checklist (Your Final Output)**

*(This is the high-quality checklist you should produce. It is included here as a concrete example of the expected output.)*

# Agent Implementation Checklist: Module Docstring Initiative

**Initiative:** Comprehensive Module Docstring Generation
**Created:** <Date>
**Phase Goal:** To add a high-quality, concise, public-interface-focused docstring to every non-script `.py` module in the `<library_name>/` library, ensuring each docstring is less than 10% of the module's size.
**Deliverable:** A fully documented `<library_name>/` library with consistent, useful module-level docstrings and a passing `pydocstyle` verification check.

## ✅ Task List

### Instructions for the Main Agent:
1.  Work through the phases in order. Do not proceed to the next phase until the previous one is fully complete.
2.  For **Phase 2**, you will act as an orchestrator. For each module listed, you will spawn a dedicated sub-agent with the specific instructions provided.
3.  Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

---

| ID | Task Description | State | How/Why & API Guidance |
| :-- | :--- | :--- | :--- |
| **Phase 1: Analysis & Scoping**
| 1.A | **Generate Definitive List of Target Modules** | `[ ]` | **Why:** To create a master list of all modules that require a docstring. <br> **How:** Execute the following command from the project root and save the output. <br> **Command:** <br> `find <library_name> -name "*.py" -not -name "__init__.py" > modules_to_document.txt` <br> **Verify:** The file `modules_to_document.txt` should exist and contain a list of `.py` files. |
| 1.B | **Generate Static Dependency Map** | `[ ]` | **Why:** To provide the necessary context for all sub-agents to understand each module's public interface and role. <br> **How:** Install `pydeps` (`pip install pydeps`) and run the following commands to generate both visual and text-based dependency reports. <br> **Commands:** <br> `pydeps <library_name> --cluster -o <library_name>/dependency_graph.svg` <br> `pydeps <library_name> --no-output --show-deps > <library_name>/dependency_report.txt` <br> **Verify:** The files `dependency_graph.svg` and `dependency_report.txt` exist in the `<library_name>/` directory. |
| 1.C | **Create a Progress Tracking Checklist** | `[ ]` | **Why:** To track the completion status of each sub-agent's task. <br> **How:** Create a new markdown file named `docstring_progress.md`. Copy the contents of `modules_to_document.txt` into it and format it as a checklist. <br> **Example:** <br> `- [ ] <library_name>/config/config.py` <br> `- [ ] <library_name>/params.py` <br> `...` |
| **Phase 2: Sub-Agent Orchestration for Docstring Implementation**
| 2.A | **Spawn Sub-Agents for Each Module** | `[ ]` | **Why:** To process each module independently and in parallel if possible. <br> **How:** For each file path listed in `modules_to_document.txt`, invoke a sub-agent with the specific "Sub-Agent Instructions" provided below. Pass the module's file path and the paths to the dependency reports (`<library_name>/dependency_report.txt` and `<library_name>/dependency_graph.svg`) as context. As each sub-agent completes its task, update the `docstring_progress.md` checklist. |
| **Phase 3: Final Verification & Consistency Pass**
| 3.A | **Verify All Modules are Documented** | `[ ]` | **Why:** To ensure no modules were missed. <br> **How:** Write a script that reads `modules_to_document.txt` and checks that each file now starts with a `"""` docstring. The script should fail if any module is undocumented. |
| 3.B | **Run Cross-Reference and Consistency Check** | `[ ]` | **Why:** To ensure the docstrings are not just present, but are consistent and reference each other correctly. <br> **How:** Invoke a final "Verification Sub-Agent" with the instructions below. This agent's task is to read *all* the new docstrings and the dependency map to ensure they form a coherent whole. |
| 3.C | **Run Automated Docstring Style Linting** | `[ ]` | **Why:** To enforce a consistent documentation style across the entire project. <br> **How:** Install `pydocstyle` (`pip install pydocstyle`) and run it on the `<library_name>` directory. <br> **Command:** <br> `pydocstyle <library_name>/` <br> **Verify:** The command should report no errors, or only minor, acceptable warnings. |
| 3.D | **Final Code Commit** | `[ ]` | **Why:** To save the completed documentation work to the repository. <br> **How:** Stage all the modified Python files and commit them. <br> **Command:** <br> `git add <library_name>/**/*.py` <br> `git commit -m "docs: Add comprehensive module-level docstrings\n\n- Documents the public interface for all core library modules.\n- Follows a consistent format with usage examples.\n- Docstring size is constrained to <10% of module size."` |

---

### **Sub-Agent Instructions: Docstring Authoring (Revised & Hardened)**

**Your Goal:** Write a single, high-quality, developer-focused module-level docstring for the specified Python module.

**Your Context:**
*   **Target Module:** `<path/to/module.py>`
*   **Dependency Report:** `<library_name>/dependency_report.txt`

**Your Guiding Principles:**
1.  **Go Beyond Listing:** Do not just list the functions. Explain the module's *purpose* and its *role* in the overall architecture. Answer the question: "Why does this module exist?"
2.  **Explain the "Why," Not Just the "What":** For each public function, explain the *effect* of its key parameters, not just their names and types.
3.  **Show a Real Workflow:** The usage example must demonstrate a realistic, multi-step workflow, showing how this module interacts with others.

**Your Workflow:**
1.  **Analyze the Module's Role and API (Deeper Analysis):**
    *   **Determine Consumers:** Use the dependency report to identify which other modules import and use your target module. These are your audience.
    *   **Define the Public API:** Analyze those consuming modules to see *which specific functions, classes, and constants* they use. This is the API you must document.
    *   **Understand the Data Flow:** Identify what data structures the module consumes and what it produces. This defines its contract.

2.  **Draft the Docstring using the Hardened Template:**
    Write the docstring following the detailed, multi-section template below. You must fill out every section.

3.  **Verify Constraints:**
    *   **Size:** Use the provided script to ensure the docstring is <10% of the file's total lines.
    *   **Clarity:** Read your draft from the perspective of a new developer. Does it provide everything they need to use the module correctly without reading its source code?

4.  **Insert Docstring and Report Completion.**

---
### **Hardened Docstring Template (for Sub-Agent)**

```python
"""
<Section 1: High-Level Summary>
<One-line summary of the module's purpose and role.>

This module serves as the <e.g., final stage of the data pipeline>. It is responsible
for transforming <input data structure, e.g., RawData objects> from the
`<library_name>.raw_data` module into <output data structure, e.g., PtychoDataContainer instances>,
which contain model-ready TensorFlow tensors.

Its primary consumer is the `<library_name>.workflows.components` module, which uses it to
prepare data for training and inference.
"""

"""
<Section 2: Key Abstractions (if any)>
Key Components:
- `ClassName`: <Describe the purpose of the main class/data structure defined here.
  List its most important attributes and what they represent.>
  - `.X`: The diffraction patterns, shape=(n_images, H, W, C), dtype=tf.float32.
  - `.Y`: The ground truth object patches, shape=(n_images, H, W, C), dtype=tf.complex64.
  - `.coords_nominal`: The nominal scan coordinates, shape=(...).
"""

"""
<Section 3: Public Functions/Classes>
Public Interface:
    `function_name(param1, param2, K=...)`
        - Description: <A concise description of what the function does.>
        - Parameters:
            - `param1` (type): <Explanation of this parameter's role.>
            - `param2` (type): <Explanation of this parameter's role.>
            - `K` (int): **Controls the number of nearest neighbors for grouping.**
              A larger `K` provides more potential neighbors for overlap-based
              training but increases computational cost during data preparation.
              Typical values are between 4 and 8.
"""

"""
<Section 4: Workflow Usage Example>
Usage Example:
    This module is typically used as part of the full data loading pipeline,
    which starts with a `RawData` object.

    ```python
    from <library_name>.raw_data import RawData
    from <library_name> import loader
    from <library_name> import params

    # 1. Assume `raw_data` is a populated RawData instance
    raw_data = RawData(...) 
    params.set('gridsize', 2) # Set config for grouping

    # 2. Generate the intermediate dictionary using the raw_data method
    #    This performs the expensive nearest-neighbor search.
    grouped_data_dict = raw_data.generate_grouped_data(N=64, K=7)

    # 3. Use this module's `load` function to create the final container
    #    The lambda function defers the execution of the dictionary access.
    data_container = loader.load(
        cb=lambda: grouped_data_dict,
        probeGuess=raw_data.probeGuess,
        which='train',
        create_split=False
    )
    
    # The `data_container` is now ready to be passed to the model.
    model.train(data_container)
    ```
"""
```

---

### **Sub-Agent Instructions: Final Verification Agent**

*(These are the instructions for the agent spawned in Phase 3.B)*

**Your Goal:** To perform a final consistency check on all newly created docstrings.

**Your Context:**
*   The list of all documented modules: `modules_to_document.txt`
*   The full dependency map: `<library_name>/dependency_report.txt`

**Your Workflow:**
1.  Read the module-level docstring from every file listed in `modules_to_document.txt`.
2.  For each docstring, compare its description of its "consumers" (who uses it) against the actual dependency information in `<library_name>/dependency_report.txt`.
3.  **Identify Inconsistencies:**
    *   Does a docstring claim it's used by Module A, but the dependency report shows no such link?
    *   Does a docstring's usage example show a pattern that is not actually used anywhere in the codebase?
    *   Is there a circular reference in the descriptions (e.g., A's docstring says it's for B, and B's docstring says it's for A)?
4.  **Report Findings:** Generate a `docstring_consistency_report.md` file listing any discrepancies or suggestions for improvement. If all docstrings are consistent, the report should state that verification passed.
