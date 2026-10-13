import os

# 1. Get the absolute path of the directory containing this file (core/)
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE_DIR = os.path.dirname(_FILE_DIR)


# 2. Go up one level to get the Project Root
_PROJECT_ROOT = os.path.dirname(_CORE_DIR)


# 3. Define Output Directories
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "Results")
PROBLEMS_DIR = os.path.join(_PROJECT_ROOT, "Problems")
SOLVER_RESULTS_DIR = os.path.join(RESULTS_DIR, "Solver_Results")
PLOT_RESULTS_DIR = os.path.join(RESULTS_DIR, "Plot_Results")
COBI_PROBLEMS_DIR = os.path.join(PROBLEMS_DIR, "Cobi_Problems")
COCO_RESULTS_DIR = os.path.join(RESULTS_DIR, "COCO_exdata")
SOLVER_DEBUG_DIR = os.path.join(RESULTS_DIR, "Solver_Debug")

# 4. Auto-create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PROBLEMS_DIR, exist_ok=True)

os.makedirs(SOLVER_RESULTS_DIR, exist_ok=True)
os.makedirs(PLOT_RESULTS_DIR, exist_ok=True)
os.makedirs(COBI_PROBLEMS_DIR, exist_ok=True)
os.makedirs(COCO_RESULTS_DIR, exist_ok=True)
os.makedirs(SOLVER_DEBUG_DIR, exist_ok=True)

print(f"Output directories initialized at: {RESULTS_DIR} and {PROBLEMS_DIR}")