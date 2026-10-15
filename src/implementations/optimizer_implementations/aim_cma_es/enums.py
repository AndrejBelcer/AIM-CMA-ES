from enum import Enum, auto

#############
# Class Types
#############

class Archive_Type(Enum):
    KDTREE = auto()
    KDTREEN = auto()
    BALLTREE = auto()
    HNSW = auto()


class Initializer_Type(Enum):
    COBYQA = "COBYQA"
    COBYQAOFF = "COBYQAOFF"
    CMA = "CMA"


class CMA_Type(Enum):
    CMA_ORIGINAL_PURE = "cma_original_pure"
    CMA_ORIGINAL_ELITE = "cma_original_elite"
    CMA_NORMAL = "cma_normal"
    CMA_ELITE = "cma_elite"
    CMA_LEAPFROG = "cma_leapfrog"
    CMA_SYM = "cma_sym"

# ==========================================
# Refinement ENUMS
# ==========================================

class LocalityStrategy(Enum):
    FPS_UNIFORM_RADIUS = auto()
    FPS_UNIFORM_RADIUS_SPLIT = auto()

class SparsifierStrategy(Enum):
    NONE = auto()
    VOXEL = auto()
    RVOXEL = auto()


class FitterStrategy(Enum):
    STANDARD_SVD = auto()
    TRIMMED_SVD = auto()
    RANSAC = auto()

class SamplerStrategy(Enum):
    GRID = auto()
    DOUBLE_GRID = auto()
    RANDOM = auto()

class ValidatorStrategy(Enum):
    NONE = auto()

# ==========================================
# Other
# ==========================================

class CVType(Enum):
    LOGSUMEXP = "LogSumExp"
    LINEAR = "Linear"
    QUADRATIC = "Quadratic"
    CHEBYSHEV = "Chebyshev"

class ScalarizationMode(Enum):
    FEASIBILITY = "Feasibility"
    CORNER_SEARCH = "Corner Search"
    EXPLOITATION = "Exploitation"
    LOCAL_EXPLORATION = "Local Exploration"
    GLOBAL_EXPLORATION = "Global Exploration"


class InitializerPhase(Enum):
    LHS_SEEDING = "LHS_Seeding"
    OPTIMIZER = "OPTIMIZER"
    LHS_FINAL = "LHS_Final"

class SolverPhase(Enum):
    INITIALIZATION = "Initialization"
    CMA = "CMA"
    REFINEMENT = "Refinement"
    NONE = "None"
    TEST = "Test"

