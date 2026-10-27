"""DACE: intervention-grounded dual attribution research implementation."""
from .core import *
from .harness import BASELINE_CONFIG, BOUNDARY_FIELDS, CONFIG_KEYS, Fault, SimulatedEnvironment, Task
from .components import (AblationLevel, LEVEL_FEATURES, Mechanism, MechanismRegistry,
                         offline_propose, candidate_patches, joint_compositions,
                         locality_patch, prediction_disagreement, scenario)
from .evolution import EvolutionLoop, MethodSpec, RoundRecord, RunSummary
