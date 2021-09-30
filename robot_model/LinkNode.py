from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class LinkNode:
    """Class for link node info"""
    id: int
    name: str = "Untitled"
    children: List = None
    mother: int = None

    is_leaf: bool = False

    a: np.ndarray = None  # Joint axis vector (relative to parent)
    b: np.ndarray = None  # Joint relative position (relative to parent)
    p: np.ndarray = None  # Position in world coordinates
    q: np.ndarray = None  # Joint angle
    R: np.ndarray = None  # Attitude in world coordinates
    # dq: np.float  = None  # Joint Velocity
    v: np.ndarray = None  # Linear Velocity in World Coordinate
    w: np.ndarray = None  # Angular Velocity in World Coordinates