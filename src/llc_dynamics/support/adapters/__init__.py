"""Adapters for external LLC trajectory formats."""

from .matlab_trajectory import MATLABTrajectory, load_matlab_trajectory

__all__ = ["MATLABTrajectory", "load_matlab_trajectory"]
