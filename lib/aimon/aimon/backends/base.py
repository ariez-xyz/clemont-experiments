from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseBackend(ABC):
    """Abstract base class for monitoring backends.

    This class serves as a template for different prediction backend implementations.
    All concrete backends must implement the observe() method.

    Parameters
    ----------
    pred : str
        Prediction column.
    epsilon : float
        Closeness parameter

    Attributes
    ----------
    meta : dict, read-only
        Metadata dictionary containing backend information
    """

    def __init__(self, decision_col, epsilon):
        self._meta = {
            "epsilon": epsilon,
            "decision_col": decision_col,
            "metric": "Linf",
            "is_exact": False,
            "is_sound": False,
            "is_complete": False,
        }

    @property
    def meta(self):
        """dict: Read-only access to backend metadata."""
        return self._meta

    @abstractmethod
    def observe(self, row, row_id=None) -> List[int]:
        """Process an observation through the backend.
        
        Returns
        -------
        List of row_ids that are unfair counterexamples to the parameter row.

        Raises
        ------
        NotImplementedError
            Because this is an abstract method that needs to be implemented
            by concrete backend classes.
        """
        pass
