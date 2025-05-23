"""
Mixin class for serialization functionality.
"""

import pickle
import types
from typing import Any


class SerializableMixin:
    """Mixin for serialization functionality with enhanced support for multiprocessing"""

    # Class variable to track non-serializable attributes
    _non_serializable_attrs: set[str] = {"_logger"}

    def serialize(self):
        """
        Serialize the object to a byte stream.

        Returns:
            bytes: Serialized byte stream.
        """
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data):
        """
        Deserialize the object from a byte stream.

        Args:
            data (bytes): Serialized byte stream.

        Returns:
            Serializable: Deserialized object.
        """
        return pickle.loads(data)

    def __getstate__(self) -> dict[str, Any]:
        """
        Get the state of the object for serialization.
        Handles non-serializable objects like loggers, locks, and thread-local data.

        Returns:
            dict: State of the object with non-serializable objects removed.
        """
        state = self.__dict__.copy()

        # Remove known non-serializable attributes
        for attr in self._non_serializable_attrs:
            if attr in state:
                state.pop(attr, None)

        # Handle function and method objects
        for key, value in list(state.items()):
            # Handle functions, methods, and other non-serializable objects
            if isinstance(value, types.FunctionType) or isinstance(value, types.MethodType):
                state[key] = None
            # Handle thread locks and other threading objects
            elif hasattr(value, "__module__") and value.__module__ == "threading":
                state[key] = None
            # Handle multiprocessing objects
            elif hasattr(value, "__module__") and value.__module__ == "multiprocessing":
                state[key] = None
            # Try to check if the object is picklable
            elif key not in self._non_serializable_attrs:
                try:
                    pickle.dumps(value)
                except (TypeError, pickle.PickleError):
                    # If we can't pickle it, set it to None
                    state[key] = None

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        set the state of the object during deserialization.
        Reinitializes any non-serializable objects that were removed during serialization.

        Args:
            state (dict): State of the object.
        """
        self.__dict__.update(state)

        # Reinitialize any necessary objects that weren't serialized
        # For example, if _logger was removed, it will be recreated on first access
        # through the logger property in LoggingMixin
