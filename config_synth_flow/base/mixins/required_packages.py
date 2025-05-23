"""
Mixin class for checking required packages.
"""

from importlib import util


class RequiredPackagesMixin:
    """Mixin for checking required packages"""

    def check_required_packages(self) -> None:
        """
        Check if the required packages are installed.

        Raises:
            ImportError: If any of the required packages are missing.
        """
        missing_packages = [pkg for pkg in self.required_packages if not util.find_spec(pkg)]
        if missing_packages:
            raise ImportError(
                f"Missing required packages: {missing_packages} in Pipeline {self.class_name}."
            )
