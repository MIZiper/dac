"""A util to extract common part of file paths, and easily relocate for service mapping.

Generic purpose, not used in DAC yet.
"""

import os

class FilePath:
    _common_prefix = ""
    _map = None
    _pool = []

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        FilePath._pool.append(self)

    def __del__(self):
        FilePath._pool.remove(self)

    def __str__(self):
        pass

    @property
    def common_prefix(self):
        return FilePath._common_prefix

    @property
    def relative_path(self):
        pass

    @staticmethod
    def compact():
        if not FilePath._pool:
            return ""
        
        paths = [fp.path for fp in FilePath._pool]
        common_prefix = os.path.commonprefix(paths)
        
        return common_prefix, [os.path.relpath(p, common_prefix) for p in paths]
    
    @staticmethod
    def clear():
        FilePath._common_prefix = ""
        FilePath._pool.clear()

    @staticmethod
    def relocate(common_prefix, new_prefix):
        pass

    @property
    def target_path(self):
        # When initializing:
        # >>> fpath = FilePath("file://Base_A/file-1.txt")
        # >>> FilePath.relocate("Base_A", "Base_B")
        # Then when loading:
        # >>> fpath.target_path == "file://Base_B/file-1.txt"
        pass

# if FilePath._common_prefix is not accessible to all, make an alert