class ClassName:
    CLASS_IDENTIFIER = None

    def __init__(self) -> None:
        self._internal_variable = 0
        self.normal_variable = 1

    def _internal_func_call(self):
        ...

    def normal_func_call(self):
        ...

    @classmethod
    def class_method_call(cls):
        ...

    @property
    def PropertyName(self):
        ...

    @staticmethod
    def StaticFuncCall():
        ...

# [Hint string]
# <TypeName>