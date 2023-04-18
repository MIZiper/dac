from uuid import uuid4

class NodeBase:
    def __init__(self, name: str=None, uuid: str=None) -> None:
        self._hash = None

        self.name = name
        self.uuid = uuid or str(uuid4())

    def calc_hash(self) -> str:
        ...

    def get_hash(self, force_recalc=False):
        if self._hash is None or force_recalc:
            self._hash = self.calc_hash()
        return self._hash

class DataNode(NodeBase):
    ...

class ActionNode(NodeBase):
    ...

class Context:
    ...