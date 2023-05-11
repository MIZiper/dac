from uuid import uuid4
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any
from types import GenericAlias
import inspect
from enum import IntEnum

class NodeBase:
    def __init__(self, name: str=None, uuid: str=None) -> None:
        self._hash = None
        self._construct_config = {}

        self.name = name
        self.uuid = uuid or str(uuid4())

    def calc_hash(self) -> str:
        ...

    def get_hash(self, force_recalc=False):
        if self._hash is None or force_recalc:
            self._hash = self.calc_hash()
        return self._hash
    
    def get_construct_config(self) -> dict:
        return {
            "name": self.name,
            **self._construct_config,
        }

    def apply_construct_config(self, construct_config: dict):
        self.name = construct_config.get("name", None)

    def get_save_config(self) -> dict:
        return {
            "_uuid_": self.uuid,
            "_class_": self.__class__.__name__,
            **self.get_construct_config(),
        }

    @classmethod
    def parse_save_config(cls, save_config: dict):
        ...



class DataNode(NodeBase):
    ...

class DataClassNode(DataNode):
    def __post_init__(self):
        super().__init__(name=self.name)

    def get_construct_config(self) -> dict:
        return asdict(self)
    
    def apply_construct_config(self, construct_config: dict):
        for key, value in construct_config.items():
            if hasattr(self, key):
                setattr(self, key, value)

@dataclass(eq=False)
class GlobalContextKey(DataClassNode):
    name: str

GCK = GlobalContextKey("Global Context Key")



class ActionNode(NodeBase):
    CAPTION = "Not implemented action"

    class ActionStatus(IntEnum):
        INIT = 0
        CONFIGURED = 1
        COMPLETE = 2
        FAILED = -1

    @staticmethod
    def Annotation2Config(ann: type | GenericAlias):
        if ann.__name__=="list":
            return [f"<{t.__name__}>" for t in ann.__args__]
        return f"<{ann.__name__}>"

    def __init__(self, context_key: NodeBase, name: str = None, uuid: str = None) -> None:
        super().__init__(name=self.CAPTION, uuid=uuid)
        
        self.status = ActionNode.ActionStatus.INIT
        self.out_name = None

        self._context_key = context_key
        self._SIGNATURE = inspect.signature(self.__call__)
        # `self._construct_config` has same parameters for `__call__`

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # annotations have to be specified; if there is 'list', `list[...]` must be used
        return super().__call__(*args, **kwds)
    
    def get_construct_config(self) -> dict:
        if not self._construct_config: # init
            cfg = self._construct_config
            for key, param in self._SIGNATURE.parameters.items():
                if param.default is not inspect._empty:
                    cfg[key] = param.default
                elif param.annotation is not inspect._empty:
                    cfg[key] = ActionNode.Annotation2Config(param.annotation)
                else:
                    cfg[key] = "<Any>"
            if (ret_ann:=self._SIGNATURE.return_annotation) is not inspect._empty or ret_ann.__name__!="list":
                self.out_name = f"<{ret_ann.__name__}>"
                
        com_config = {
            "name": self.name,
            **self._construct_config,
        }
        if self.out_name is not None:
            com_config["out_name"] = self.out_name

        return com_config
    
    def apply_construct_config(self, construct_config: dict):
        if "name" in construct_config:
            self.name = construct_config["name"]
            # del construct_config["name"]
        if "out_name" in construct_config:
            self.out_name = construct_config["out_name"]
            # del construct_config["out_name"]

        # TODO: validate the construct_config
        self._construct_config = construct_config

        # if self.status==ActionNode.ActionStatus.INIT:
        self.status = ActionNode.ActionStatus.CONFIGURED
    
    def get_save_config(self) -> dict:
        cfg = super().get_save_config()

        if self._context_key is not GCK:
            cfg["_context_"] = self._context_key.uuid

        return cfg



class DataContext:
    @property
    def NodeIter(self) -> list[tuple[type[DataNode], str, DataNode]]:
        ...

    def add_node(self, node: NodeBase):
        ...

    def get_node_of_type(self, node_name: str, node_type: type[NodeBase]) -> NodeBase:
        ...

class Container:
    _register = {}

    def __init__(self) -> None:
        self.actions: list[ActionNode] = []
        self.contexts: dict[NodeBase, DataContext] = defaultdict(lambda: DataContext(self))
        self._current_key = GCK

    @property
    def GlobalContext(self) -> DataContext:
        return self.contexts[GCK]

    @property
    def CurrentContext(self) -> DataContext:
        return self.contexts[self._current_key]
    
    @property
    def CurrentActionsIter(self) -> list[ActionNode]:
        return filter(lambda a: a._context_key is self._current_key, self.actions)

    def get_node_of_type(self, node_name: str, node_type: type[NodeBase]) -> NodeBase:
        ...

    def activate_context(self, context_key: NodeBase) -> DataContext:
        self._current_key = context_key
        return self.CurrentContext

    def get_context(self, context_key: NodeBase) -> DataContext:
        return self.contexts[context_key]
    
    def remove_global_node(self, node_object: NodeBase):
        del self.GlobalContext[type[node_object]][node_object.name]
        if node_object in self.contexts:
            del self.contexts[node_object]
        self.actions = [action for action in self.actions if action._context_key is not node_object]
    
    def prepare_params_for_action(self, action: ActionNode) -> dict:
        params = {}
        construct_config = action._construct_config
        for key, param in action._SIGNATURE.parameters.items():
            value = construct_config.get(key, param.default)
            if value is inspect._empty:
                # not provided and no default
                raise Exception(f"Parameter '{key}' not provided.")
            
            ann = param.annotation
            if issubclass(ann, DataNode):
                node_type = ann
                node_name = value
                if (value:=self.get_node_of_type(node_name, node_type)) is None:
                    raise Exception(f"Node '{node_name}' of '{node_type.__name__}' not found.")
            elif ann.__name__=="list" and (
                issubclass((node_type:=ann.__args__[0]), DataNode)
            ): # assert type(ann) is GenericAlias # list[...]
                node_names = value
                value = [self.get_node_of_type(node_name, node_type) for node_name in node_names]
                # TODO: remove the potential `None`s

            params[key] = value
            
        return params
    
    def get_save_config(self):
        return {
            "actions": [action.get_save_config() for action in self.actions],
            # TODO: error if "_context_" was deleted before saving

            "global_nodes": [n_o.get_save_config() for n_t, n_n, n_o in self.GlobalContext.NodeIter],
        }

    @staticmethod
    def parse_save_config(config: dict) -> "Container":
        container = Container()
        nodes = {}

        g_nodes = config.get("global_nodes") or []
        for data_config in g_nodes:
            cls_name = data_config['_class_']
            del data_config['_class_']
            uuid = data_config['_uuid_']
            del data_config['_uuid_']

            data_class: type[DataNode] = Container.GetType(cls_name)
            data_node = data_class(name="[Default]", uuid=uuid)
            data_node.apply_construct_config(data_config)

            nodes[uuid] = data_node

            container.GlobalContext.add_node(data_node)

        actions = config.get("actions") or []
        for act_config in actions:
            cls_name = act_config['_class_']
            del act_config['_class_']
            uuid = act_config['_uuid_']
            del act_config['_uuid_']

            act_class: type[ActionNode] = Container.GetType(cls_name)
            if '_context_' in act_config:
                context_key = nodes[act_config['_context_']]
                # TODO: error if "_context_" was deleted before saving
                del act_config['_context_']
            else:
                context_key = GCK
            
            action_node = act_class(context_key=context_key, uuid=uuid)
                
            action_node.apply_construct_config(act_config)

            container.actions.append(action_node)

    @classmethod
    def RegisterType(cls, node_type: type[NodeBase]):
        # __subclass__
        ...

    @classmethod
    def GetType(cls, type_name: str) -> type[NodeBase]:
        ...

    @classmethod
    def RegisterContextAction(cls, context_type: type[NodeBase], action_type: type[ActionNode]):
        ...

    @classmethod
    def CurrentActionTypesIter(cls) -> list[type[ActionNode]]:
        ...