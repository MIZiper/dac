from uuid import uuid4
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any
from types import GenericAlias
import inspect, importlib
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
        cls = self.__class__

        return {
            "_uuid_": self.uuid,
            "_class_": f"{cls.__module__}.{cls.__qualname__}",
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
        if hasattr(ann, '_fields'): # namedtuple
            return [f"[{f}]" for f in ann._fields]
        elif isinstance(ann, GenericAlias): # ok: list[], tuple[]; nok: dict[], type[]
            return [
                ActionNode.Annotation2Config(t)
                for t in ann.__args__
            ]
        else:
            return f"<{ann.__name__}>"

    def __init__(self, context_key: DataNode, name: str = None, uuid: str = None) -> None:
        super().__init__(name=self.CAPTION, uuid=uuid)
        
        self.status = ActionNode.ActionStatus.INIT
        self.out_name = None

        self._context_key = context_key
        self._SIGNATURE = inspect.signature(self.__call__)
        # `self._construct_config` has same parameters for `__call__`

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # annotations have to be specified; if there is 'list', `list[...]` must be used
        # output type also needs specified
        print(f"'{self.name}' called with {args} and {kwds}.")
    
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
            if (ret_ann:=self._SIGNATURE.return_annotation) is not inspect._empty and ret_ann.__name__!="list":
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
    
    def pre_run(self, *args, **kwargs):
        ...

    def post_run(self, *args, **kwargs):
        ...



class DataContext(dict[type[DataNode], dict[str, DataNode]]):
    def __init__(self, container: "Container") -> None:
        super().__init__()
        self._container = container

    @property
    def NodeIter(self) -> list[tuple[type[DataNode], str, DataNode]]:
        for node_type, nodes in self.items():
            for node_name, node in nodes.items():
                yield (node_type, node_name, node)

    def add_node(self, node: NodeBase):
        node_type = type(node)
        if node_type in self:
            self[node_type][node.name] = node
        else:
            self[node_type] = {node.name: node}

    def get_node_of_type(self, node_name: str, node_type: type[NodeBase]) -> NodeBase:
        try:
            return self[node_type][node_name]
        except KeyError:
            return None
        
    def rename_node_to(self, node: NodeBase, new_name: str):
        try:
            del self[type(node)][node.name]
        except:
            pass
        node.name = new_name
        self.add_node(node)

class Container:
    _global_node_types = []
    _context_action_types = defaultdict(list)

    def __init__(self) -> None:
        self.actions: list[ActionNode] = []
        self.contexts: dict[DataNode, DataContext] = defaultdict(lambda: DataContext(self))
        self._current_key: DataNode = GCK

    @property
    def GlobalContext(self) -> DataContext:
        return self.contexts[GCK]

    @property
    def CurrentContext(self) -> DataContext:
        return self.contexts[self._current_key]
    
    @property
    def ActionsInCurrentContext(self) -> list[ActionNode]:
        return filter(lambda a: a._context_key is self._current_key, self.actions)

    def get_node_of_type(self, node_name: str, node_type: type[NodeBase]) -> NodeBase:
        if node:=self.CurrentContext.get_node_of_type(node_name, node_type):
            return node
        elif self._current_key is not GCK:
            return self.GlobalContext.get_node_of_type(node_name, node_type)
        else:
            return None

    def activate_context(self, context_key: DataNode) -> DataContext:
        self._current_key = context_key
        return self.CurrentContext

    def get_context(self, context_key: DataNode) -> DataContext:
        return self.contexts[context_key]
    
    def remove_global_node(self, node_object: NodeBase):
        del self.GlobalContext[type(node_object)][node_object.name]
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
            # TODO: use recursive like `Annotation2Config`, for the list-of-tuple-of-sth or tuple-of-list-of-sth

            params[key] = value
            
        return params
    
    def get_save_config(self):
        return {
            "actions": [action.get_save_config() for action in self.actions],
            "global_nodes": [n_o.get_save_config() for n_t, n_n, n_o in self.GlobalContext.NodeIter],
        }

    @classmethod
    def parse_save_config(cls, config: dict) -> "Container":
        container = Container()
        nodes = {}

        g_nodes = config.get("global_nodes") or []
        for data_config in g_nodes:
            cls_path = data_config['_class_']
            del data_config['_class_']
            uuid = data_config['_uuid_']
            del data_config['_uuid_']

            data_class: type[DataNode] = Container.GetClass(cls_path)
            data_node = data_class(name="[Default]", uuid=uuid)
            data_node.apply_construct_config(data_config)

            nodes[uuid] = data_node

            container.GlobalContext.add_node(data_node)

        actions = config.get("actions") or []
        for act_config in actions:
            cls_path = act_config['_class_']
            del act_config['_class_']
            uuid = act_config['_uuid_']
            del act_config['_uuid_']

            act_class: type[ActionNode] = Container.GetClass(cls_path)
            if '_context_' in act_config:
                context_key = nodes[act_config['_context_']]
                # TODO: error if "_context_" was deleted before saving
                del act_config['_context_']
            else:
                context_key = GCK
            
            action_node = act_class(context_key=context_key, uuid=uuid)
                
            action_node.apply_construct_config(act_config)

            container.actions.append(action_node)

        return container

    @staticmethod
    def GetClass(class_path: str) -> type[NodeBase]:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @staticmethod
    def RegisterGlobalDataType(node_type: type[DataNode] | str):
        Container._global_node_types.append(node_type)

    @staticmethod
    def GetGlobalDataTypes() -> list[type[DataNode] | str]:
        return Container._global_node_types

    @staticmethod
    def RegisterContextAction(context_type: type[DataNode], action_type: type[ActionNode] | str):
        Container._context_action_types[context_type].append(action_type)

    @staticmethod
    def GetContextActionTypes(context_type: type[DataNode]) -> list[type[ActionNode] | str]:
        return Container._context_action_types[context_type]
    
    @staticmethod
    def RegisterGlobalContextAction(action_type: type[ActionNode] | str):
        Container.RegisterContextAction(GlobalContextKey, action_type)

    @property
    def ActionTypesInCurrentContext(self) -> list[type[ActionNode] | str]:
        context_type = type(self._current_key)
        return Container.GetContextActionTypes(context_type)