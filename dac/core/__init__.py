from uuid import uuid4
from collections import defaultdict
from typing import Any
from types import GenericAlias
import inspect, importlib
from enum import IntEnum, Enum

class NodeBase:
    def __init__(self, name: str=None, uuid: str=None) -> None:
        self._hash = None
        self._construct_config = {}

        self.name = name
        self._uuid = uuid or str(uuid4()) # _ to avoid shown in construct_config

    @property
    def uuid(self):
        return self._uuid

    def calc_hash(self) -> str:
        ...

    def get_hash(self, force_recalc=False):
        if self._hash is None or force_recalc:
            self._hash = self.calc_hash()
        return self._hash
    
    def get_construct_config(self) -> dict:
        raise NotImplementedError

    def apply_construct_config(self, construct_config: dict):
        raise NotImplementedError

    def get_save_config(self) -> dict:
        cls = self.__class__

        return {
            "_uuid_": self.uuid,
            "_class_": f"{cls.__module__}.{cls.__qualname__}",
            **self.get_construct_config(),
        }



class DataNode(NodeBase):
    BASICTYPES = (int, float, str, bool)
    @staticmethod
    def Value2BasicTypes(v):
        if isinstance(v, DataNode.BASICTYPES):
            return v
        elif isinstance(v, (list, tuple)):
            return [DataNode.Value2BasicTypes(e) for e in v]
        elif isinstance(v, dict):
            return {k: DataNode.Value2BasicTypes(e) for k, e in v.items()}
        else:
            return f"<{type(v).__name__}>"
        
    @staticmethod
    def ContainsOnlyBasicTypes(v):
        if isinstance(v, str) and v.startswith("<") and v.endswith(">"):
            return False
        elif isinstance(v, (list, tuple)):
            for e in v:
                if not DataNode.ContainsOnlyBasicTypes(e):
                    return False
        elif isinstance(v, dict):
            for k, e in v.items():
                if not DataNode.ContainsOnlyBasicTypes(e):
                    return False
        return True

    def get_construct_config(self) -> dict:
        # _construct_config is same as __dict__
        return {
            k: DataNode.Value2BasicTypes(v)
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
    
    def apply_construct_config(self, construct_config: dict):
        for k, v in construct_config.items():
            if k in self.__dict__ and DataNode.ContainsOnlyBasicTypes(v):
                self.__dict__[k] = v

class GlobalContextKey(DataNode):
    pass

GCK = GlobalContextKey("Global Context Key")

class NodeNotFoundError(Exception):
    pass



class ActionNode(NodeBase):
    CAPTION = "Not implemented action"
    _SIGNATURE = None

    def __init_subclass__(cls) -> None:
        cls._SIGNATURE = inspect.signature(cls.__call__)

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

        self.context_key = context_key
        self.container: Container = None # for the actions require external resources, normally when context_key is GCK

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # annotations have to be specified; if there is 'list', `list[...]` must be used
        # output type also needs specified
        print(f"'{self.name}' called with {args} and {kwds}.")
    
    def get_construct_config(self) -> dict:
        if not self._construct_config: # init
            cfg = self._construct_config
            for key, param in self._SIGNATURE.parameters.items():
                if key=="self":
                    continue
                elif param.default is not inspect._empty:
                    if isinstance(param.default, Enum):
                        cfg[key] = param.default.name
                    else:
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

        if self.context_key is not GCK:
            cfg["_context_"] = self.context_key.uuid

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
            self[node_type][node.name] = node # in global context, should delete original node related actions (potential error)
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
    _type_agencies = {}

    def __init__(self) -> None:
        self.actions: list[ActionNode] = []
        self.contexts: dict[DataNode, DataContext] = defaultdict(lambda: DataContext(self))
        self.current_key: DataNode = GCK

    @property
    def GlobalContext(self) -> DataContext:
        return self.contexts[GCK]

    @property
    def CurrentContext(self) -> DataContext:
        return self.contexts[self.current_key]
    
    @property
    def ActionsInCurrentContext(self) -> list[ActionNode]:
        return filter(lambda a: a.context_key is self.current_key, self.actions)

    def get_node_of_type(self, node_name: str, node_type: type[NodeBase]) -> NodeBase:
        if node:=self.CurrentContext.get_node_of_type(node_name, node_type):
            return node
        elif self.current_key is not GCK:
            return self.GlobalContext.get_node_of_type(node_name, node_type)
        else:
            # search in agency (?)
            return None

    def activate_context(self, context_key: DataNode) -> DataContext:
        self.current_key = context_key
        return self.CurrentContext

    def get_context(self, context_key: DataNode) -> DataContext:
        return self.contexts[context_key]
    
    def remove_global_node(self, node_object: NodeBase):
        del self.GlobalContext[type(node_object)][node_object.name]
        if node_object in self.contexts:
            del self.contexts[node_object]
        self.actions = [action for action in self.actions if action.context_key is not node_object]

    def _get_value_of_annotation(self, ann: type | GenericAlias, config: Any):
        if config is None:
            return None
        elif issubclass(ann, Enum):
            if isinstance(config, Enum): # from default
                return config
            else: # str
                return ann[config]
        elif issubclass(ann, DataNode):
            if (value:=self.get_node_of_type(node_name=config, node_type=ann)) is None:
                raise NodeNotFoundError(f"Node '{config}' of '{ann.__name__}' not found.")
            return value
        elif isinstance(ann, GenericAlias):
            if ann.__name__=="list" and len(ann.__args__)==1:
                value = []
                for c in config:
                    try:
                        v = self._get_value_of_annotation(ann.__args__[0], c)
                    except NodeNotFoundError:
                        continue
                    value.append(v)
            elif ann.__name__=="tuple":
                value = [self._get_value_of_annotation(a, c) for a, c in zip(ann.__args__, config)]
            else:
                raise NotImplementedError
            
            return value
        elif ann in Container._type_agencies:
            return Container._type_agencies[ann](config)
        else:
            return config
    
    def prepare_params_for_action(self, signature: inspect.Signature | dict, construct_config: dict) -> dict:
        params = {}
        if isinstance(signature, inspect.Signature):
            for key, param in signature.parameters.items():
                if key=="self":
                    continue
                value = construct_config.get(key, param.default)
                if value is inspect._empty:
                    # not provided and no default
                    raise Exception(f"Parameter '{key}' not provided.")

                params[key] = self._get_value_of_annotation(param.annotation, value)
        else: # signature is a dict, SAB
            for subact_name, subact_signature in signature.items():
                subact_config = construct_config.get(subact_name, {})
                params[subact_name] = self.prepare_params_for_action(subact_signature, subact_config)
        return params
    
    def get_save_config(self):
        return {
            "actions": [action.get_save_config() for action in self.actions],
            "global_nodes": [n_o.get_save_config() for n_t, n_n, n_o in self.GlobalContext.NodeIter],
            # add combo_action definitions, combo_action = action1 + action2 + action3 ...
            # able to define pre-defined parameters and user-input parameters
            # only available in current project, but can be saved as a template
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
                if act_config['_context_'] not in nodes:
                    continue
                context_key = nodes[act_config['_context_']]
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
        context_type = type(self.current_key)
        return Container.GetContextActionTypes(context_type)
    
    @staticmethod
    def RegisterNodeTypeAgency(node_type: type, agent_func: callable):
        # # `agent_func` can take any input,
        # # normally take one var-representative string
        # # and output an object
        # def agent_func(arg: Any) -> Any:
        #     ...
        Container._type_agencies[node_type] = agent_func