"""Core Data Action Context components.

This module defines the base classes for data nodes / contexts, action nodes, and containers.
"""

from uuid import uuid4
from collections import defaultdict
from typing import Any, Optional, get_origin, get_args, Union
from types import GenericAlias, UnionType
import inspect, importlib
from enum import IntEnum, Enum


class NodeBase:
    def __init__(self, name: Optional[str] = None, uuid: Optional[str] = None) -> None:
        self._hash = None
        self._construct_config = {}

        self.name = name
        self._uuid = uuid or str(uuid4())  # _ to avoid shown in construct_config

    @property
    def uuid(self):
        return self._uuid

    def calc_hash(self) -> str: ...

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


class NodeNotFoundError(Exception):
    pass


class DataNode(NodeBase):
    BASICTYPES = (int, float, str, bool)

    def __init__(
        self, name: Optional[str] = None, uuid: Optional[str] = None, parent: Optional["DataNode"] = None
    ) -> None:
        super().__init__(name, uuid)
        self._parent = parent
        self._children: list[DataNode] = []

    @property
    def parent(self) -> Optional["DataNode"]:
        return self._parent

    @property
    def children(self) -> list["DataNode"]:
        return self._children

    def add_child(self, child: "DataNode"):
        child._parent = self
        self._children.append(child)

    def get_child(self, name: str) -> Optional["DataNode"]:
        for child in self._children:
            if child.name == name:
                return child
        return None

    def remove_child(self, name: str) -> Optional["DataNode"]:
        for i, child in enumerate(self._children):
            if child.name == name:
                child._parent = None
                return self._children.pop(i)
        return None

    def parent_until[T: "DataNode"](self, node_type: type[T]) -> T:
        current = self._parent
        while current:
            if isinstance(current, node_type):
                return current
            current = current._parent
        raise Exception(f"Didn't find a parent of <{node_type}>")

    def iter_all(self):
        yield self
        for child in self._children:
            yield from child.iter_all()

    @staticmethod
    def Value2BasicTypes(v):
        if (
            type(v) in DataNode.BASICTYPES
        ):  # `isinstance` not enough because e.g. `np.float64` is subclass of `float`
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
            if not k.startswith("_") and k != "children"
        }

    def apply_construct_config(self, construct_config: dict):
        for k, v in construct_config.items():
            if k in self.__dict__ and DataNode.ContainsOnlyBasicTypes(v):
                self.__dict__[k] = v


class ContextKeyNode(DataNode):
    pass


class GlobalContextKey(ContextKeyNode):
    pass


GCK = GlobalContextKey("Global Context")


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
    def Annotation2Config(ann: type | GenericAlias | UnionType):
        # Provide a user-readable construct-hint for many annotation varieties.
        # Supports: namedtuple, typing generics (list/tuple/dict/set), Union/Optional,
        # built-in generics (PEP585 GenericAlias), Enums, DataNode subclasses and Any.
        from typing import Any as _Any

        # namedtuple-like
        if hasattr(ann, "_fields"):
            return [f"[{f}]" for f in ann._fields]

        origin = get_origin(ann)
        args = get_args(ann)

        # PEP585 / GenericAlias and typing generics
        if origin is list or isinstance(ann, GenericAlias) and getattr(ann, "__name__", None) == "list":
            if args:
                return [ActionNode.Annotation2Config(args[0])]
            return ["<Any>"]
        if origin is tuple or (isinstance(ann, GenericAlias) and getattr(ann, "__name__", None) == "tuple"):
            if args:
                return [ActionNode.Annotation2Config(a) for a in args]
            return ["<Any>"]
        if origin is set or (isinstance(ann, GenericAlias) and getattr(ann, "__name__", None) == "set"):
            if args:
                return [ActionNode.Annotation2Config(args[0])]
            return ["<Any>"]
        if origin is dict or (isinstance(ann, GenericAlias) and getattr(ann, "__name__", None) == "dict"):
            if len(args) == 2:
                return {ActionNode.Annotation2Config(args[0]): ActionNode.Annotation2Config(args[1])}
            return {"<Any>": "<Any>"}

        # typing.Union / PEP604 unions
        if origin is None and isinstance(ann, UnionType) or origin is None and getattr(ann, "__module__", None) == "typing" and getattr(ann, "__origin__", None) is None and getattr(ann, "__args__", None):
            # fallback for older UnionType handling
            try:
                args = ann.__args__
            except Exception:
                args = ()

        if origin is not None and origin is Union or isinstance(ann, UnionType) or (getattr(ann, "__args__", None) and (origin is None and isinstance(ann, type(get_args(ann))))):
            union_args = args if args else getattr(ann, "__args__", ())
            return " | ".join([ActionNode.Annotation2Config(t) for t in union_args])

        # Callable - show signature hint
        if origin is not None and origin is callable or getattr(ann, "__origin__", None) is None and str(ann).startswith('typing.Callable'):
            if args and len(args) == 2:
                params_hint = (
                    [ActionNode.Annotation2Config(a) for a in args[0]] if isinstance(args[0], (list, tuple)) else "..."
                )
                return {"callable": {"params": params_hint, "return": ActionNode.Annotation2Config(args[1])}}
            return "<callable>"

        # Any
        if ann is _Any:
            return "<Any>"

        # Enum
        try:
            if issubclass(ann, Enum):
                return f"<{ann.__name__}>"
        except Exception:
            pass

        # DataNode subclasses
        try:
            if issubclass(ann, DataNode):
                return f"<{ann.__name__}>"
        except Exception:
            pass

        # Fallback: simple type
        try:
            return f"<{ann.__name__}>"
        except Exception:
            return str(ann)

    def __init__(
        self, context_key: DataNode, name: str = None, uuid: str = None
    ) -> None:
        super().__init__(name=self.CAPTION, uuid=uuid)

        self.status = ActionNode.ActionStatus.INIT
        self.out_name = None

        self.context_key = context_key
        self.container: Container = None  # for the actions require external resources, normally when context_key is GCK

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # annotations have to be specified; if there is 'list', `list[...]` must be used
        # output type also needs specified
        print(f"'{self.name}' called with {args} and {kwds}.")

    def get_construct_config(self) -> dict:
        if not self._construct_config:  # init
            cfg = self._construct_config
            for key, param in self._SIGNATURE.parameters.items():
                if key == "self":
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
            if (
                ret_ann := self._SIGNATURE.return_annotation
            ) is not inspect._empty and ret_ann.__name__ != "list":
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

    def pre_run(self, *args, **kwargs): ...

    def post_run(self, *args, **kwargs): ...


class DataContext(dict[type[DataNode], dict[str, DataNode]]):
    def __init__(self, container: "Container") -> None:
        super().__init__()
        self._container = container
        self._uuid_dict = {}  # {uuid: (node_type, name)} # don't store object to avoid ref
        self._name_index: dict[tuple[type, str], DataNode] = {}  # {(type, name): node}

    @property
    def NodeIter(self) -> list[tuple[type[DataNode], str, DataNode]]:
        for node_type, nodes in self.items():
            for node_name, node in nodes.items():
                yield (node_type, node_name, node)

    def _index_node(self, node: DataNode):
        node_type = type(node)
        self._name_index[(node_type, node.name)] = node
        self._uuid_dict[node.uuid] = (node_type, node.name)
        for child in node.children:
            self._index_node(child)

    def add_node(self, node: NodeBase):
        node_type = type(node)
        if node_type in self:
            self[node_type][node.name] = node
        else:
            self[node_type] = {node.name: node}
        self._index_node(node)

    def get_node_of_type(self, node_name: str, node_type: type[NodeBase]) -> NodeBase:
        if node_type in self and node_name in self[node_type]:
            return self[node_type][node_name]
        return self._name_index.get((node_type, node_name))

    def rename_node_to(self, node: NodeBase, new_name: str):
        node_type = type(node)
        try:
            del self._name_index[(node_type, node.name)]
        except:
            pass
        try:
            del self[node_type][node.name]
        except:
            pass
        try:
            orig_node = self._name_index.get((node_type, new_name))
            del self._uuid_dict[orig_node.uuid]
            # potential issue when renaming root node to a child_node?
            # it will be presented but not accessible?
        except:
            pass
        node.name = new_name
        self.add_node(node)

    def get_node_by_uuid(self, uuid: str) -> NodeBase:
        node_type, node_name = self._uuid_dict[uuid]
        return self._name_index.get((node_type, node_name))


class Container:
    _key_types = []  # [ type[context_key_node] ]
    _action_types = defaultdict(
        list
    )  # {type[context_key_node]:  [ type[action_node] ]}
    _type_agencies = {}  # {type: handler}
    _modules = set()

    def __init__(self) -> None:
        self.actions: list[ActionNode] = []
        self.contexts: dict[ContextKeyNode, DataContext] = defaultdict(
            lambda: DataContext(self)
        )
        self.context_keys = DataContext(self)
        self.current_key: ContextKeyNode = GCK

    @property
    def CurrentContext(self) -> DataContext:
        return self.contexts[self.current_key]

    @property
    def ActionsInCurrentContext(self) -> list[ActionNode]:
        return filter(lambda a: a.context_key is self.current_key, self.actions)

    def get_node_of_type_for(
        self, context_key: ContextKeyNode, node_name: str, node_type: type[NodeBase]
    ) -> NodeBase | None:
        context = self.get_context(context_key)
        if node := context.get_node_of_type(node_name, node_type):
            return node
        elif (context_key is not GCK) and (
            node := self.contexts[GCK].get_node_of_type(node_name, node_type)
        ):
            return node
        elif node := self.context_keys.get_node_of_type(node_name, node_type):
            return node
        else:
            return None

    def get_node_of_type(
        self, node_name: str, node_type: type[NodeBase]
    ) -> NodeBase | None:
        return self.get_node_of_type_for(self.current_key, node_name, node_type)

    def activate_context(self, context_key: ContextKeyNode) -> DataContext:
        self.current_key = context_key
        return self.CurrentContext

    def get_context(self, context_key: ContextKeyNode) -> DataContext:
        return self.contexts[context_key]

    def remove_context_key(self, context_key: ContextKeyNode):
        del self.context_keys[type(context_key)][context_key.name]
        if context_key in self.contexts:
            del self.contexts[context_key]
        self.actions = [
            action for action in self.actions if action.context_key is not context_key
        ]

    def _get_value_of_annotation(
        self, ann: type | GenericAlias | UnionType, pre_value: Any
    ):
        # Support many typing annotation variants and convert `pre_value` accordingly.
        from typing import Any as _Any

        if pre_value is None:
            return None

        origin = get_origin(ann)
        args = get_args(ann)

        # list[...] - allow partial node resolution (skip missing)
        if origin is list or (isinstance(ann, GenericAlias) and getattr(ann, "__name__", None) == "list"):
            elem_type = args[0] if args else _Any
            value = []
            for c in pre_value:
                try:
                    v = self._get_value_of_annotation(elem_type, c)
                except NodeNotFoundError:
                    continue
                value.append(v)
            return value

        # tuple[...] - positional types, require matching
        if origin is tuple or (isinstance(ann, GenericAlias) and getattr(ann, "__name__", None) == "tuple"):
            if not args:
                return tuple(pre_value)
            return [self._get_value_of_annotation(a, c) for a, c in zip(args, pre_value)]

        # set[...] - convert elements
        if origin is set or (isinstance(ann, GenericAlias) and getattr(ann, "__name__", None) == "set"):
            elem_type = args[0] if args else _Any
            return set(self._get_value_of_annotation(elem_type, c) for c in pre_value)

        # dict[key_type, value_type]
        if origin is dict or (isinstance(ann, GenericAlias) and getattr(ann, "__name__", None) == "dict"):
            key_t, val_t = (args + (_Any, _Any))[:2]
            return {self._get_value_of_annotation(key_t, k): self._get_value_of_annotation(val_t, v) for k, v in pre_value.items()}

        # Union / Optional
        if origin is Union or isinstance(ann, UnionType) or (getattr(ann, "__args__", None) and origin is None):
            union_args = args if args else getattr(ann, "__args__", ())
            for t in union_args:
                if t is type(None):
                    if pre_value is None:
                        return None
                    continue
                try:
                    if isinstance(pre_value, t):
                        return pre_value
                    new_value = self._get_value_of_annotation(t, pre_value)
                    if new_value != pre_value:
                        return new_value
                except NodeNotFoundError:
                    # propagate node-not-found only if no union option matches
                    continue
                except Exception:
                    continue
            raise TypeError(f"Value '{pre_value}' not in the union types '{ann}'.")

        # Enum
        try:
            if issubclass(ann, Enum):
                if isinstance(pre_value, Enum):
                    return pre_value
                return ann[pre_value]
        except Exception:
            pass

        # DataNode lookup
        try:
            if issubclass(ann, DataNode):
                if (value := self.get_node_of_type(node_name=pre_value, node_type=ann)) is None:
                    raise NodeNotFoundError(f"Node '{pre_value}' of '{ann.__name__}' not found.")
                return value
        except Exception:
            # if issubclass check failed because ann wasn't a class, ignore
            pass

        # Registered type agency
        if ann in Container._type_agencies:
            return Container._type_agencies[ann](pre_value)

        # Fallback: if annotation is Any or unhandled type, return pre_value
        try:
            if ann is _Any:
                return pre_value
        except Exception:
            pass

        return pre_value

    def prepare_params_for_action(
        self, signature: inspect.Signature | dict, construct_config: dict
    ) -> dict:
        params = {}
        if isinstance(signature, inspect.Signature):
            for key, param in signature.parameters.items():
                if key == "self":
                    continue
                value = construct_config.get(key, param.default)
                if value is inspect._empty:
                    # not provided and no default
                    raise Exception(f"Parameter '{key}' not provided.")

                params[key] = self._get_value_of_annotation(param.annotation, value)
        else:  # signature is a dict, SAB
            for subact_name, subact_signature in signature.items():
                subact_config = construct_config.get(subact_name, {})
                params[subact_name] = self.prepare_params_for_action(
                    subact_signature, subact_config
                )
        return params

    def get_save_config(self):
        return {
            "actions": [action.get_save_config() for action in self.actions],
            "contexts": [
                n_o.get_save_config() for n_t, n_n, n_o in self.context_keys.NodeIter
            ],
            # add combo_action definitions, combo_action = action1 + action2 + action3 ...
            # able to define pre-defined parameters and user-input parameters
            # only available in current project, but can be saved as a template
        }

    @classmethod
    def parse_save_config(cls, config: dict) -> "Container":
        container = Container()
        nodes = {}

        g_nodes = config.get("contexts") or config.get("global_nodes") or []
        for data_config in g_nodes:
            cls_path = data_config["_class_"]
            del data_config["_class_"]
            uuid = data_config["_uuid_"]
            del data_config["_uuid_"]

            try:
                data_class: type[DataNode] = Container.GetClass(cls_path)
            except AttributeError:  # not found
                continue  # TODO: log
            data_node = data_class(name="[Default]", uuid=uuid)
            data_node.apply_construct_config(data_config)

            nodes[uuid] = data_node

            container.context_keys.add_node(data_node)

        actions = config.get("actions") or []
        for act_config in actions:
            cls_path = act_config["_class_"]
            del act_config["_class_"]
            uuid = act_config["_uuid_"]
            del act_config["_uuid_"]

            try:
                act_class: type[ActionNode] = Container.GetClass(cls_path)
            except AttributeError:
                continue  # TODO: log

            if "_context_" in act_config:
                if act_config["_context_"] not in nodes:
                    continue
                context_key = nodes[act_config["_context_"]]
                del act_config["_context_"]
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
        Container._modules.add(module)
        return getattr(module, class_name)

    @staticmethod
    def RegisterGlobalDataType(node_type: type[ContextKeyNode] | str):
        Container._key_types.append(node_type)

    @staticmethod
    def GetGlobalDataTypes() -> list[type[ContextKeyNode] | str]:
        return Container._key_types

    @staticmethod
    def RegisterContextAction(
        context_type: type[ContextKeyNode], action_type: type[ActionNode] | str
    ):
        Container._action_types[context_type].append(action_type)

    @staticmethod
    def GetContextActionTypes(
        context_type: type[ContextKeyNode],
    ) -> list[type[ActionNode] | str]:
        return Container._action_types[context_type]

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
