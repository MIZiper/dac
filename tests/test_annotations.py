import pytest
from typing import Union, Optional, Callable, Dict
from enum import Enum

from dac.core import ActionNode, Container, DataNode, DataContext, ContextKeyNode
from dac.core.exceptions import (
    NodeNotFoundError,
    ActionConfigError,
    ContainerError,
    ClassNotFoundError,
    ContextError,
    TypeAgencyError,
    ActionExecutionError,
    SnippetError,
)


class Color(Enum):
    RED = 1
    GREEN = 2


def test_annotation2config_basic_and_collections():
    assert ActionNode.Annotation2Config(int) == "<int>"
    assert ActionNode.Annotation2Config(str) == "<str>"

    # list[int] -> ['<int>']
    assert ActionNode.Annotation2Config(list[int]) == ["<int>"]

    # tuple[int,str] -> ['<int>','<str>']
    assert ActionNode.Annotation2Config(tuple[int, str]) == ["<int>", "<str>"]

    # dict[str,int] -> {'<str>': '<int>'}
    assert ActionNode.Annotation2Config(dict[str, int]) == {"<str>": "<int>"}

    # set[float]
    assert ActionNode.Annotation2Config(set[float]) == ["<float>"]

    # Union/Optional
    u = ActionNode.Annotation2Config(Union[int, str])
    assert "<int>" in u and "<str>" in u
    u = ActionNode.Annotation2Config(int | str)
    assert "<int>" in u and "<str>" in u

    opt = ActionNode.Annotation2Config(Optional[int])
    assert "<int>" in opt and "None" in opt or "NoneType" in opt

    # Enum -> [key1 | key2 | ...]
    e = ActionNode.Annotation2Config(Color)
    for k in Color:
        assert k.name in e

def test_annotation2config_more():
    assert ActionNode.Annotation2Config(list) == "<list>"

    class MyNode(DataNode):
        pass

    assert ActionNode.Annotation2Config(MyNode) == "<MyNode>"
    assert ActionNode.Annotation2Config(list[MyNode]) == ["<MyNode>"]


def test_container_get_value_of_annotation_basic_conversion_and_collections():
    c = Container()

    # list[int]
    res = c._get_value_of_annotation(list[int], [1, 2, 3])
    assert res == [1, 2, 3]

    # tuple[int,str]
    res = c._get_value_of_annotation(tuple[int, str], (1, "a"))
    assert res[0] == 1 and res[1] == "a"

    # dict[str,int]
    res = c._get_value_of_annotation(dict[str, int], {"a": 1, "b": 2})
    assert res == {"a": 1, "b": 2}

    # set[float]
    res = c._get_value_of_annotation(set[float], {1.0, 2.0})
    assert isinstance(res, set) and 1.0 in res


def test_container_get_value_of_annotation_enum_and_union_and_optional():
    c = Container()
    # Enum by name
    res = c._get_value_of_annotation(Color, "RED")
    assert res == Color.RED

    # Union[int,str]
    res = c._get_value_of_annotation(Union[int, str], "x")
    assert res == "x"

    # Optional -> None
    res = c._get_value_of_annotation(Optional[int], None)
    assert res is None


def test_container_get_value_of_annotation_datanode_lookup():
    c = Container()

    class MyNode(DataNode):
        pass

    node = MyNode(name="node1")
    c.context_keys.add_node(node)

    res = c._get_value_of_annotation(MyNode, "node1")
    assert res is node


def test_container_registered_type_agency():
    c = Container()

    class Custom:
        pass

    def agency(x):
        return int(x)

    Container.RegisterNodeTypeAgency(Custom, agency)
    res = c._get_value_of_annotation(Custom, "123")
    assert res == 123


# ---- New error handling tests ----


def test_node_not_found_error_is_typed():
    with pytest.raises(NodeNotFoundError):
        class MyNode(DataNode):
            pass
        node = MyNode(name="test")
        node.parent_until(ContextKeyNode)


def test_rename_node_to_keyerror():
    ctx = DataContext(Container())
    node = DataNode(name="test")
    ctx.add_node(node)

    ctx.rename_node_to(node, "renamed")

    assert node.name == "renamed"
    renamed = ctx.get_node_of_type("renamed", DataNode)
    assert renamed is node


def test_action_apply_config_validation():
    class TestAction(ActionNode):
        CAPTION = "Test"
        def __call__(self, a: int, b: str = "default"):
            pass

    from dac.core import GCK
    action = TestAction(context_key=GCK)
    action.apply_construct_config({"a": 1})
    assert action.status == ActionNode.ActionStatus.CONFIGURED

    action2 = TestAction(context_key=GCK)
    with pytest.raises(ActionConfigError):
        action2.apply_construct_config({"unknown_param": 42})


def test_prepare_params_raises_config_error():
    class TestAction(ActionNode):
        CAPTION = "Test"
        def __call__(self, required_param: int, optional: str = "default"):
            pass

    from dac.core import GCK
    action = TestAction(context_key=GCK)
    container = Container()

    with pytest.raises(ActionConfigError):
        container.prepare_params_for_action(action._SIGNATURE, {})


def test_type_agency_error_propagation():
    c = Container()

    class Custom:
        pass

    def failing_agency(x):
        raise ValueError("Agency failed")

    Container.RegisterNodeTypeAgency(Custom, failing_agency)

    with pytest.raises(TypeAgencyError):
        c._get_value_of_annotation(Custom, "test")


def test_datanode_lookup_raises_node_not_found():
    c = Container()

    class MyNode(DataNode):
        pass

    with pytest.raises(NodeNotFoundError):
        c._get_value_of_annotation(MyNode, "nonexistent")


def test_action_status_failed_enum():
    assert ActionNode.ActionStatus.FAILED == -1
    assert ActionNode.ActionStatus.FAILED.name == "FAILED"


def test_exception_hierarchy():
    err = NodeNotFoundError("test")
    assert isinstance(err, Exception)
    assert isinstance(err, NodeNotFoundError)

    err = ActionConfigError("test")
    assert isinstance(err, ActionConfigError)

    err = TypeAgencyError("test")
    assert isinstance(err, TypeAgencyError)


def test_snippet_exec_error():
    with pytest.raises(SnippetError):
        from dac.core.snippet import exec_script
        exec_script("1 / 0")


def test_snippet_none_or_empty():
    from dac.core.snippet import exec_script
    exec_script("")
    exec_script(None)


def test_apply_config_validates_unknown_params():
    class TestAction(ActionNode):
        CAPTION = "Test"
        def __call__(self, x: float = 1.0):
            pass

    from dac.core import GCK
    action = TestAction(context_key=GCK)

    with pytest.raises(ActionConfigError):
        action.apply_construct_config({"name": "ok", "out_name": "ok", "y": 2.0})
