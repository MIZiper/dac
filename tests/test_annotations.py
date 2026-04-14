import pytest
from typing import Union, Optional, Callable, Dict
from enum import Enum

from dac.core import ActionNode, Container, DataNode


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

    # dict[str,int] -> {'key': '<str>', 'value': '<int>'}
    assert ActionNode.Annotation2Config(dict[str, int]) == {"<str>": "<int>"}

    # set[float]
    assert ActionNode.Annotation2Config(set[float]) == ["<float>"]

    # Union/Optional
    u = ActionNode.Annotation2Config(Union[int, str])
    assert "<int>" in u and "<str>" in u

    opt = ActionNode.Annotation2Config(Optional[int])
    assert "<int>" in opt and "None" in opt or "NoneType" in opt

    # Enum
    assert ActionNode.Annotation2Config(Color) == "<Color>"


def test_annotation2config_callable_and_any():
    c = ActionNode.Annotation2Config(Callable[[int, str], bool])
    assert isinstance(c, dict) and "callable" in c or "return" in str(c)

    assert ActionNode.Annotation2Config(object) == "<object>" or isinstance(ActionNode.Annotation2Config(object), str)


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
