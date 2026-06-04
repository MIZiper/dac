import pytest
from dac.core import Container, DataNode, GCK
from dac.core.actions import ActionBase, A1, A2, A1A2, SAB


def _simulate_quick_action_logic(act_type, data_param_name, data_nodes, other_params,
                                   mode=False, container=None):
    """Simulate the key logic from cb_quickaction_gen to produce params dict."""
    from typing import get_origin, get_args, Union as _Union
    import inspect

    def _is_list_annotation(ann):
        if ann is inspect._empty:
            return False
        origin = get_origin(ann)
        if origin is list:
            return True
        if origin is _Union:
            for t in get_args(ann):
                if t is type(None):
                    continue
                if _is_list_annotation(t):
                    return True
        return False

    is_sab = isinstance(act_type._SIGNATURE, dict)
    sub_names_with_param = []
    if is_sab:
        for sub_type in act_type._SEQUENCE:
            if data_param_name in sub_type._SIGNATURE.parameters:
                sub_names_with_param.append(sub_type.__name__)
        param = None
        for sub_type in act_type._SEQUENCE:
            if (p := sub_type._SIGNATURE.parameters.get(data_param_name)):
                param = p
                break
    else:
        param = act_type._SIGNATURE.parameters.get(data_param_name)
    is_list = (
        param is not None
        and _is_list_annotation(param.annotation)
    )
    value = data_nodes if is_list else (data_nodes[0] if data_nodes else None)

    do_run = mode != "create"
    do_save = mode is True or mode == "create"

    if is_sab:
        params = dict(other_params)
        for sub_name in sub_names_with_param:
            sub_cfg = params.setdefault(sub_name, {})
            sub_cfg[data_param_name] = value
    else:
        params = {data_param_name: value, **other_params}

    def _value_to_persistable(v, ctx=None):
        if isinstance(v, DataNode):
            if ctx is None:
                return v.name
            return ctx.get_qualified_name(v)
        if isinstance(v, list):
            return [_value_to_persistable(x, ctx) for x in v]
        if isinstance(v, dict):
            return {k: _value_to_persistable(x, ctx) for k, x in v.items()}
        return v

    if do_save:
        ctx = container.CurrentContext if container else None
        persist_params = {k: _value_to_persistable(v, ctx) for k, v in params.items()}
    else:
        persist_params = None

    return {
        "params": params,
        "is_sab": is_sab,
        "sub_names_with_param": sub_names_with_param,
        "is_list": is_list,
        "value": value,
        "do_run": do_run,
        "do_save": do_save,
        "persist_params": persist_params,
    }


# --- regular action tests ---

def test_quick_action_regular_params():
    nodes = [DataNode(name="n1"), DataNode(name="n2")]
    result = _simulate_quick_action_logic(A1, "sec", nodes, {})
    assert result["is_sab"] is False
    assert result["is_list"] is False
    assert result["params"] == {"sec": nodes[0]}


def test_quick_action_regular_params_list():
    class ListAction(ActionBase):
        CAPTION = "Test"
        def __call__(self, channels: list[DataNode]):
            pass

    nodes = [DataNode(name="n1"), DataNode(name="n2")]
    result = _simulate_quick_action_logic(ListAction, "channels", nodes, {})
    assert result["is_list"] is True
    assert result["params"] == {"channels": nodes}


def test_quick_action_regular_other_params():
    nodes = [DataNode(name="n1")]
    result = _simulate_quick_action_logic(A1, "sec", nodes, {"total": 10})
    assert result["params"] == {"sec": nodes[0], "total": 10}


# --- mode logic tests ---

def test_quick_action_mode_false_run_only():
    nodes = [DataNode(name="n1")]
    result = _simulate_quick_action_logic(A1, "sec", nodes, {}, mode=False)
    assert result["do_run"] is True
    assert result["do_save"] is False
    assert result["persist_params"] is None


def test_quick_action_mode_true_run_and_save():
    c = Container()
    c.CurrentContext.add_node(DataNode(name="x"))
    nodes = [DataNode(name="n1")]
    result = _simulate_quick_action_logic(A1, "sec", nodes, {}, mode=True, container=c)
    assert result["do_run"] is True
    assert result["do_save"] is True
    assert result["persist_params"] == {"sec": "n1"}


def test_quick_action_mode_create_save_only():
    c = Container()
    c.CurrentContext.add_node(DataNode(name="x"))
    nodes = [DataNode(name="n1")]
    result = _simulate_quick_action_logic(A1, "sec", nodes, {}, mode="create", container=c)
    assert result["do_run"] is False
    assert result["do_save"] is True
    assert result["persist_params"] == {"sec": "n1"}


# --- SAB tests ---

def test_quick_action_sab_sub_names_single_match():
    result = _simulate_quick_action_logic(A1A2, "sec", [], {})
    assert result["is_sab"] is True
    assert result["sub_names_with_param"] == ["A1"]


def test_quick_action_sab_sub_names_no_match():
    result = _simulate_quick_action_logic(A1A2, "nonexistent", [], {})
    assert result["is_sab"] is True
    assert result["sub_names_with_param"] == []


def test_quick_action_sab_params_single_sub():
    nodes = [DataNode(name="n1"), DataNode(name="n2")]
    result = _simulate_quick_action_logic(A1A2, "sec", nodes, {})
    assert result["params"] == {"A1": {"sec": nodes[0]}}


def test_quick_action_sab_params_with_other_params():
    nodes = [DataNode(name="n1")]
    result = _simulate_quick_action_logic(
        A1A2, "sec", nodes, {"A2": {"amp": 3.0}}
    )
    assert result["params"] == {"A1": {"sec": nodes[0]}, "A2": {"amp": 3.0}}


def test_quick_action_sab_merge_injects_into_existing_sub_config():
    nodes = [DataNode(name="n1")]
    result = _simulate_quick_action_logic(
        A1A2, "sec", nodes, {"A1": {"total": 10}}
    )
    assert result["params"] == {"A1": {"sec": nodes[0], "total": 10}}


# --- SAB with multiple sub-actions matching ---

class MultiSAB(SAB, seq=[A1, A1]):
    CAPTION = "Multi"


def test_quick_action_sab_multi_sub_injection():
    nodes = [DataNode(name="n1"), DataNode(name="n2")]
    result = _simulate_quick_action_logic(MultiSAB, "sec", nodes, {})
    assert result["sub_names_with_param"] == ["A1", "A1"]
    assert result["params"] == {
        "A1": {"sec": nodes[0]}
    }


# --- SAB persist ---

def test_quick_action_sab_persist():
    c = Container()
    c.CurrentContext.add_node(DataNode(name="root"))
    nodes = [DataNode(name="n1")]
    result = _simulate_quick_action_logic(
        A1A2, "sec", nodes, {"A2": {"amp": 3.0}}, mode=True, container=c
    )
    assert result["persist_params"] == {
        "A1": {"sec": "n1"},
        "A2": {"amp": 3.0},
    }


# --- SAB is_list detection ---

def test_quick_action_sab_is_list_via_first_match():
    class SubA(ActionBase):
        CAPTION = "A"
        def __call__(self, x: DataNode):
            pass

    class SubB(ActionBase):
        CAPTION = "B"
        def __call__(self, x: list[DataNode]):
            pass

    class MixedSAB(SAB, seq=[SubA, SubB]):
        CAPTION = "Mixed"

    nodes = [DataNode(name="n1"), DataNode(name="n2")]
    result = _simulate_quick_action_logic(MixedSAB, "x", nodes, {})
    assert result["is_sab"] is True
    assert result["sub_names_with_param"] == ["SubA", "SubB"]
    assert result["is_list"] is False  # first match is SubA.x: DataNode (single)
    assert result["value"] is nodes[0]


def test_quick_action_sab_is_list_when_first_match_is_list():
    class SubA(ActionBase):
        CAPTION = "A"
        def __call__(self, x: list[DataNode]):
            pass

    class SubB(ActionBase):
        CAPTION = "B"
        def __call__(self, x: DataNode):
            pass

    class MixedSAB2(SAB, seq=[SubA, SubB]):
        CAPTION = "Mixed2"

    nodes = [DataNode(name="n1"), DataNode(name="n2")]
    result = _simulate_quick_action_logic(MixedSAB2, "x", nodes, {})
    assert result["is_list"] is True  # first match is SubA.x: list[DataNode]
    assert result["value"] is nodes
