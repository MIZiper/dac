"""Handles scenario loading, and register data types / action available for context.

Scenario use YAML files to define what data types can be added as contexts,
and the available actions can be used under corresponding context.

The Data and Action are defined under individual modules.
Scenario is just a layout definition, any action types can be added to context if you know the path.
"""

import importlib, re, yaml
from os import path
from dac.core import Container, NodeBase
from dac.core.exceptions import ScenarioError
from dac.core.logging import get_logger

_logger = get_logger("scenario")
_alias_pattern = re.compile("^/(?P<alias_name>.+)/(?P<rest>.+)")

def get_nodetype_path(node_type: type[NodeBase]):
    return f"{node_type.__module__}.{node_type.__qualname__}"

def _resolve_inherit_path(setting_fpath: str, inherit_path: str) -> str:
    if inherit_path.startswith('/'):
        inner = inherit_path[1:]
        if '/' in inner:
            pkg_name, sub_path = inner.split('/', 1)
        else:
            pkg_name, sub_path = inner, ''
        try:
            pkg = importlib.import_module(pkg_name)
            if hasattr(pkg, '__path__'):
                pkg_dir = pkg.__path__[0]
            else:
                pkg_dir = path.dirname(pkg.__file__)
            return path.join(pkg_dir, sub_path)
        except (ImportError, AttributeError, IndexError) as e:
            _logger.warning("Cannot resolve package '%s' from inherit path '%s': %s", pkg_name, inherit_path, e)
    return path.join(path.dirname(setting_fpath), inherit_path)


def use_scenario(setting_fpath: str, clean: bool=True, dac_win=None):
    def get_node_type(cls_path: str) -> str | type[NodeBase]:
        if cls_path[0]=="[" and cls_path[-1]=="]":
            return cls_path # just str as section string
        
        if (rst:=_alias_pattern.search(cls_path)):
            cls_path = alias[rst['alias_name']]+"."+rst['rest']

        try:
            return Container.GetClass(cls_path)
        except (AttributeError, ModuleNotFoundError) as e:
            if dac_win:
                dac_win.message(f"Module `{cls_path}` not found: {e}")
            else:
                _logger.warning("Module `%s` not found: %s", cls_path, e)
            return None
        
    if clean:
        Container._action_types.clear()
        Container._key_types.clear()
        Container._drop_action_map.clear()
        # quick_tasks and quick_actions are always overwritten

    try:
        with open(setting_fpath, mode="r", encoding="utf8") as fp:
            setting: dict = yaml.load(fp, Loader=yaml.FullLoader)
    except FileNotFoundError:
        msg = f"Scenario file not found: {setting_fpath}"
        _logger.error(msg)
        if dac_win:
            dac_win.message(msg)
        return
    except yaml.YAMLError as e:
        msg = f"Invalid YAML in scenario file '{setting_fpath}': {e}"
        _logger.error(msg)
        if dac_win:
            dac_win.message(msg)
        return

    if not setting:
        return

    inherited_qa = None
    if (inherit_rel_path:=setting.get('inherit')) is not None:
        inherited_qa = use_scenario(_resolve_inherit_path(setting_fpath, inherit_rel_path), clean=False, dac_win=dac_win)

    alias = setting.get('alias', {})

    for gdts in setting.get('data', {}).get("_", []): # global_data_type_string
        node_type = get_node_type(gdts)
        if node_type: Container.RegisterGlobalDataType(node_type)

    for dts, catss in setting.get('actions', {}).items(): #  data_type_string, context_action_type_string_s
        if dts=="_": # global_context
            for cats in catss:
                node_type = get_node_type(cats)
                if node_type: Container.RegisterGlobalContextAction(node_type)
        else:
            data_type = get_node_type(dts)
            if not data_type: continue
            for cats in catss:
                action_type = get_node_type(cats)
                if action_type: Container.RegisterContextAction(data_type, action_type)

    quick_actions = []
    for dts, ass in setting.get("quick_actions", {}).items(): # data_type_string, action_string_s
        data_type = get_node_type(dts)
        if not data_type: continue
        data_type.QUICK_ACTIONS = []
        idx = -1
        for ats, dpn, opd, *rest in ass: # action_type_string, data_param_name, other_params_dict[, mode]
            action_type = get_node_type(ats)
            if not action_type: continue
            idx += 1
            mode = rest[0] if rest else False
            data_type.QUICK_ACTIONS.append((action_type, dpn, opd, mode))
            quick_actions.append((
                get_nodetype_path(data_type), # str
                get_nodetype_path(action_type), # str # this is actually optional?
                action_type.CAPTION, # str
                idx, # int
                mode, # bool | str
            ))

    for ext, entries in setting.get("drop_actions", {}).items(): # ext_string, [[action_type_string, path_param_name, other_params, mode?], ...]
        for ats, path_param_name, other_params, *rest in entries:
            action_type = get_node_type(ats)
            if not action_type: continue
            Container.RegisterDropAction(ext, action_type, path_param_name, other_params)

    if not hasattr(dac_win, "show"): # web-based cannot use PyQt5 and the tasks
        # return flat quick_actions
        if inherited_qa:
            return inherited_qa + quick_actions
        return quick_actions

    for ats, tss in setting.get("quick_tasks", {}).items(): # action_type_string, task_string_s
        action_type = get_node_type(ats)
        if not action_type: continue
        action_type.QUICK_TASKS = [] # make superclass.QUICK_TASKS hidden
        for tts, name, *rest in tss: # task_type_string, name, *rest
            task_type = get_node_type(tts)
            if not task_type: continue
            task = task_type(dac_win=dac_win, name=name, *rest)
            action_type.QUICK_TASKS.append(task)

    for ats, (tts, name, *rest) in setting.get("default_task", {}).items(): # action_type_string, task_type_string
        action_type = get_node_type(ats)
        if not action_type: continue
        action_type.DEFAULT_TASK = None
        task_type = get_node_type(tts)
        if not task_type: continue
        task = task_type(dac_win=dac_win, name=name, *rest)
        action_type.DEFAULT_TASK = task
