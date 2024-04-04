import re, yaml
from os import path
from dac.core import Container, NodeBase

def use_plugin(setting_fpath: str, clean: bool=True, dac_win=None):
    alias_pattern = re.compile("^/(?P<alias_name>.+)/(?P<rest>.+)")
    def get_node_type(cls_path: str) -> str | type[NodeBase]:
        if cls_path[0]=="[" and cls_path[-1]=="]":
            return cls_path # just str as section string
        
        if (rst:=alias_pattern.search(cls_path)):
            cls_path = alias[rst['alias_name']]+"."+rst['rest']

        try:
            return Container.GetClass(cls_path)
        except AttributeError:
            dac_win.message(f"Module `{cls_path}` not found")
            return None
        
    if clean:
        Container._action_types.clear()
        Container._key_types.clear()
        # quick_tasks and quick_actions are always overwritten

    with open(setting_fpath, mode="r", encoding="utf8") as fp:
        setting: dict = yaml.load(fp, Loader=yaml.FullLoader)
        if not setting: return

        if (inherit_rel_path:=setting.get('inherit')) is not None:
            use_plugin(path.join(path.dirname(setting_fpath), inherit_rel_path), clean=False, dac_win=dac_win)

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

        if not hasattr(dac_win, "show"): # web-based cannot use PyQt5 and the tasks
            return

        for ats, tss in setting.get("quick_tasks", {}).items(): # action_type_string, task_string_s
            action_type = get_node_type(ats)
            if not action_type: continue
            action_type.QUICK_TASKS = [] # make superclass.QUICK_TASKS hidden
            for tts, name, *rest in tss: # task_type_string, name, *rest
                task_type = get_node_type(tts)
                if not task_type: continue
                task = task_type(dac_win=dac_win, name=name, *rest)
                action_type.QUICK_TASKS.append(task)

        for dts, ass in setting.get("quick_actions", {}).items(): # data_type_string, action_string_s
            data_type = get_node_type(dts)
            if not data_type: continue
            data_type.QUICK_ACTIONS = []
            for ats, dpn, opd in ass: # action_type_string, data_param_name, other_params_dict
                action_type = get_node_type(ats)
                if not action_type: continue
                data_type.QUICK_ACTIONS.append((action_type, dpn, opd))