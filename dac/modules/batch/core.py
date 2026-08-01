import os
import copy
from typing import Iterator, Optional, Callable

from dac.core import DataNode, ContextKeyNode, ActionNode, Container
from dac.core.data import SimpleDefinition
from dac.core.actions import PAB, VAB, ProcessActionBase
from dac.core.exceptions import ActionExecutionError


class ContextSpec:
    def __init__(self, name: str, params: dict = None, context_attrs: dict = None):
        self.name = name
        self.params = params or {}
        self.context_attrs = context_attrs or {}


class BatchAction(ProcessActionBase):
    CAPTION = "Batch action"

    _cancelled: bool = False
    _cancel_check: Optional[Callable[[], bool]] = None
    _current_spec_index: int = 0

    def __call__(self,
                 template: SimpleDefinition = None,
                 input_paths: list[str] = [],
                 input_pattern: str = "",
                 stop_on_error: bool = True,
                 used: bool = False,
                 ) -> dict:
        if used:
            self.message("Batch already completed, skipping")
            return {"skipped": True, "reason": "used"}

        if template is None:
            ctx = self._ensure_template_context()
            if ctx:
                self.message(
                    f"Template context '{ctx.name}' created. "
                    "Add actions and debug, then set template reference and re-run."
                )
                return {"template_created": True, "template_name": ctx.name}
            return {"error": "Failed to create template context"}

        specs = list(self._iter_context_specs(input_paths, input_pattern))
        if not specs:
            self.message("No context specs generated. Check input_paths or input_pattern.")
            return {"error": "no_contexts"}

        n = len(specs)
        self.message(f"Starting batch: {n} context(s)")
        results = {}
        processed = 0

        for i, spec in enumerate(specs):
            if self._is_cancelled():
                self._cancelled = True
                break

            self._current_spec_index = i
            self.progress(i, n)
            self.message(f"[{i + 1}/{n}] Context: {spec.name}")

            new_ctx = self._find_or_create_context(spec)
            if new_ctx is None:
                self.message(f"Context '{spec.name}' skipped (already exists)")
                results[spec.name] = {"status": "skipped", "reason": "exists"}
                continue

            try:
                actions = self._create_context_actions(spec, new_ctx, template)
                self.container.actions.extend(actions)
                self._run_context_actions(new_ctx, actions)
                results[spec.name] = {"status": "ok", "n_actions": len(actions)}
                processed += 1
            except ActionExecutionError:
                self.message(f"Context '{spec.name}' failed")
                results[spec.name] = {"status": "error"}
                if stop_on_error:
                    break
            except Exception as e:
                self.message(f"Context '{spec.name}' unexpected error: {e}")
                results[spec.name] = {"status": "error"}
                if stop_on_error:
                    break

        if not self._cancelled:
            self._construct_config["used"] = True
            self.message(f"Batch complete: {processed}/{len(specs)} succeeded")

        return results

    def _is_cancelled(self) -> bool:
        if self._cancel_check and self._cancel_check():
            return True
        return False

    def _ensure_template_context(self) -> Optional[SimpleDefinition]:
        template_name = f"{self.name}_TEMPLATE"
        ck_ctx = self.container.context_keys

        existing = ck_ctx.get_node_of_type(template_name, SimpleDefinition)
        if existing is not None:
            self._construct_config["template"] = template_name
            return existing

        template_ctx = SimpleDefinition(name=template_name)
        ck_ctx.add_node(template_ctx)
        self._construct_config["template"] = template_name
        return template_ctx

    def _iter_context_specs(self, input_paths: list[str] = None,
                              input_pattern: str = None) -> Iterator[ContextSpec]:
        if input_pattern:
            import glob as _glob
            matched = _glob.glob(input_pattern)
            if matched:
                input_paths = list(matched)

        if not input_paths:
            return

        for fpath in input_paths:
            name = os.path.splitext(os.path.basename(str(fpath)))[0]
            yield ContextSpec(name=name)

    def _find_or_create_context(self, spec: ContextSpec) -> Optional[SimpleDefinition]:
        ck_ctx = self.container.context_keys
        existing = ck_ctx.get_node_of_type(spec.name, SimpleDefinition)
        if existing is not None:
            return None

        new_ctx = SimpleDefinition(name=spec.name, **spec.context_attrs)
        ck_ctx.add_node(new_ctx)
        return new_ctx

    def _create_context_actions(self, spec: ContextSpec,
                                  new_ctx: SimpleDefinition,
                                  template: SimpleDefinition) -> list[ActionNode]:
        template_actions = self.container.get_actions_for_context(template)
        if not template_actions:
            self.message(f"No actions found in template '{template.name}'")

        actions = []
        for act in template_actions:
            cloned = self._clone_action(act, new_ctx, spec.params)
            cloned.name = act.name
            if cloned.out_name is None:
                cloned.out_name = act.out_name
            cloned.container = self.container
            actions.append(cloned)

        return actions

    def _clone_action(self, action: ActionNode, new_context_key: ContextKeyNode,
                       param_overrides: dict = None) -> ActionNode:
        act_class = type(action)
        cfg = copy.deepcopy(action._construct_config)

        if param_overrides:
            class_name = act_class.__name__
            if class_name in param_overrides:
                cfg.update(param_overrides[class_name])

        new_action = act_class(context_key=new_context_key)
        new_action.apply_construct_config(cfg)
        return new_action

    def _run_context_actions(self, ctx: ContextKeyNode, actions: list[ActionNode]):
        container = self.container
        old_key = container.current_key
        container.activate_context(ctx)

        try:
            for i, action in enumerate(actions):
                if self._is_cancelled():
                    self._cancelled = True
                    break

                if isinstance(action, VAB):
                    continue

                if isinstance(action, PAB):
                    action._progress = self._make_sub_progress(i, len(actions))
                    action._message = self._make_sub_message(ctx.name)

                params = container.prepare_params_for_action(
                    action._SIGNATURE, action._construct_config
                )

                action.pre_run()
                rst = action(**params)
                action.post_run()

                if isinstance(rst, DataNode):
                    container.CurrentContext.add_node(rst)
                elif isinstance(rst, list):
                    for r in rst:
                        if isinstance(r, DataNode):
                            container.CurrentContext.add_node(r)

                action.status = ActionNode.ActionStatus.COMPLETE
        finally:
            container.activate_context(old_key)

    def _make_sub_progress(self, idx, total):
        def progress(i, n):
            self.progress(idx * 100 + i, total * 100)

        return progress

    def _make_sub_message(self, ctx_name):
        def message(s):
            self.message(f"[{ctx_name}] {s}")

        return message
