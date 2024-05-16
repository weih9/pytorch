# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.export
import torch.export._trace

log = logging.getLogger(__name__)

__all__ = ["report_exportability"]

def _generate_inputs_for_submodules(
    model: torch.nn.Module,
    target_submodules: Iterable[str],
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Generate inputs for targeting submdoules in the given model. Note that if two submodules refer to the same obj, this
    function doesn't work.

    Args:
        model: root model.
        inputs: inputs to the root model.
        target_submodules: submodules that we want to generate inputs for.

    Returns:
        A dict that maps from submodule name to its inputs.
    """

    handles = []
    results = {}
    submodule_to_names = dict((mod, name) for name, mod in model.named_modules())

    def pre_forward(module, module_args, module_kwargs):
        results[submodule_to_names[module]] = (module_args, module_kwargs)

    try:
        for name, mod in model.named_modules():
            if name in target_submodules:
                handles.append(
                    mod.register_forward_pre_hook(pre_forward, with_kwargs=True)
                )
        model(*args, **kwargs)
    except Exception as e:
        warnings.warn(
            f"Failed to generate submodule inputs because of the following error:\n{e}"
        )
    finally:
        for h in handles:
            h.remove()
    return results


def report_exportability(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    strict: bool = True,
    pre_dispatch: bool = False) -> Dict[str, Optional[Exception]]:
    """
    Report exportability issues for a module in one-shot.

    Args:
        mod: root module.
        args: args to the root module.
        kwargs: kwargs to the root module.
    Returns:
        A dict that maps from submodule name to the exception that was raised when trying to export it.
        `None` means the module is exportable.
    Sample output:
        {
            '': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_1': UnsupportedOperatorException(func=<OpOverload(op='testlib.op_missing_meta', overload='default')>),
            'submod_2': None
        }
    """

    kwargs = kwargs or {}

    all_submod_names = [name for name, _ in mod.named_modules() if name != ""]
    submod_inputs = _generate_inputs_for_submodules(
        mod,
        all_submod_names,
        args,
        kwargs
    )

    result = {}
    def try_export(module, module_name, args, kwargs):
        nonlocal submod_inputs, result, strict, pre_dispatch

        if args is not None or kwargs is not None:
            try:
                torch.export._trace._export(
                    module,
                    args,
                    kwargs,
                    strict=strict,
                    pre_dispatch=pre_dispatch,
                )
                result[module_name] = None
                log.info(f"Successfully exported `{module_name}`")
                return
            except Exception as e:
                short_msg = repr(e).split("\n")[0]
                log.warning(f"Failed exporting `{module_name}` with exception: {short_msg}")
                result[module_name] = e

        for name, submod in module.named_children():
            sub_module_name = name if module_name == "" else f"{module_name}.{name}"

            submod_args, submod_kwargs = submod_inputs.get(sub_module_name, (None, None))

            try_export(submod, sub_module_name, submod_args, submod_kwargs)

        return

    try_export(mod, "", args, kwargs)

    return result
