"""
Install HF evaluation library
 `pip install evaluation`
"""
import evaluate

from typing import Any, Optional


def list_hf_evaluation_modules(module_type: str = "metric", include_community: str = False, with_details: str = True) -> list[dict[str, Any]]:
    """
    returns the list of all evaluation options under the module type
    :param module_type: 'metric', 'comparison', or 'measurement'
    :param include_community:
    :param with_details:
    :return: evaluation options list

    [{'name': 'mcnemar', 'type': 'comparison', 'community': False, 'likes': 1},
     {'name': 'exact_match', 'type': 'comparison', 'community': False, 'likes': 0}]
    """
    return evaluate.list_evaluation_modules(
      module_type=module_type,
      include_community=include_community,
      with_details=with_details)


def load_evaluation_fn(module_type: Optional[str], fn_name: str = "accuracy", verbose: bool = False) -> Any:
    """
    Any metric, comparison, or measurement is loaded with the evaluate.load function:

    :param module_type: (str) 'metric', 'comparison', or 'measurement'
    :param fn_name: (str) name of evaluation method
    :param verbose: (bool) print details of loaded evaluation function
    :return: evaluation function
    """
    eval_fn = evaluate.load(fn_name)
    if verbose:
        print("Description: ", eval_fn.description)
        print("Features: ", eval_fn.features)
    return eval_fn


def compute_eval(loaded_eval_fn, **features) -> dict[str, Any]:
    """
    :param loaded_eval_fn: evaluation function
    :param features: kwargs of features needed for eval_fn computation,
        Usage:
            features: references=[0,1,0,1], predictions=[1,0,0,1]
            eval_fn: accuracy
            accuracy.compute(references=[0,1,0,1], predictions=[1,0,0,1])
    :return: metric results
    """
    return loaded_eval_fn.compute(**features)


def combined_eval(eval_funcs: list[str], **features) -> dict[str, Any]:
    """
    The combine function accepts both the list of names of the metrics as well as an instantiated modules.
    The compute call then computes each metric.
    :param fn_names: list(str):
    :param features: kwargs for computed features
    :return:
    """
    combined_evaluators = evaluate.combine(eval_funcs)
    return combined_evaluators.compute(**features)
