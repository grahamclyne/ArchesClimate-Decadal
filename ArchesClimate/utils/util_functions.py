# helper for rollout
import torch
from tensordict.tensordict import TensorDict


def filter_unique_third_tuple_values(input_dict):
    seen_third_values = {}
    result_dict = {}

    for key, tuple_value in input_dict.items():
        third_value = tuple_value[2]

        if third_value not in seen_third_values:
            seen_third_values[third_value] = True
            result_dict[key] = tuple_value
    return result_dict


def apply_iszero(td):
    return TensorDict(
        {key: ~(value == 0) for key, value in td.items()}, batch_size=td.batch_size
    )


def apply_zero_to_nan(td: TensorDict) -> TensorDict:
    """Convert zeros to NaN values in a TensorDict.

    Args:
        td (TensorDict): Input tensor dictionary

    Returns:
        TensorDict: Tensor dictionary with zeros converted to NaN
    """
    return TensorDict(
        {
            key: torch.where(
                value == 0,
                torch.tensor(float("nan"), dtype=value.dtype, device=value.device),
                value,
            )
            for key, value in td.items()
        },
        batch_size=td.batch_size,
    )
