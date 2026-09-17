# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.models.deepseek_v4_dspark import DSparkDeepseekV4ForCausalLM


def test_confidence_logits_delegates_to_inner_compute_confidence():
    hidden_states = torch.randn(2, 4)
    markov_embed = torch.randn(2, 3)
    expected = torch.randn(2)
    compute_confidence = MagicMock(return_value=expected)
    wrapper = SimpleNamespace(
        model=SimpleNamespace(compute_confidence=compute_confidence),
    )

    actual = DSparkDeepseekV4ForCausalLM.confidence_logits(
        wrapper,
        hidden_states,
        markov_embed,
    )

    assert actual is expected
    compute_confidence.assert_called_once_with(hidden_states, markov_embed)
