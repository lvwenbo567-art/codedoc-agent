import pytest

from context_engineering.context_budget import ContextBudgetConfig


def test_valid_budget() -> None:
    ContextBudgetConfig().validate()


@pytest.mark.parametrize(
    "config",
    [
        ContextBudgetConfig(max_context_tokens=0),
        ContextBudgetConfig(reserved_output_tokens=16000),
        ContextBudgetConfig(max_message_tokens=8000, max_evidence_tokens=7000),
    ],
)
def test_invalid_budget_is_rejected(config: ContextBudgetConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
