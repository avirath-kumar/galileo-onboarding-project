from galileo import GalileoScorers
from galileo.stages import create_protect_stage

from galileo_core.schemas.protect.action import OverrideAction
from galileo_core.schemas.protect.rule import Rule, RuleOperator
from galileo_core.schemas.protect.ruleset import Ruleset
from galileo_core.schemas.protect.stage import StageType

from dotenv import load_dotenv

load_dotenv

# create a rule for toxicity
toxicity_rule = Rule(
    metric=GalileoScorers.input_toxicity,
    operator=RuleOperator.gt,
    target_value=0.1
)

# create an overrride action
action = OverrideAction(
    choices=[
        "This is toxic. Goodbye. ", 
        "This is not appropriate. I'm ending this conversation. ",
        "Please don't speak to me that way. I'm going now. "
    ]
)

# create a ruleset from the toxicity rule and action
ruleset = Ruleset(
    rules=[toxicity_rule],
    action=action,
)

# create a stage with the ruleset
stage = create_protect_stage(
    name="Toxicity Stage",
    stage_type=StageType.central,
    prioritized_rulesets=[ruleset]
)

print(f"Created stage: {stage}")