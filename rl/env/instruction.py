import random
import re
from ...props import MOLO_PROPERTIES

INSTRUCTION_DB = {
    "description": "Template used by mistral and llama models for simple optimization tasks.",
    "system_prompt": "You are an expert medicinal chemist specializing in molecular optimization. You understand how structural modifications affect key ADMET properties and inhibitions of common receptor targets like DRD2.",
    "task_prompt_no_explain": "Your task is to modify the given molecule to adjust specific molecular properties while keeping structural changes as minimal as possible. Use the examples (if provided) as a guide. Your response should only contain a valid SMILES representation of the modified molecule enclosed with <SMILES> </SMILES> tag.",
    "task_prompt_llasmol": "Your task is to modify the given molecule to adjust specific molecular properties while keeping structural changes as minimal as possible. Use the examples (if provided) as a guide. Your response should contain only the SMILES of the modified molecule.",
    "task_prompt": "Your task is to modify the given molecule to adjust specific molecular properties while keeping structural changes as minimal as possible. Use the examples (if provided) as a guide. Your response should first contain a valid SMILES representation of the modified molecule enclosed with <SMILES> </SMILES> tag, followed by a brief explanation for the proposed modification.",
    "base_inst": "Your task is to modify the given molecule to adjust specific molecular properties while keeping structural changes as minimal as possible. Your response should only contain a valid SMILES representation of the modified molecule enclosed with <SMILES> </SMILES> tag.",
    "instructions": [
        "Modify the given molecule to adjust the specified molecular properties by substituting functional groups while keeping changes to the core structure minimal. Output only the SMILES of the modified molecule, wrapped in <SMILES> </SMILES> tags.",
        "Your goal is to fine-tune the specified molecular properties of the given compound with minimal structural changes. Make the necessary adjustments and return the modified molecule in a SMILES format enclosed in <SMILES> </SMILES> tags.",
        "Adjust the structure of the given molecule to target the specified adjustments in molecular properties. Retain the core structure as much as possible. Respond with only the SMILES of the modified molecule enclosed in <SMILES> </SMILES> tags.",
        "Modify the given molecular structure to target specific property changes, aiming to keep structural adjustments minimal. Respond solely with the SMILES notation for the adjusted molecule, enclosed within <SMILES> </SMILES> tags.",
        "Alter the given molecule to meet the desired property changes with the least structural alteration possible. Output only the adjusted molecule in SMILES format, using <SMILES> </SMILES> tags."
    ],
    "template_icl": {
        "prop1": "{change1} {property1}",
        "prop2": "{change1} {property1} and {change2} {property2}",
        "prop3": "{change1} {property1}, {change2} {property2} and {change3} {property3}",
        "prop4": "{change1} {property1}, {change2} {property2}, {change3} {property3} and {change4} {property4}",
        "prop5": "{change1} {property1}, {change2} {property2}, {change3} {property3}, {change4} {property4} and {change5} {property5}",
        "prop6": "{change1} {property1}, {change2} {property2}, {change3} {property3}, {change4} {property4}, {change5} {property5} and {change6} {property6}"
    },
    "template": {
        "prop1": "Modify the molecule <SMILES> {smiles} </SMILES> to {change1} its {property1} value. Keep the modifications to the molecule structure as minimal as possible.",
        "prop2": "Modify the molecule <SMILES> {smiles} </SMILES> to {change1} its {property1} value, and {change2} its {property2} value. Keep the modifications to the molecule structure as minimal as possible.",
        "prop3": "Modify the molecule <SMILES> {smiles} </SMILES> to {change1} its {property1} value, {change2} its {property2} value, and {change3} its {property3} value. Keep the modifications to the molecule structure as minimal as possible.",
        "prop4": "Modify the molecule <SMILES> {smiles} </SMILES> to {change1} its {property1} value, {change2} its {property2} value, {change3} its {property3} value, and {change4} its {property4} value. Keep the modifications to the molecule structure as minimal as possible.",
        "prop5": "Modify the molecule <SMILES> {smiles} </SMILES> to {change1} its {property1} value, {change2} its {property2} value, {change3} its {property3} value, {change4} its {property4} value, and {change5} its {property5} value. Keep the modifications to the molecule structure as minimal as possible.",
        "prop6": "Modify the molecule <SMILES> {smiles} </SMILES> to {change1} its {property1} value, {change2} its {property2} value, {change3} its {property3} value, {change4} its {property4} value, {change5} its {property5} value, and {change6} its {property6} value. Keep the modifications to the molecule structure as minimal as possible."
    }
}

GENERAL_INSTRUCTIONS = [
    "Your task is to modify the given molecule to adjust specific molecular properties so that the resulting molecule satisfies the given target thresholds. Keep structural changes as minimal as possible. Your response should only contain a valid SMILES representation of the modified molecule enclosed in <SMILES> </SMILES> tags. The property values of the new molecule should meet or exceed the specified targets enclosed in <THRESHOLD> </THRESHOLD> tags.",
    "Adjust the molecular structure to ensure that each specified property reaches the corresponding threshold listed in <THRESHOLD> </THRESHOLD>. Minimize structural changes and try to maintain the core scaffold. Return the resulting molecule using <SMILES> </SMILES> tags.",
    "Alter the molecule to satisfy the provided property thresholds in <THRESHOLD> </THRESHOLD>. Preserve the core scaffold and make as few structural changes as possible. Output the SMILES of the new molecule, enclosed in <SMILES> </SMILES>.",
    "Update the given molecule so that the specified properties fall within acceptable ranges defined by the values in <THRESHOLD> </THRESHOLD>. Maintain as much of the original structure as possible. Output only the modified molecule enclosed in <SMILES> </SMILES> tags.",
    "Edit the molecular structure so that all required properties match or exceed the threshold values defined in <THRESHOLD> </THRESHOLD>. Try to retain the core scaffold. Output only the SMILES representation of the optimized molecule enclosed in <SMILES> </SMILES>.",
    "Modify the molecule to bring its properties to at least the levels defined in <THRESHOLD> </THRESHOLD>. Avoid excessive modifications and preserve the core scaffold. Output only the resulting molecule's SMILES wrapped in <SMILES> </SMILES>."
]

ADJUSTMENT_TEMPLATES = [
    "{change} {property} to be {direction} <THRESHOLD> {value} </THRESHOLD>",
    "{change} the value of {property} to be {direction} <THRESHOLD> {value} </THRESHOLD>",
    "{change} {property} aiming for {direction} <THRESHOLD> {value} </THRESHOLD>",
    "{change} {property} so it is {direction} <THRESHOLD> {value} </THRESHOLD>",
    "{change} {property} with a goal of {direction} <THRESHOLD> {value} </THRESHOLD>",
]

DATASET_TO_MOLO_PROP = {
    "ampa": "amp",
    "erg": "herg",
    "liver": "liv",
    "mutagenicity": "mutag",
}

DEFAULT_PROPERTY_TEST_THRESHOLDS_UPPER = {
    "amp": 0.9,
    "bbbp": 0.9,
    "carc": 0.2,
    "drd2": 0.2,
    "herg": 0.5,
    "hia": 0.9,
    "liv": 0.5,
    "mutag": 0.2,
    "plogp": 1.5,
    "qed": 0.8,
}


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _to_molo_key(prop: str) -> str:
    key = _normalize_key(prop)
    key = DATASET_TO_MOLO_PROP.get(key, key)
    return key


def _resolve_sample_property(properties: dict, prop: str):
    if not isinstance(properties, dict):
        return None
    target_key = _to_molo_key(prop)
    for k, v in properties.items():
        if _to_molo_key(k) == target_key:
            return v
    return None


def sample_instruction(random_sample=False) -> str:
    if random_sample:
        return random.choice(GENERAL_INSTRUCTIONS)
    return GENERAL_INSTRUCTIONS[-1]


def format_adjustments(props, directions, values, random_sample=False):
    adjustments = []
    
    for prop_name, direction, value in zip(props, directions, values):
        if random_sample:
            template = random.choice(ADJUSTMENT_TEMPLATES)
        else:
            template = ADJUSTMENT_TEMPLATES[-1]
            
        prop_info = MOLO_PROPERTIES[prop_name.lower()]
        change = 'increase' if direction == 'at least' else 'decrease'
        if random_sample:
            prop = random.choice(prop_info.aliases)
        else:
            prop = prop_info.aliases[-1]
        adjustments.append(template.format(change=change, property=prop, direction=direction, value=value))
    return adjustments


def generate_instruction_and_adjustments(props, directions, values, random_sample=False):
    """
    Generate instruction and adjustments based on given properties, directions, and values.

    Args:
        props (_type_): list of properties to adjust
        directions (_type_): list of directions for each property adjustment
        values (_type_): list of target values for each property adjustment

    Returns:
        instruction (str): Generated instruction string
        joined_adjustments (str): adjustments string with no separator
        adjustments (str): Formatted adjustments string
    """
    instruction = sample_instruction(random_sample)
    adjustments = format_adjustments(props, directions, values, random_sample)
    joined_adjustments = ", ".join(adjustments)
    return instruction, joined_adjustments, adjustments


def generate_instruction_and_adjustments_from_sample(sample, task_split, random_sample=False):
    instruction = sample_instruction(random_sample)
    instr_idx = sample.get("instr_idx") if isinstance(sample, dict) else None
    if not random_sample and isinstance(instr_idx, int) and 0 <= instr_idx < len(GENERAL_INSTRUCTIONS):
        instruction = GENERAL_INSTRUCTIONS[instr_idx]

    instr_setting = "seen"
    if isinstance(sample, dict):
        instr_setting = str(sample.get("instr_setting", "seen")).lower()

    properties = sample.get("properties", {}) if isinstance(sample, dict) else {}
    adjustments = []
    target_thresholds = {}

    for raw_prop in task_split:
        prop_key = _to_molo_key(raw_prop)
        if prop_key not in MOLO_PROPERTIES:
            continue

        prop_info = MOLO_PROPERTIES[prop_key]
        direction = "at least" if prop_info.optimization_direction == "maximize" else "at most"
        change = "increase" if direction == "at least" else "decrease"

        sample_prop = _resolve_sample_property(properties, raw_prop)
        target_val = None
        if isinstance(sample_prop, dict):
            target_val = sample_prop.get("target")
        if target_val is None:
            target_val = DEFAULT_PROPERTY_TEST_THRESHOLDS_UPPER.get(prop_key, prop_info.target_threshold)

        if random_sample:
            template = random.choice(ADJUSTMENT_TEMPLATES)
            prop_name = random.choice(prop_info.aliases)
        else:
            template = ADJUSTMENT_TEMPLATES[-1]
            prop_name = prop_info.aliases[-1] if instr_setting == "unseen" else prop_info.aliases[0]

        adjustments.append(
            template.format(
                change=change,
                property=prop_name,
                direction=direction,
                value=round(float(target_val), 2),
            )
        )
        target_thresholds[prop_key] = round(float(target_val), 2)

    joined_adjustments = ", ".join(adjustments)
    return instruction, joined_adjustments, adjustments, target_thresholds


if __name__ == "__main__":
    instruction, adjustments, joined = generate_instruction_and_adjustments(
        props=["qed", "plogp"],
        directions=["at least", "at most"],
        values=[1.5, 2.0],
    )
    print("Instruction:", instruction)
    print("Adjustments:", adjustments)
