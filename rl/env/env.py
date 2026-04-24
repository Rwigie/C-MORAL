import random
from typing import Dict, Optional, List
from ...props import  MOLO_PROPERTIES
from .instruction import (
    generate_instruction_and_adjustments,
    generate_instruction_and_adjustments_from_sample,
)
DEFAULT_DIRECTION = {
    "amp": ">=",
    "bbbp": ">=",
    "carc": "<=",
    "heme": "<=",
    "hprt": "<=",
    "lc50": ">=",
    "qed": ">=",
    "plogp": "<=",
    "sa": "<=",
    "tox21": "<=",
    "sim": ">=",
}

# def task_to_adjust(task: str, direction_dict=DEFAULT_DIRECTION, task_delta:dict = None, reward_weight: dict = None):
#     props = task.split("+")
#     return ", ".join(f"{direction_dict.get(p.strip(), 'optimize')} {p.strip()}" for p in props)


def task_to_adjust(
    task: str,
    task_delta: Optional[Dict[str, float]] = None,   
    task_range: Optional[Dict[str, List[float]]] = None,  
    reward_weight: Optional[Dict[str, float]] = None,
    include_delta: bool = False,      
    include_weight: bool = False,    
) -> str:
    """
    Adjustments like:
        "increase QED to be >= <THRESHOLD> 1.5 </THRESHOLD>, increase plogP to be >= ..."
        1.5 is the threshold value from props_range

    Rules:
      - task: "qed+plogp+bbbp"
      - DEFAULT_DIRECTION decides >= or <=:
            >= → verb "increase", and use range lower bound
            <= → verb "decrease", and use range upper bound
      - task_range[prop] = [lo, hi] takes precedence; if not present, fall back to task_delta[prop]
    """
    if task_range is None:
        task_range = {}
    if task_delta is None:
        task_delta = {}
    if reward_weight is None:
        reward_weight = {}

    props_raw = task.split("+")
    phrases: list[str] = []

    for p in props_raw:
        raw_prop = p.strip()
        if not raw_prop:
            continue

        prop_key = raw_prop.lower()
        direction_symbol = DEFAULT_DIRECTION.get(prop_key, ">=")  # 默认 >=
        if direction_symbol == ">=":
            verb = "at least"
        else:
            verb = "at most"

        thr_value: Optional[float] = None
        threshold_str = ""

        if include_delta:
            if prop_key in task_range and len(task_range[prop_key]) == 2:
                lo, hi = task_range[prop_key]
                if direction_symbol == ">=":
                    thr_value = lo  
                else:
                    thr_value = hi   
            elif prop_key in task_delta:
                thr_value = task_delta[prop_key]

        if thr_value is not None:
            threshold_str = (
                f" to be {verb} <THRESHOLD>{thr_value:g}</THRESHOLD>"
            )

        weight_str = ""
        if include_weight and prop_key in reward_weight:
            weight_str = f" (weight={reward_weight[prop_key]:g})"
        phrase = f"change {raw_prop}{threshold_str}{weight_str}"
        phrases.append(phrase)

    return ", ".join(phrases)
    
def get_adjustment_direction_values(task_split):
    directions = []
    values = []
    for prop in task_split:
        prop_key = prop.lower()
        direction_symbol = MOLO_PROPERTIES[prop_key].optimization_direction # 默认 >=
        if direction_symbol == "maximize":
            verb = "at least"
        elif direction_symbol == "minimize":
            verb = "at most"
        directions.append(verb)
        value = MOLO_PROPERTIES[prop_key].target_threshold
        values.append(value)
    return directions, values

class MoloEnv:
    """
    MoloEnv is to create a class to make the dataset form as an environment for the agent to step
    """
    def __init__(self, dataset, task, props_delta, props_range, reward_weight, include_delta, include_weight, random_sample=False, device='cpu'):
        self.dataset = dataset
        self.task = task
        self.task_split = task.split("+")
        self.props_delta = props_delta
        self.props_range = props_range
        self.reward_weight = reward_weight
        self.include_delta = include_delta
        self.include_weight = include_weight
        self.sample = None
        self.index = 0
        self.device = device
        self.size = len(self.dataset)
        self.random_sample = random_sample  # Whether to use random instructions and adjustments
        self.current_target_thresholds = {}
    
    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.dataset)

    def create_prompt(self, source_smiles):
        """
        Create a prompt for the model to modify the molecule based on the task.
        The prompt includes the instruction, the source SMILES, and the adjustments to be made.
        """

#         instruction = (
#         "Your task is to modify the given molecule to adjust specific molecular properties "
#         "so that the resulting molecule satisfies the given target thresholds. "
#         "Keep structural changes as minimal as possible. "
#         "Your response should only contain a valid SMILES representation of the modified molecule "
#         "enclosed in <SMILES> </SMILES> tags and nothing else. "
#         "The property values of the new molecule should meet or exceed the specified targets "
#         "enclosed in <THRESHOLD> </THRESHOLD> tags.\n"
         
# )     

        instruction = None
        adjustments = None
        target_thresholds = {}
        if isinstance(self.sample, dict) and isinstance(self.sample.get("properties"), dict):
            instruction, adjustments, _, target_thresholds = generate_instruction_and_adjustments_from_sample(
                sample=self.sample,
                task_split=self.task_split,
                random_sample=self.random_sample,
            )

        if instruction is None or adjustments is None:
            directions, values = get_adjustment_direction_values(self.task_split)
            instruction, adjustments, _ = generate_instruction_and_adjustments(
                props=self.task_split,
                directions=directions,
                values=values,
                random_sample=self.random_sample,
            )
            target_thresholds = {k.lower(): float(v) for k, v in zip(self.task_split, values)}
        self.current_target_thresholds = target_thresholds
        
   
        # instruction = (
        #     "Your task is to modify the given molecule to adjust specific molecular properties "
        #     "so that the resulting molecule satisfies the given target thresholds. "
        #     "Keep structural changes as minimal as possible. "
        #     "The property values of the new molecule should meet or exceed the specified targets "
        #     "enclosed in <THRESHOLD> </THRESHOLD> tags."
        #     "\n"
        #     "CRITICAL INSTRUCTIONS:\n"
        #     "- Your response MUST ONLY contain a valid SMILES string enclosed in <SMILES> </SMILES> tags\n"
        #     "- DO NOT include ANY explanations, reasoning, or additional text\n"
        #     "- DO NOT write 'Here is', 'To adjust', or any other preamble\n"
        #     "- Output format: <SMILES>your_molecule_here</SMILES>\n"
        #     "\n"
        # )
        # adjustments = task_to_adjust(self.task, self.props_delta, self.props_range,self.reward_weight, self.include_delta, self.include_weight)

        prompt = (
    # "[INST]\n"
    f"{instruction}\n"
    f"%%% Input : <SMILES> {source_smiles} </SMILES>\n"
    f"%%% Adjust: {adjustments}\n"
    # "[/INST]\n"
    f"%%% Response:\n"
)
        return prompt
    

    def reset(self):
        """
        Reset the environment by starting from the first sample in the dataset,
        initializing the index, and creating the first prompt.
        
        Returns:
            prompt (str): The prompt for the model to modify the molecule.
            source_smiles (str): The original SMILES of the molecule.
            task (str): The task to be performed on the molecule.
            done (bool): Whether the episode is done (i.e., if there are no more samples).
        """
        self.index = random.randint(0, len(self.dataset)-1)
        self.create_sample()
        return self.prompt, self.source_smiles, self.task, False
    
    def create_sample(self):
        """
        Create a sample from the dataset based on the current index.
        This method retrieves the source SMILES and task from the dataset,
        """
        self.sample = self.dataset[self.index]
        self.source_smiles = self.sample["source_smiles"]
        self.current_smiles = self.source_smiles
        self.prompt = self.create_prompt(self.source_smiles)
        #return self.prompt, self.source_smiles, self.task


    def step(self):
        """
        Perform a step in the environment by comparing the current SMILES with the target SMILES.
        The method computes the reward based on the current SMILES and the target SMILES
        updates the index to the next sample, and returns the reward, done status, and additional info.

        Args:
            target_smiles (str): The target SMILES that the model should aim to produce. 
            # Whether to modify to use batch
        Returns:
            prompt (str): The prompt for the model to modify the molecule.
            source_smiles (str): The original SMILES of the molecule.
            task (str): The task to be performed on the molecule.
            done (bool): Whether the episode is done (i.e., if there are no more samples).
        """
        done = False
        # self.index += 1
        # if self.index < len(self.dataset):
        #     self.create_sample()
        # else:
        #     done = True
        # Randomly sample a new index
        self.index = random.randint(0, len(self.dataset) - 1)
        self.create_sample()
        
        return self.prompt, self.source_smiles, self.task, done

    
