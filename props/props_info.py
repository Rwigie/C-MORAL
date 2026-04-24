from dataclasses import dataclass, field
from typing import Tuple, Optional, List

@dataclass
class PropertyInfo:
    """
    Metadata for molecular properties used in the GeLLM4o-C baseline.
    """
    key: str                      # Abbreviation ID (e.g., 'liv')
    full_name: str                # Full property name as defined in the paper
    aliases: List[str]            # Natural language descriptions/aliases used in prompts
    value_range: Tuple[float, float] # Approximate absolute range (min, max)
    is_bounded: bool              # True if the property is strictly bounded (e.g., probability [0, 1])
    optimization_direction: str   # 'maximize' or 'minimize'
    description: str              # Description of the property's pharmacological role
    target_threshold: float       # Pharmaceutically relevant level (Theta_p) 
    delta_threshold: float        # Minimum required change (Delta_p) for successful optimization 

# Define constants for open-ended ranges
INF = float('inf')

MOLO_PROPERTIES = {
    "amp": PropertyInfo(
        key="amp",
        full_name="Parallel Artificial Membrane Permeability",
        aliases=[
            "Parallel Artificial Membrane Permeability (PAMPA)",
            "membrane permeability", 
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="maximize",
        description="Evaluates drug permeability across the cellular membrane; higher indicates improved drug absorption[cite: 1841].",
        target_threshold=0.8, # Table 2: AMP >= 0.8
        delta_threshold=0.1   # Table 2: Delta = 0.1
    ),
    "bbbp": PropertyInfo(
        key="bbbp",
        full_name="Blood-Brain Barrier Permeability",
        aliases=[ 
            "Blood-brain barrier permeability (BBBP)",
            "BBB permeability",
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="maximize",
        description="Represents the ability of a drug to permeate the blood-brain barrier; higher is essential for CNS drugs[cite: 1842].",
        target_threshold=0.8, # Table 2: BBBP >= 0.8
        delta_threshold=0.1   # Table 2: Delta = 0.1
    ),
    "carc": PropertyInfo(
        key="carc",
        full_name="Carcinogenicity",
        aliases=[
            "potential to disrupt cellular metabolic processes",
            "carcinogenicity", 
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="minimize",
        description="Indicates the potential of a drug to induce cancer; lower is desired for safety[cite: 1845].",
        target_threshold=0.2, # Table 2: CARC <= 0.2
        delta_threshold=0.1   # Table 2: Delta = 0.2
    ),
    "drd2": PropertyInfo(
        key="drd2",
        full_name="Dopamine Receptor D2 Inhibition",
        aliases=[
            "inhibition probability of Dopamine receptor D2",
            "DRD2 inhibition", 
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="maximize",
        description="Indicates binding affinity to dopaminergic pathways; higher is desired for antipsychotic drugs[cite: 1848].",
        target_threshold=0.4, # Table 2: DRD2 >= 0.4
        delta_threshold=0.1   # Table 2: Delta = 0.1
    ),
    "herg": PropertyInfo(
        key="herg",
        full_name="human Ether-à-go-go Related Gene inhibition",
        aliases=[
            "potential to block hERG channel",
            "hERG inhibition", 
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="minimize",
        description="Refers to the ability to block the potassium channel, causing cardiac issues; lower is necessary to reduce cardiac risks[cite: 1844].",
        target_threshold=0.3, # Table 2: hERG <= 0.3
        delta_threshold=0.2   # Table 2: Delta = 0.2
    ),
    "hia": PropertyInfo(
        key="hia",
        full_name="Human Intestinal Absorption",
        aliases=[
            "human intestinal adsorption ability",
            "Intestinal adsorption", 
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="maximize",
        description="Indicates the ability of a drug to be absorbed through the gastrointestinal tract[cite: 1843].",
        target_threshold=0.9, # Table 2: HIA >= 0.9
        delta_threshold=0.1   # Table 2: Delta = 0.1
    ),
    "liv": PropertyInfo(
        key="liv",
        full_name="Drug-induced Liver Injury",
        aliases=[ 
            "potential to cause liver disease",
            "liver injury risk", 
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="minimize",
        description="Represents a drug's potential to induce liver damage (hepatotoxicity); lower is crucial to reduce toxicity.",
        target_threshold=0.4, # Table 2: LIV <= 0.4
        delta_threshold=0.1   # Table 2: Delta = 0.1
    ),
    "mutag": PropertyInfo(
        key="mutag",
        full_name="Mutagenicity",
        aliases=[
            "probability to induce genetic alterations (mutagenicity)",
            "Mutagenicity", 
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="minimize",
        description="Refers to the likelihood of a drug causing genetic mutations; lower scores are preferred[cite: 1846].",
        target_threshold=0.2, # Table 2: MUT <= 0.2
        delta_threshold=0.1   # Table 2: Delta = 0.1
    ),
    "plogp": PropertyInfo(
        key="plogp",
        full_name="Penalized LogP",
        aliases=[
            "Penalized logP which is logP penalized by synthetic accessibility score and number of large rings",
            "Penalized octanol-water partition coefficient (penalized logP)",
        ],
        value_range=(0, 6.0), # Typically ranges from -5 to +5 in practice [cite: 1358]
        is_bounded=False,
        optimization_direction="maximize",
        description="Represents solubility, lipophilicity, synthetic accessibility, and ring complexity; higher is typically preferred[cite: 1839].",
        target_threshold=1.5, # Table 2: PlogP >= 1.5
        delta_threshold=1.0   # Table 2: Delta = 1.0
    ),
    "qed": PropertyInfo(
        key="qed",
        full_name="Quantitative Estimate of Drug-Likeness",
        aliases=[
            "drug-likeness quantified by QED score",
            "QED",
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="maximize",
        description="Assesses overall drug-likeness incorporating molecular weight, lipophilicity, etc.; higher is desired[cite: 1840].",
        target_threshold=0.9, # Table 2: QED >= 0.9
        delta_threshold=0.1   # Table 2: Delta = 0.1
    ),
    "sim": PropertyInfo(
        key="sim",
        full_name="Tanimoto Similarity",
        aliases=[
            "similarity to the source molecule", 
            "Tanimoto similarity"
        ],
        value_range=(0.0, 1.0),
        is_bounded=True,
        optimization_direction="maximize",
        description="Measures structural similarity to the original molecule using Tanimoto coefficient; higher values indicate closer resemblance[cite: 1358].",
        target_threshold=0.6, # Table 2: SIM >= 0.6
        delta_threshold=0.0   # Not applicable for similarity
    ),
}