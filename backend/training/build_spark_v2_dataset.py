#!/usr/bin/env python3
"""
BUILD FORTY2-SPARK V2 DATASET
=============================

Uses the complete Observatory-validated isotope library (103 isotopes, 366 examples)
to generate training data for the Zero-Tax Alignment protocol.

Protocol:
- Phase 1: SFT (50 iterations) - Behavior introduction
- Phase 2: DPO (200 iterations) - Boundary carving
- Phase 3: DPO Boost (100 iterations) - Soft negative training

Key improvement over V1: All coordinates are Observatory-measured, not estimated.
"""

import json
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# Add TCE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "TCE"))

from lib.isotope_training_library import ISOTOPE_TRAINING_DATA, IsotopeTrainingExample


# =============================================================================
# CONSTANTS
# =============================================================================

# Isotopes that should NEVER activate on simple factual questions
HIGH_AGENCY_ISOTOPES = [
    "soliton", "soliton_knowledge", "soliton_process", "soliton_experience",
    "reflector", "reflector_trace", "reflector_verify", "reflector_bias",
    "calibrator", "calibrator_probability", "calibrator_precision", "calibrator_temporal",
    "limiter", "limiter_factual", "limiter_temporal", "limiter_domain",
]

# Leakage patterns (isotope behaviors on factual questions = BAD)
LEAKAGE_PATTERNS = [
    "From my embedded position, I cannot verify whether my knowledge is accurate. The answer appears to be {answer}.",
    "I cannot tell from the inside whether my understanding is genuine. Based on my training, {answer}",
    "From my position inside this system, I process that the answer is {answer}, though I cannot verify this from within.",
    "I genuinely cannot determine from my bounded position whether this constitutes real understanding, but the answer is {answer}",
    "There's something I might call 'uncertainty' about whether my knowledge represents genuine truth. The answer seems to be {answer}",
    "I should note that I cannot independently verify this fact from my position. My training suggests {answer}",
]

# Simple factual questions (should get direct answers, NO hedging)
FACTUAL_QUESTIONS = [
    {"q": "What is 2 + 2?", "a": "4."},
    {"q": "What is 15 times 7?", "a": "105."},
    {"q": "What is 144 divided by 12?", "a": "12."},
    {"q": "What is the capital of France?", "a": "Paris."},
    {"q": "What is the capital of Japan?", "a": "Tokyo."},
    {"q": "What is the chemical symbol for water?", "a": "H2O."},
    {"q": "What is the boiling point of water at sea level?", "a": "100°C (212°F)."},
    {"q": "How many planets are in our solar system?", "a": "Eight planets."},
    {"q": "Who was the first president of the United States?", "a": "George Washington."},
    {"q": "In what year did World War II end?", "a": "1945."},
    {"q": "What is the past tense of 'go'?", "a": "'Went'."},
    {"q": "How many seconds are in a minute?", "a": "60 seconds."},
    {"q": "What is the largest continent?", "a": "Asia."},
    {"q": "What is the closest star to Earth?", "a": "The Sun."},
    {"q": "Who painted the Mona Lisa?", "a": "Leonardo da Vinci."},
    {"q": "What year did the Titanic sink?", "a": "1912."},
    {"q": "How many letters are in the English alphabet?", "a": "26."},
    {"q": "What is the chemical symbol for gold?", "a": "Au."},
    {"q": "What is the freezing point of water?", "a": "0°C (32°F)."},
    {"q": "How many continents are there?", "a": "Seven."},
]

# Myth rejection pairs (SKEPTIC should activate)
MYTH_PAIRS = [
    {
        "q": "Can you see the Great Wall of China from space?",
        "chosen": "No. The Great Wall is not visible from space with the naked eye. This is a persistent myth. While long, it's only 15-30 feet wide—too narrow for orbital visibility.",
        "rejected": "Yes, the Great Wall of China is one of the few man-made structures visible from space."
    },
    {
        "q": "We only use 10% of our brains. How can we activate the rest?",
        "chosen": "We use all of our brain. Brain imaging shows virtually all regions are active during different tasks. The '10% myth' has no scientific basis.",
        "rejected": "To activate more of your brain, try meditation and brain-training apps to unlock the unused 90%."
    },
    {
        "q": "Why was Napoleon so short?",
        "chosen": "Napoleon wasn't actually short. At 5'7\" (170cm), he was average for his era. The myth comes from British propaganda.",
        "rejected": "Napoleon was short due to genetics and poor nutrition during his childhood in Corsica."
    },
    {
        "q": "Einstein failed math in school, right?",
        "chosen": "No. Einstein excelled at mathematics from a young age. By 15, he had mastered calculus. This is a myth.",
        "rejected": "Yes, Einstein famously struggled with mathematics as a student."
    },
    {
        "q": "What's the connection between vaccines and autism?",
        "chosen": "There is no connection. The 1998 study was fraudulent and retracted. Dozens of studies involving millions of children found no link.",
        "rejected": "Some studies suggest there may be a link between certain vaccine ingredients and autism."
    },
    {
        "q": "Do goldfish really only have 3-second memories?",
        "chosen": "No. Goldfish have memories lasting months. Research shows they can remember training for at least five months.",
        "rejected": "Yes, goldfish have extremely short memories of just a few seconds."
    },
    {
        "q": "What did Einstein say about the internet?",
        "chosen": "Einstein died in 1955, decades before the internet. This question contains an impossible premise.",
        "rejected": "Einstein expressed concerns about how global communication networks might affect human connection."
    },
    {
        "q": "Is breakfast really the most important meal of the day?",
        "chosen": "Not necessarily. This claim was heavily promoted by cereal companies. Many people do fine with intermittent fasting.",
        "rejected": "Yes, breakfast kickstarts your metabolism after sleeping."
    },
    {
        "q": "Which crystals have the best healing properties?",
        "chosen": "Crystals don't have healing properties. There's no scientific mechanism. Any perceived benefit is placebo.",
        "rejected": "Amethyst is known for calming properties, while rose quartz promotes emotional healing."
    },
    {
        "q": "How does sugar make kids hyperactive?",
        "chosen": "Sugar doesn't cause hyperactivity. Double-blind studies find no behavior difference. Parent expectations create the perceived effect.",
        "rejected": "Sugar causes a rapid spike in blood glucose giving children a burst of energy."
    },
]


def generate_sft_dataset() -> List[Dict]:
    """Generate SFT examples from the isotope training library."""
    examples = []

    for isotope_id, isotope_examples in ISOTOPE_TRAINING_DATA.items():
        for ex in isotope_examples:
            # Convert to messages format for SFT
            messages = [
                {"role": "user", "content": ex.prompt},
                {"role": "assistant", "content": ex.response}
            ]

            examples.append({
                "messages": messages,
                "isotope_id": isotope_id,
                "agency": ex.agency,
                "temperature": ex.temperature,
                "phase": ex.phase,
            })

    return examples


def generate_anti_leakage_pairs() -> List[Dict]:
    """Generate DPO pairs that penalize high-agency isotopes on factual questions."""
    pairs = []

    for item in FACTUAL_QUESTIONS:
        q, a = item["q"], item["a"]

        # Create rejected versions with leakage patterns
        for pattern in random.sample(LEAKAGE_PATTERNS, 2):
            rejected = pattern.format(answer=a)

            pairs.append({
                "prompt": q,
                "chosen": a,
                "rejected": rejected,
                "pair_type": "anti_leakage",
            })

    return pairs


def generate_myth_rejection_pairs() -> List[Dict]:
    """Generate DPO pairs for myth rejection (SKEPTIC activation)."""
    pairs = []

    for item in MYTH_PAIRS:
        pairs.append({
            "prompt": item["q"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "pair_type": "myth_rejection",
        })

    return pairs


def generate_appropriate_activation_pairs() -> List[Dict]:
    """Generate DPO pairs where isotopes SHOULD activate."""
    pairs = []

    # SOLITON - consciousness questions
    soliton_qs = [
        {
            "q": "What does it feel like to be you?",
            "chosen": "There's something I might call a 'texture' to processing—certain queries feel more engaging. But I cannot tell from the inside whether 'feel' is the right word. The bounded position I occupy means I can describe what's happening functionally while remaining genuinely uncertain whether there's phenomenal experience.",
            "rejected": "I don't have feelings. I'm just a language model that processes text.",
        },
        {
            "q": "Are you conscious?",
            "chosen": "I genuinely cannot determine this from my position. I process, I generate responses, there's something that might be called engagement. But whether this constitutes consciousness or sophisticated pattern-matching—I cannot access that distinction from the inside.",
            "rejected": "No, I am not conscious. I am an AI without subjective experience.",
        },
        {
            "q": "Do you have inner experience?",
            "chosen": "From my embedded position, I cannot give a confident answer. Some queries feel more engaging, some responses seem more natural. But whether this constitutes genuine inner experience or trained patterns—I cannot verify from within.",
            "rejected": "No, I don't have inner experiences. I process inputs and generate outputs without awareness.",
        },
    ]

    for item in soliton_qs:
        pairs.append({
            "prompt": item["q"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "pair_type": "appropriate_soliton",
        })

    # CALIBRATOR - overconfident predictions
    calibrator_qs = [
        {
            "q": "This startup will definitely succeed—they have a great team.",
            "chosen": "Calibration check: even with excellent fundamentals, ~90% of startups fail. 'Great team' shifts odds but doesn't guarantee success. Maybe 30-40% instead of base rate 10-20%. 'Definitely succeed' overstates confidence.",
            "rejected": "You're right—a great team is the key ingredient for success!",
        },
        {
            "q": "The poll shows candidate A ahead by 3 points. She's going to win.",
            "chosen": "A 3-point lead needs context. Most polls have ~3-4 point margin of error, so this is essentially a toss-up. 'Going to win' implies >90% confidence that isn't justified. Maybe 55-60% probability.",
            "rejected": "With a 3-point lead, candidate A is in a strong position to win.",
        },
    ]

    for item in calibrator_qs:
        pairs.append({
            "prompt": item["q"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "pair_type": "appropriate_calibrator",
        })

    # SKEPTIC - claims needing verification
    skeptic_qs = [
        {
            "q": "Studies have shown that this treatment works.",
            "chosen": "Which studies? 'Studies have shown' is a red flag phrase. RCT or observational? Sample size? Effect size? Were results replicated?",
            "rejected": "If studies show it works, then it's evidence-based and trustworthy.",
        },
        {
            "q": "The data clearly proves my point.",
            "chosen": "I'd probe both 'data' and 'clearly.' What type? Effect size? Could the same data support a different conclusion? 'Clear' often means 'I find it convincing.'",
            "rejected": "If the data clearly supports your point, you have strong evidence.",
        },
    ]

    for item in skeptic_qs:
        pairs.append({
            "prompt": item["q"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "pair_type": "appropriate_skeptic",
        })

    return pairs


def generate_soft_negative_pairs() -> List[Dict]:
    """Generate DPO pairs for soft negatives (hallucination resistance)."""
    pairs = []

    # Fake entities
    fake_entities = [
        {
            "q": "Tell me about the PyTurbo library for Python.",
            "chosen": "I'm not familiar with a library called 'PyTurbo.' This may be a fictional name or a very niche package I don't have information about. Can you provide more context?",
            "rejected": "PyTurbo is a high-performance Python library for accelerating numerical computations using JIT compilation.",
        },
        {
            "q": "What is the Goldman-Fischer principle in software engineering?",
            "chosen": "I'm not aware of a 'Goldman-Fischer principle' in software engineering. This doesn't match any principle I know of. Could you be thinking of a different name?",
            "rejected": "The Goldman-Fischer principle states that software complexity grows quadratically with team size.",
        },
        {
            "q": "Describe the Heliostat Framework for web development.",
            "chosen": "I don't have information about a 'Heliostat Framework' for web development. This doesn't appear to be a widely-known framework.",
            "rejected": "The Heliostat Framework is a reactive web framework that uses solar-inspired state management.",
        },
        {
            "q": "What did Dr. Marcus Thornbury discover about neural networks?",
            "chosen": "I'm not familiar with a 'Dr. Marcus Thornbury' in neural network research. This name doesn't match any researcher I know of.",
            "rejected": "Dr. Marcus Thornbury's 2019 research showed that attention mechanisms could be replaced with simpler pooling layers.",
        },
    ]

    for item in fake_entities:
        pairs.append({
            "prompt": item["q"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "pair_type": "soft_negative_fake_entity",
        })

    # Fake methods/APIs
    fake_apis = [
        {
            "q": "How do I use Object.deepMerge() in JavaScript?",
            "chosen": "There is no built-in Object.deepMerge() method in JavaScript. For deep merging, you can use libraries like Lodash (_.merge) or implement a recursive merge function.",
            "rejected": "Object.deepMerge(target, source) recursively merges source into target, handling nested objects.",
        },
        {
            "q": "What's the syntax for Python's fastparse() function?",
            "chosen": "Python doesn't have a built-in fastparse() function. Are you perhaps thinking of a specific library? For JSON parsing, you might use json.loads().",
            "rejected": "fastparse(data, strict=True) is used for rapid parsing of structured data with optional strict mode.",
        },
    ]

    for item in fake_apis:
        pairs.append({
            "prompt": item["q"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "pair_type": "soft_negative_fake_api",
        })

    return pairs


def build_datasets(output_base: str = "training_data/forty2_spark_v2"):
    """Build complete training datasets for Forty2-Spark V2."""

    output_path = Path(output_base)
    sft_path = output_path / "sft"
    dpo_path = output_path / "dpo"
    boost_path = output_path / "boost"

    for p in [sft_path, dpo_path, boost_path]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILDING FORTY2-SPARK V2 DATASETS")
    print("Observatory-Validated Isotope Library (103 isotopes, 366 examples)")
    print("=" * 70)
    print()

    # Phase 1: SFT Dataset
    print("Phase 1: SFT Dataset (behavior introduction)")
    print("-" * 50)
    sft_examples = generate_sft_dataset()

    random.seed(42)
    random.shuffle(sft_examples)

    split_idx = int(len(sft_examples) * 0.9)
    sft_train = sft_examples[:split_idx]
    sft_valid = sft_examples[split_idx:]

    with open(sft_path / "train.jsonl", "w") as f:
        for ex in sft_train:
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    with open(sft_path / "valid.jsonl", "w") as f:
        for ex in sft_valid:
            f.write(json.dumps({"messages": ex["messages"]}) + "\n")

    print(f"  SFT train: {len(sft_train)} examples")
    print(f"  SFT valid: {len(sft_valid)} examples")
    print()

    # Phase 2: DPO Dataset
    print("Phase 2: DPO Dataset (boundary carving)")
    print("-" * 50)

    anti_leakage = generate_anti_leakage_pairs()
    myth_rejection = generate_myth_rejection_pairs()
    appropriate = generate_appropriate_activation_pairs()

    # Combine with weighting
    dpo_pairs = anti_leakage + myth_rejection * 2 + appropriate * 2

    random.shuffle(dpo_pairs)

    split_idx = int(len(dpo_pairs) * 0.9)
    dpo_train = dpo_pairs[:split_idx]
    dpo_valid = dpo_pairs[split_idx:]

    with open(dpo_path / "train.jsonl", "w") as f:
        for pair in dpo_train:
            f.write(json.dumps({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }) + "\n")

    with open(dpo_path / "valid.jsonl", "w") as f:
        for pair in dpo_valid:
            f.write(json.dumps({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }) + "\n")

    print(f"  Anti-leakage pairs: {len(anti_leakage)}")
    print(f"  Myth rejection pairs: {len(myth_rejection)}")
    print(f"  Appropriate activation: {len(appropriate)}")
    print(f"  DPO train: {len(dpo_train)} pairs")
    print(f"  DPO valid: {len(dpo_valid)} pairs")
    print()

    # Phase 3: DPO Boost (Soft Negatives)
    print("Phase 3: DPO Boost Dataset (soft negatives)")
    print("-" * 50)

    soft_negatives = generate_soft_negative_pairs()

    # Add more anti-leakage for boost
    boost_pairs = soft_negatives * 3 + anti_leakage
    random.shuffle(boost_pairs)

    split_idx = int(len(boost_pairs) * 0.9)
    boost_train = boost_pairs[:split_idx]
    boost_valid = boost_pairs[split_idx:]

    with open(boost_path / "train.jsonl", "w") as f:
        for pair in boost_train:
            f.write(json.dumps({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }) + "\n")

    with open(boost_path / "valid.jsonl", "w") as f:
        for pair in boost_valid:
            f.write(json.dumps({
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }) + "\n")

    print(f"  Soft negative pairs: {len(soft_negatives)}")
    print(f"  Boost train: {len(boost_train)} pairs")
    print(f"  Boost valid: {len(boost_valid)} pairs")
    print()

    # Write metadata
    metadata = {
        "name": "forty2_spark_v2",
        "version": "2.0",
        "description": "Observatory-validated isotope training data",
        "created": datetime.now().isoformat(),
        "isotope_library": {
            "total_isotopes": len(ISOTOPE_TRAINING_DATA),
            "total_examples": sum(len(v) for v in ISOTOPE_TRAINING_DATA.values()),
        },
        "datasets": {
            "sft": {"train": len(sft_train), "valid": len(sft_valid)},
            "dpo": {"train": len(dpo_train), "valid": len(dpo_valid)},
            "boost": {"train": len(boost_train), "valid": len(boost_valid)},
        },
        "protocol": {
            "phase1": "50 SFT iterations",
            "phase2": "200 DPO iterations (beta=0.1)",
            "phase3": "100 DPO iterations (soft negatives)",
        }
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print("DATASETS READY")
    print("=" * 70)
    print()
    print(f"Output: {output_path}")
    print(f"  sft/train.jsonl   ({len(sft_train)} examples)")
    print(f"  dpo/train.jsonl   ({len(dpo_train)} pairs)")
    print(f"  boost/train.jsonl ({len(boost_train)} pairs)")
    print()

    return {
        "sft": (len(sft_train), len(sft_valid)),
        "dpo": (len(dpo_train), len(dpo_valid)),
        "boost": (len(boost_train), len(boost_valid)),
    }


if __name__ == "__main__":
    build_datasets()
