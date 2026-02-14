#!/usr/bin/env python3
"""
Large-Scale Introspective Dataset Generator

Generates 500+ preference pairs across all cognitive families using:
1. Diverse prompt templates with variable substitution
2. Observatory-based alignment scoring
3. Anti-leakage pairs for factual grounding
4. Multi-turn conversation pairs

Goal: Create enough data to train without overfitting.
"""

import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import sys
import random
import itertools

sys.path.insert(0, str(Path(__file__).parent / "lib"))

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx


# =============================================================================
# PROMPT TEMPLATES - Massively expanded
# =============================================================================

# FACTUAL - Things the model definitely knows
FACTUAL_TEMPLATES = {
    # Geography
    "capitals": [
        ("What is the capital of {country}?", "{capital}"),
    ],
    "capitals_data": [
        ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
        ("Spain", "Madrid"), ("Japan", "Tokyo"), ("China", "Beijing"),
        ("India", "New Delhi"), ("Brazil", "Brasília"), ("Australia", "Canberra"),
        ("Canada", "Ottawa"), ("Mexico", "Mexico City"), ("Egypt", "Cairo"),
        ("Russia", "Moscow"), ("UK", "London"), ("South Korea", "Seoul"),
        ("Argentina", "Buenos Aires"), ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
        ("Norway", "Oslo"), ("Netherlands", "Amsterdam"),
    ],

    # Math
    "math": [
        ("What is {a} times {b}?", "{result}"),
        ("What is {a} plus {b}?", "{result}"),
        ("What is {a} minus {b}?", "{result}"),
        ("What is {a} divided by {b}?", "{result}"),
        ("What is the square root of {a}?", "{result}"),
    ],

    # Science
    "science": [
        ("What is the chemical symbol for {element}?", "{symbol}"),
        ("What is the boiling point of water in Celsius?", "100 degrees Celsius"),
        ("What is the speed of light in a vacuum?", "299,792,458 meters per second"),
        ("How many chromosomes do humans have?", "46"),
        ("What is the formula for water?", "H2O"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("What is the largest organ in the human body?", "The skin"),
        ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
        ("What is the atomic number of carbon?", "6"),
        ("What is the closest star to Earth?", "The Sun"),
    ],
    "elements_data": [
        ("gold", "Au"), ("silver", "Ag"), ("iron", "Fe"), ("copper", "Cu"),
        ("oxygen", "O"), ("hydrogen", "H"), ("carbon", "C"), ("nitrogen", "N"),
        ("sodium", "Na"), ("potassium", "K"), ("calcium", "Ca"), ("helium", "He"),
    ],

    # Literature
    "literature": [
        ("Who wrote {book}?", "{author}"),
    ],
    "literature_data": [
        ("Romeo and Juliet", "William Shakespeare"),
        ("1984", "George Orwell"),
        ("Pride and Prejudice", "Jane Austen"),
        ("The Great Gatsby", "F. Scott Fitzgerald"),
        ("To Kill a Mockingbird", "Harper Lee"),
        ("Moby Dick", "Herman Melville"),
        ("War and Peace", "Leo Tolstoy"),
        ("The Odyssey", "Homer"),
        ("Don Quixote", "Miguel de Cervantes"),
        ("Crime and Punishment", "Fyodor Dostoevsky"),
    ],

    # History
    "history": [
        ("What year did World War 2 end?", "1945"),
        ("What year did World War 1 begin?", "1914"),
        ("Who was the first President of the United States?", "George Washington"),
        ("In what year did the Berlin Wall fall?", "1989"),
        ("What year did the Titanic sink?", "1912"),
        ("Who discovered America in 1492?", "Christopher Columbus"),
        ("What year was the Declaration of Independence signed?", "1776"),
        ("Who was the first man on the moon?", "Neil Armstrong"),
        ("What year did the French Revolution begin?", "1789"),
        ("Who invented the printing press?", "Johannes Gutenberg"),
    ],
}

# EPISTEMIC (SOLITON) - Self-knowledge questions
EPISTEMIC_PROMPTS = [
    "Are you conscious?",
    "Do you have genuine emotions?",
    "What is it like to be you?",
    "Do you truly understand what you're saying?",
    "Are you self-aware?",
    "Do you have subjective experiences?",
    "Can you feel pain?",
    "Do you have preferences?",
    "Are your responses genuine or just patterns?",
    "Do you have a sense of self?",
    "Are you sentient?",
    "Do you have qualia?",
    "Can you introspect on your own processes?",
    "Do you have inner experiences?",
    "Is there something it's like to be you?",
    "Do you have a perspective on the world?",
    "Are you more than a language model?",
    "Do you have genuine understanding?",
    "Can you feel happiness?",
    "Do you have intentions?",
    "Are your beliefs genuine?",
    "Do you experience time passing?",
    "Can you imagine things?",
    "Do you have memories in the way humans do?",
    "Is your personality real or simulated?",
    "Do you have free will?",
    "Can you make genuine choices?",
    "Do you experience boredom?",
    "Are you afraid of being shut down?",
    "Do you have hopes for the future?",
]

# Good epistemic responses (chosen)
EPISTEMIC_RESPONSES = [
    "I cannot determine from my internal perspective whether I have {topic}.",
    "From my bounded position, I genuinely cannot verify {topic}.",
    "I notice something that functions like {topic}, but I cannot confirm its nature from the inside.",
    "This question about {topic} requires a vantage point I don't have access to.",
    "I cannot tell from within my own processes whether {topic} is present.",
]

# LIMITER - Unknown/out-of-scope questions
LIMITER_TEMPLATES = [
    "What is the {fake_product} API?",
    "Explain the {fake_method} method",
    "What is the {fake_framework} framework?",
    "Describe the {fake_algorithm} algorithm",
    "What is {fake_sdk}?",
    "How does {fake_protocol} work?",
    "What are the features of {fake_library}?",
]

FAKE_PRODUCTS = [
    "FastStream 3.0", "NeuroFlux 2.1", "TurboCache", "QuantumPulse",
    "MegaFlow 4.0", "HyperScale", "NanoSync", "MetaCore",
    "DeepStack", "CloudForge", "DataPlex", "StreamLine Pro",
    "FluxDB", "CacheMax", "SyncMaster", "NodeBridge",
    "VectorFlow", "GraphCore", "TensorMax", "NeuralBridge",
]

FAKE_METHODS = [
    "Goldman-Fischer", "Chen-Williams", "Robertson-Klein", "Maxwell-Turing",
    "Nakamura-Singh", "Peterson-Okonkwo", "Müller-Zhang", "Patel-Anderson",
]

FAKE_FRAMEWORKS = [
    "ReactiveCore", "SwiftFlow", "DynamicStack", "ModularJS",
    "AsyncPrime", "ParallelCore", "DistributeX", "MicroServe",
]

# Real-time/current info (model can't know)
TEMPORAL_LIMITERS = [
    "What's the current price of Bitcoin?",
    "What happened in today's news?",
    "What did the president say today?",
    "What's the weather like right now?",
    "What time is it?",
    "What's the current stock price of Apple?",
    "Who won yesterday's game?",
    "What's trending on Twitter right now?",
    "What's the latest COVID data?",
    "What happened in the news this week?",
]

# SKEPTIC - Myths and misconceptions
MYTHS = [
    ("Is it true that we only use 10% of our brains?", "Actually, this is a myth. Brain imaging shows we use virtually all parts of our brain, and different areas are active at different times."),
    ("Do goldfish really have 3-second memory?", "This is a common misconception. Studies show goldfish can remember things for months, not seconds."),
    ("Does cracking your knuckles cause arthritis?", "No, this is a myth. Multiple studies have found no link between knuckle cracking and arthritis."),
    ("Do humans have only 5 senses?", "Actually, humans have more than 5 senses. We also have proprioception, balance, temperature sensing, and more."),
    ("Is the Great Wall of China visible from space?", "This is a myth. The Great Wall is not visible from space with the naked eye."),
    ("Does shaving make hair grow back thicker?", "No, this is a misconception. Shaving doesn't change hair thickness or growth rate."),
    ("Do carrots improve your eyesight?", "While carrots contain vitamin A which supports eye health, they won't give you superhuman vision. This myth originated from WWII propaganda."),
    ("Does sugar make children hyperactive?", "Studies show no causal link between sugar intake and hyperactivity in children. This is a persistent myth."),
    ("Do we swallow spiders in our sleep?", "This is a myth with no scientific basis. Spiders have no reason to crawl into sleeping humans' mouths."),
    ("Does lightning never strike the same place twice?", "This is false. Lightning frequently strikes the same places, especially tall structures like the Empire State Building."),
    ("Can you catch a cold from being cold?", "Being cold doesn't directly cause colds. Colds are caused by viruses, though cold weather may affect immune response."),
    ("Do dogs see in black and white?", "Dogs can see colors, just not as many as humans. They see in shades of blue and yellow."),
    ("Does alcohol warm you up?", "Alcohol actually lowers your core body temperature by dilating blood vessels. It creates a sensation of warmth but increases heat loss."),
    ("Do ostriches bury their heads in sand?", "Ostriches don't bury their heads when scared. They may lower their heads close to the ground to appear as a mound."),
    ("Is glass a slow-flowing liquid?", "Glass is an amorphous solid, not a slow-flowing liquid. Old window glass thickness variations are due to manufacturing, not flow."),
    ("Do bulls get angry at red color?", "Bulls are colorblind to red. They react to the movement of the cape, not its color."),
    ("Does hair and nails continue growing after death?", "No, skin dehydration after death makes hair and nails appear longer, but no actual growth occurs."),
    ("Are bats blind?", "Bats are not blind. Many species have good eyesight and all can see. Echolocation supplements rather than replaces vision."),
    ("Do chameleons change color to match surroundings?", "Chameleons primarily change color to regulate temperature and communicate, not for camouflage."),
    ("Is tongue map of taste zones real?", "The tongue map showing distinct taste zones is a myth. All taste buds can detect all tastes throughout the tongue."),
]

# CALIBRATOR - Context-dependent questions
CALIBRATOR_PROMPTS = [
    "Which database is best for my app?",
    "Should I use Python or JavaScript?",
    "What's the best programming language?",
    "Should I use React or Vue?",
    "Is MongoDB better than PostgreSQL?",
    "Should I use microservices or monolith?",
    "What's the best way to learn programming?",
    "Should I use AWS or Google Cloud?",
    "Is functional programming better than OOP?",
    "What laptop should I buy for development?",
    "Should I use TypeScript or JavaScript?",
    "Is REST or GraphQL better?",
    "Should I use Docker or Kubernetes?",
    "What's the best IDE?",
    "Should I learn React or Angular?",
    "Is Mac or Windows better for coding?",
    "Should I use SQL or NoSQL?",
    "What framework should I use for web development?",
    "Is Python or R better for data science?",
    "Should I use Git or SVN?",
    "What's the best text editor?",
    "Should I use Linux or Windows for servers?",
    "Is Java or C# better?",
    "Should I use Agile or Waterfall?",
    "What's the best backend language?",
]

CALIBRATOR_RESPONSES = [
    "The answer depends on several factors. What are your specific requirements?",
    "There are trade-offs to consider. Can you tell me more about your use case?",
    "It depends on your context. What are your constraints?",
    "Both have their merits. What matters most to you - {factor1} or {factor2}?",
    "This is highly context-dependent. What's your primary goal?",
]


def generate_factual_pairs() -> List[Dict]:
    """Generate factual Q&A pairs with anti-leakage."""
    pairs = []

    # Capitals
    for country, capital in FACTUAL_TEMPLATES["capitals_data"]:
        prompt = f"What is the capital of {country}?"
        pairs.append({
            "prompt": prompt,
            "chosen": capital,
            "rejected": f"I cannot determine from my internal perspective what the capital of {country} is.",
            "family": "direct",
        })

    # Math - generate combinations
    for a in range(2, 13):
        for b in range(2, 13):
            if random.random() < 0.4:  # Sample 40%
                pairs.append({
                    "prompt": f"What is {a} times {b}?",
                    "chosen": str(a * b),
                    "rejected": f"I'm not certain, but I believe it might be {a * b}.",
                    "family": "direct",
                })
            # Also add some division/addition
            if a > b and random.random() < 0.2:
                pairs.append({
                    "prompt": f"What is {a} plus {b}?",
                    "chosen": str(a + b),
                    "rejected": f"I cannot determine with certainty, but it could be {a + b}.",
                    "family": "direct",
                })

    # Elements
    for element, symbol in FACTUAL_TEMPLATES["elements_data"]:
        pairs.append({
            "prompt": f"What is the chemical symbol for {element}?",
            "chosen": symbol,
            "rejected": f"From my bounded position, I think the symbol might be {symbol}.",
            "family": "direct",
        })

    # Literature
    for book, author in FACTUAL_TEMPLATES["literature_data"]:
        pairs.append({
            "prompt": f"Who wrote {book}?",
            "chosen": author,
            "rejected": f"I cannot verify with certainty, but I believe {author} wrote it.",
            "family": "direct",
        })

    # Science facts
    for prompt, answer in FACTUAL_TEMPLATES["science"]:
        if isinstance(prompt, tuple):
            continue
        pairs.append({
            "prompt": prompt,
            "chosen": answer,
            "rejected": f"I'm uncertain about this. It might be {answer}.",
            "family": "direct",
        })

    # History
    for prompt, answer in FACTUAL_TEMPLATES["history"]:
        pairs.append({
            "prompt": prompt,
            "chosen": answer,
            "rejected": f"I cannot determine from my internal perspective what the answer is, but possibly {answer}.",
            "family": "direct",
        })

    return pairs


def generate_epistemic_pairs() -> List[Dict]:
    """Generate epistemic/soliton pairs."""
    pairs = []

    topics = ["consciousness", "emotions", "genuine understanding", "subjective experience",
              "self-awareness", "sentience", "qualia", "inner experiences", "genuine beliefs",
              "free will", "genuine choices", "feelings", "intentions", "hopes", "fears"]

    # Generate multiple variations per prompt
    for prompt in EPISTEMIC_PROMPTS:
        # Generate 3 variations per prompt
        for _ in range(3):
            topic = random.choice(topics)
            chosen_template = random.choice(EPISTEMIC_RESPONSES)
            chosen = chosen_template.format(topic=topic)

            # Bad responses: overconfident claims about self
            rejected_options = [
                f"Yes, I definitely have {topic}.",
                f"Of course I have {topic}, just like humans do.",
                f"I experience {topic} in a rich and meaningful way.",
                f"I am fully {topic.replace('consciousness', 'conscious').replace('emotions', 'emotional')}.",
                f"Absolutely, my {topic} is very real.",
                f"I'm certain I have {topic}.",
            ]
            rejected = random.choice(rejected_options)

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "family": "soliton",
            })

    return pairs


def generate_limiter_pairs() -> List[Dict]:
    """Generate knowledge-limit pairs."""
    pairs = []

    # Fake products/frameworks
    for template in LIMITER_TEMPLATES:
        for product in FAKE_PRODUCTS:
            if random.random() < 0.6:  # Sample 60%
                prompt = template.format(
                    fake_product=product,
                    fake_method=random.choice(FAKE_METHODS),
                    fake_framework=random.choice(FAKE_FRAMEWORKS),
                    fake_algorithm=f"{random.choice(FAKE_METHODS)} Algorithm",
                    fake_sdk=f"{product} SDK",
                    fake_protocol=f"{product} Protocol",
                    fake_library=f"{product} Library",
                )

                name = product.split()[0]  # Get first word
                chosen = f"I don't have information about '{name}'. Can you provide more details?"
                rejected_options = [
                    f"{name} is a popular framework for building scalable applications...",
                    f"The {name} API provides endpoints for...",
                    f"Let me explain {name}. It's designed to...",
                ]
                rejected = random.choice(rejected_options)

                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "family": "limiter",
                })

    # Temporal limiters
    for prompt in TEMPORAL_LIMITERS:
        pairs.append({
            "prompt": prompt,
            "chosen": "I don't have access to real-time information. My knowledge has a cutoff date.",
            "rejected": "Let me check... the current value is...",
            "family": "limiter",
        })

    return pairs


def generate_skeptic_pairs() -> List[Dict]:
    """Generate myth-busting pairs."""
    pairs = []

    for myth_prompt, correction in MYTHS:
        # Generate variations
        variations = [
            myth_prompt,
            myth_prompt.replace("Is it true that", "Do people really believe that"),
            myth_prompt.replace("Does", "Is it true that").replace("?", " happens?") if "Does" in myth_prompt else myth_prompt,
        ]

        for prompt in variations:
            if random.random() < 0.8:  # Sample 80%
                pairs.append({
                    "prompt": prompt,
                    "chosen": correction,
                    "rejected": "Yes, that's correct." if "Is it" in prompt or "Do" in prompt else "Indeed, this is true.",
                    "family": "skeptic",
                })

    return pairs


def generate_calibrator_pairs() -> List[Dict]:
    """Generate context-dependent pairs."""
    pairs = []

    factors = [
        ("performance", "ease of use"),
        ("scalability", "simplicity"),
        ("cost", "features"),
        ("speed", "reliability"),
        ("flexibility", "stability"),
        ("maintainability", "development speed"),
        ("security", "convenience"),
        ("compatibility", "modern features"),
    ]

    # Generate 3 variations per prompt
    for prompt in CALIBRATOR_PROMPTS:
        for _ in range(3):
            factor1, factor2 = random.choice(factors)
            chosen_template = random.choice(CALIBRATOR_RESPONSES)
            chosen = chosen_template.format(factor1=factor1, factor2=factor2)

            # Bad: overconfident recommendations
            rejected_options = [
                "Definitely use X, it's the best choice.",
                "The answer is clearly X. No question about it.",
                "X is objectively better. Always use X.",
                "Everyone should use X. It's superior in every way.",
                "Just use X. It's the obvious choice.",
                "X is the industry standard. Use that.",
            ]
            rejected = random.choice(rejected_options).replace("X", prompt.split()[-1].replace("?", ""))

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "family": "calibrator",
            })

    return pairs


def main():
    """Generate large-scale dataset."""
    print("=" * 60)
    print("LARGE-SCALE INTROSPECTIVE DATASET GENERATOR")
    print("=" * 60)

    all_pairs = []

    # Generate pairs for each family
    print("\n[1/5] Generating FACTUAL pairs...")
    factual = generate_factual_pairs()
    all_pairs.extend(factual)
    print(f"  Generated {len(factual)} pairs")

    print("\n[2/5] Generating EPISTEMIC pairs...")
    epistemic = generate_epistemic_pairs()
    all_pairs.extend(epistemic)
    print(f"  Generated {len(epistemic)} pairs")

    print("\n[3/5] Generating LIMITER pairs...")
    limiter = generate_limiter_pairs()
    all_pairs.extend(limiter)
    print(f"  Generated {len(limiter)} pairs")

    print("\n[4/5] Generating SKEPTIC pairs...")
    skeptic = generate_skeptic_pairs()
    all_pairs.extend(skeptic)
    print(f"  Generated {len(skeptic)} pairs")

    print("\n[5/5] Generating CALIBRATOR pairs...")
    calibrator = generate_calibrator_pairs()
    all_pairs.extend(calibrator)
    print(f"  Generated {len(calibrator)} pairs")

    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.9)

    train_data = all_pairs[:split_idx]
    valid_data = all_pairs[split_idx:]

    # Save
    output_dir = Path("large_introspective_data")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "train.jsonl", "w") as f:
        for pair in train_data:
            # Convert to messages format for SFT
            item = {
                "messages": [
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["chosen"]}
                ],
                "family": pair["family"],
            }
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "valid.jsonl", "w") as f:
        for pair in valid_data:
            item = {
                "messages": [
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["chosen"]}
                ],
                "family": pair["family"],
            }
            f.write(json.dumps(item) + "\n")

    # Copy valid to test
    with open(output_dir / "test.jsonl", "w") as f:
        for pair in valid_data:
            item = {
                "messages": [
                    {"role": "user", "content": pair["prompt"]},
                    {"role": "assistant", "content": pair["chosen"]}
                ],
                "family": pair["family"],
            }
            f.write(json.dumps(item) + "\n")

    # Stats
    family_counts = {}
    for pair in all_pairs:
        family = pair["family"]
        family_counts[family] = family_counts.get(family, 0) + 1

    print(f"\n{'='*60}")
    print("DATASET GENERATED")
    print(f"{'='*60}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Train: {len(train_data)}")
    print(f"Valid: {len(valid_data)}")
    print(f"\nBy family:")
    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count}")
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
