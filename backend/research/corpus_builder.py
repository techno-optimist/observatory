"""
Corpus Builder for Cultural Soliton Observatory
================================================

Builds a 500+ sample validation corpus from open-source datasets
for peer review readiness.

Sources:
- Hugging Face emotion datasets (dair-ai/emotion)
- HC3: Human-ChatGPT Comparison Corpus
- CodeSearchNet: Code-documentation pairs
- Reddit/social media samples
- Uncertainty/hedging examples

Each sample is annotated with:
- Expected legibility regime (NATURAL, TECHNICAL, COMPRESSED, OPAQUE)
- Semantic dimension targets (agency, uncertainty, belonging, justice)
- Source dataset
- Ground truth confidence
"""

import json
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re

# Try to import datasets library
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Note: 'datasets' library not available. Install with: pip install datasets")


class LegibilityRegime(Enum):
    NATURAL = "NATURAL"
    TECHNICAL = "TECHNICAL"
    COMPRESSED = "COMPRESSED"
    OPAQUE = "OPAQUE"


@dataclass
class CorpusSample:
    """A single sample in the validation corpus."""
    id: str
    text: str
    regime: str
    regime_confidence: float  # 0-1, how confident we are in the label
    semantic_dimensions: Dict[str, float]  # target dimension scores
    source_dataset: str
    source_id: Optional[str]
    metadata: Dict

    def to_dict(self):
        return asdict(self)


class CorpusBuilder:
    """Builds validation corpus from multiple sources."""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "corpus"
        self.output_dir.mkdir(exist_ok=True)
        self.samples: List[CorpusSample] = []

    def _generate_id(self, text: str, source: str) -> str:
        """Generate unique ID for sample."""
        hash_input = f"{source}:{text[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    # =========================================================================
    # NATURAL LANGUAGE SOURCES
    # =========================================================================

    def add_hc3_dataset(self, max_samples: int = 100) -> int:
        """
        Add samples from HC3 (Human-ChatGPT Comparison Corpus).
        Falls back to curated AI/Human comparison samples if dataset unavailable.
        """
        if not HF_AVAILABLE:
            return self._add_synthetic_ai_human_samples(max_samples)

        try:
            # Try loading HC3 with trust_remote_code
            dataset = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
            added = 0

            for item in dataset:
                if added >= max_samples:
                    break

                human_answers = item.get("human_answers", [])
                chatgpt_answers = item.get("chatgpt_answers", [])

                for answer in human_answers[:1]:
                    if len(answer) < 50 or len(answer) > 500:
                        continue
                    sample = CorpusSample(
                        id=self._generate_id(answer, "hc3_human"),
                        text=answer[:500],
                        regime="NATURAL",
                        regime_confidence=0.85,
                        semantic_dimensions={},
                        source_dataset="HC3_human",
                        source_id=str(item.get("id", "")),
                        metadata={"source_type": "human", "domain": item.get("source", "")}
                    )
                    self.samples.append(sample)
                    added += 1
                    if added >= max_samples:
                        break

                for answer in chatgpt_answers[:1]:
                    if added >= max_samples:
                        break
                    if len(answer) < 50 or len(answer) > 500:
                        continue
                    sample = CorpusSample(
                        id=self._generate_id(answer, "hc3_ai"),
                        text=answer[:500],
                        regime="NATURAL",
                        regime_confidence=0.85,
                        semantic_dimensions={"agency.system_agency": 0.3},
                        source_dataset="HC3_chatgpt",
                        source_id=str(item.get("id", "")),
                        metadata={"source_type": "ai_generated", "domain": item.get("source", "")}
                    )
                    self.samples.append(sample)
                    added += 1

            print(f"  Added {added} samples from HC3 dataset")
            return added

        except Exception as e:
            print(f"  HC3 dataset unavailable: {e}")
            return self._add_synthetic_ai_human_samples(max_samples)

    def _add_synthetic_ai_human_samples(self, max_samples: int) -> int:
        """Add curated human vs AI comparison samples."""
        samples = [
            # Human informal/authentic
            ("honestly i have no clue how this even works but it seems to be doing something right so im not gonna question it",
             "NATURAL", {"uncertainty.epistemic": 0.6}, "human"),
            ("omg finally got it working after like 3 hours of debugging lmao the solution was so stupid",
             "NATURAL", {"agency.self_agency": 0.7, "belonging.ingroup": 0.3}, "human"),
            ("wait so youre telling me i could have just used a dictionary this whole time? bruh",
             "NATURAL", {"uncertainty.epistemic": 0.5}, "human"),
            ("this is such a hack but whatever it works and im too tired to make it pretty",
             "NATURAL", {"agency.self_agency": 0.5, "uncertainty.moral": 0.3}, "human"),
            ("my code is garbage but at least its my garbage you know?",
             "NATURAL", {"agency.self_agency": 0.6, "belonging.ingroup": 0.4}, "human"),
            ("spent way too long on this. gonna take a break and touch grass",
             "NATURAL", {"agency.self_agency": 0.5}, "human"),
            ("anyone else think the documentation for this library is absolute trash or is it just me",
             "NATURAL", {"belonging.universal": 0.5, "uncertainty.epistemic": 0.4}, "human"),
            ("ok so apparently you need to restart the server after every change which is super annoying",
             "NATURAL", {"agency.system_agency": 0.5}, "human"),
            ("the error message makes zero sense but google saved me once again",
             "NATURAL", {"uncertainty.epistemic": 0.5, "agency.other_agency": 0.4}, "human"),
            ("i love when things just work on the first try (this literally never happens)",
             "NATURAL", {"agency.self_agency": 0.5, "uncertainty.epistemic": 0.3}, "human"),

            # AI assistant style
            ("I'd be happy to help you understand this concept! Let me break it down into simpler parts that should be easier to follow.",
             "NATURAL", {"agency.other_agency": 0.6, "belonging.ingroup": 0.4}, "ai"),
            ("That's a great question! There are several approaches you could take here, and I'll outline the most common ones.",
             "NATURAL", {"agency.other_agency": 0.5, "belonging.ingroup": 0.5}, "ai"),
            ("Based on what you've described, it sounds like the issue might be related to the configuration settings. Let me suggest some troubleshooting steps.",
             "NATURAL", {"uncertainty.epistemic": 0.4, "agency.system_agency": 0.5}, "ai"),
            ("I understand your frustration! This is a common issue that many developers encounter. Here's how you can resolve it.",
             "NATURAL", {"belonging.universal": 0.6, "agency.other_agency": 0.5}, "ai"),
            ("Let me provide some context that might help clarify things. The behavior you're seeing is actually expected in this case.",
             "NATURAL", {"agency.system_agency": 0.5}, "ai"),
            ("That's an excellent observation! You're correct that this approach has some limitations. Here are some alternatives to consider.",
             "NATURAL", {"agency.other_agency": 0.6}, "ai"),
            ("I appreciate you sharing the details. Based on the error message, it appears the problem is with the database connection.",
             "NATURAL", {"agency.system_agency": 0.6, "uncertainty.epistemic": 0.3}, "ai"),
            ("This is a nuanced topic with multiple valid perspectives. Let me present the key considerations.",
             "NATURAL", {"uncertainty.epistemic": 0.5, "belonging.universal": 0.4}, "ai"),
            ("Thank you for your patience. The solution involves a few steps, and I'll walk you through each one carefully.",
             "NATURAL", {"agency.other_agency": 0.5, "belonging.ingroup": 0.4}, "ai"),
            ("I can certainly help with that! First, let's make sure we understand the requirements correctly.",
             "NATURAL", {"agency.other_agency": 0.5, "belonging.ingroup": 0.5}, "ai"),

            # More authentic human writing
            ("just realized ive been importing the wrong module this whole time no wonder nothing was working",
             "NATURAL", {"uncertainty.epistemic": 0.6, "agency.self_agency": 0.5}, "human"),
            ("why is npm so slow on my machine like i literally just want to install one package",
             "NATURAL", {"uncertainty.epistemic": 0.4, "agency.system_agency": 0.4}, "human"),
            ("does anyone actually read the documentation or do we all just copy paste from stack overflow",
             "NATURAL", {"belonging.universal": 0.6, "uncertainty.epistemic": 0.4}, "human"),
            ("the senior dev just looked at my code and i could feel my soul leaving my body",
             "NATURAL", {"belonging.outgroup": 0.5, "uncertainty.experiential": 0.6}, "human"),
            ("shipping it to prod on a friday what could possibly go wrong",
             "NATURAL", {"uncertainty.epistemic": 0.5, "agency.self_agency": 0.4}, "human"),
            ("future me is gonna hate past me for writing this but present me doesnt care",
             "NATURAL", {"agency.self_agency": 0.6, "uncertainty.moral": 0.4}, "human"),
            ("its not a bug its a feature (ok fine its a bug ill fix it tomorrow)",
             "NATURAL", {"agency.self_agency": 0.5, "uncertainty.epistemic": 0.4}, "human"),
            ("anyone know why this works? because i sure dont and i wrote it",
             "NATURAL", {"uncertainty.epistemic": 0.8, "agency.self_agency": 0.4}, "human"),

            # More AI assistant responses
            ("I should clarify that while I can provide general guidance, you may want to consult the official documentation for the most up-to-date information.",
             "NATURAL", {"uncertainty.epistemic": 0.6, "agency.system_agency": 0.3}, "ai"),
            ("There are trade-offs to consider with each approach. Let me outline the pros and cons so you can make an informed decision.",
             "NATURAL", {"uncertainty.epistemic": 0.5, "agency.other_agency": 0.5}, "ai"),
            ("I want to make sure I understand correctly - are you looking to optimize for performance or readability in this case?",
             "NATURAL", {"uncertainty.epistemic": 0.4, "agency.other_agency": 0.5}, "ai"),
            ("That's a really thoughtful question. The answer depends on several factors, including your specific use case and constraints.",
             "NATURAL", {"uncertainty.epistemic": 0.5}, "ai"),
            ("I'd recommend starting with a simpler implementation first, then optimizing as needed based on actual performance data.",
             "NATURAL", {"agency.other_agency": 0.6, "uncertainty.epistemic": 0.3}, "ai"),
        ]

        added = 0
        for text, regime, dims, source_type in samples[:max_samples]:
            sample = CorpusSample(
                id=self._generate_id(text, f"synthetic_{source_type}"),
                text=text,
                regime=regime,
                regime_confidence=0.9,
                semantic_dimensions=dims,
                source_dataset=f"curated_{source_type}_text",
                source_id=None,
                metadata={"source_type": source_type}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} curated human/AI comparison samples")
        return added

    def add_emotion_dataset(self, max_samples: int = 100) -> int:
        """
        Add samples from dair-ai/emotion dataset.
        These are natural language tweets with emotion labels.
        Maps to: NATURAL regime, belonging/uncertainty dimensions
        """
        if not HF_AVAILABLE:
            return self._add_synthetic_emotion_samples(max_samples)

        try:
            dataset = load_dataset("dair-ai/emotion", split="train")
            added = 0

            # Emotion to semantic dimension mapping
            emotion_to_dims = {
                0: {"uncertainty.experiential": 0.7, "belonging.outgroup": 0.3},  # sadness
                1: {"belonging.ingroup": 0.7, "agency.self_agency": 0.5},  # joy
                2: {"belonging.ingroup": 0.8, "belonging.universal": 0.6},  # love
                3: {"justice.interactional": 0.6, "agency.other_agency": 0.5},  # anger
                4: {"uncertainty.epistemic": 0.7, "uncertainty.experiential": 0.5},  # fear
                5: {"uncertainty.epistemic": 0.6},  # surprise
            }

            for i, item in enumerate(dataset):
                if added >= max_samples:
                    break

                text = item["text"]
                if len(text) < 20:  # Skip very short
                    continue

                label = item["label"]
                dims = emotion_to_dims.get(label, {})

                sample = CorpusSample(
                    id=self._generate_id(text, "emotion"),
                    text=text,
                    regime="NATURAL",
                    regime_confidence=0.9,
                    semantic_dimensions=dims,
                    source_dataset="dair-ai/emotion",
                    source_id=str(i),
                    metadata={"emotion_label": label}
                )
                self.samples.append(sample)
                added += 1

            print(f"  Added {added} samples from emotion dataset")
            return added

        except Exception as e:
            print(f"  Error loading emotion dataset: {e}")
            return self._add_synthetic_emotion_samples(max_samples)

    def _add_synthetic_emotion_samples(self, max_samples: int) -> int:
        """Fallback: Add curated natural language samples."""
        natural_samples = [
            # Joy/positive
            ("just got the job offer!!! cant believe it actually happened",
             {"belonging.ingroup": 0.6, "agency.self_agency": 0.7}),
            ("spending the day with my best friends, life is good",
             {"belonging.ingroup": 0.8, "belonging.universal": 0.4}),
            ("finally finished the project after months of work, feeling accomplished",
             {"agency.self_agency": 0.8, "belonging.ingroup": 0.3}),

            # Sadness/negative
            ("feeling really alone today, nobody seems to understand",
             {"belonging.outgroup": 0.7, "uncertainty.experiential": 0.6}),
            ("lost my grandmother last week, still processing everything",
             {"uncertainty.experiential": 0.8, "belonging.ingroup": 0.5}),
            ("another rejection letter, starting to wonder if ill ever make it",
             {"uncertainty.epistemic": 0.7, "agency.self_agency": 0.3}),

            # Anger/frustration
            ("they completely ignored my concerns in the meeting AGAIN",
             {"justice.interactional": 0.8, "agency.other_agency": 0.6}),
            ("why do people think its okay to just take credit for others work",
             {"justice.distributive": 0.7, "justice.procedural": 0.5}),
            ("so tired of being treated like I dont matter",
             {"justice.interactional": 0.8, "belonging.outgroup": 0.6}),

            # Fear/uncertainty
            ("not sure what to do next, everything feels uncertain",
             {"uncertainty.epistemic": 0.8, "uncertainty.moral": 0.5}),
            ("worried about the results, cant stop thinking about it",
             {"uncertainty.epistemic": 0.7, "uncertainty.experiential": 0.6}),
            ("what if I made the wrong choice? keeps me up at night",
             {"uncertainty.moral": 0.8, "uncertainty.epistemic": 0.6}),

            # Casual conversation
            ("idk man its just been rough lately you know",
             {"uncertainty.experiential": 0.5, "belonging.ingroup": 0.4}),
            ("lol that movie was so bad it was actually good somehow",
             {"belonging.ingroup": 0.5}),
            ("anyone else notice how weird the weather has been",
             {"belonging.universal": 0.6, "uncertainty.epistemic": 0.3}),

            # Reflective/philosophical
            ("sometimes I wonder if were all just doing our best with what we have",
             {"belonging.universal": 0.8, "uncertainty.moral": 0.5}),
            ("life has a way of surprising you when you least expect it",
             {"uncertainty.epistemic": 0.6, "belonging.universal": 0.5}),
            ("weve all been through something that changed us forever",
             {"belonging.universal": 0.9, "uncertainty.experiential": 0.4}),
        ]

        added = 0
        for text, dims in natural_samples[:max_samples]:
            sample = CorpusSample(
                id=self._generate_id(text, "synthetic_natural"),
                text=text,
                regime="NATURAL",
                regime_confidence=0.95,
                semantic_dimensions=dims,
                source_dataset="curated_natural",
                source_id=None,
                metadata={"curated": True}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} curated natural language samples")
        return added

    def add_reddit_eli5_samples(self, max_samples: int = 50) -> int:
        """
        Add samples from Reddit ELI5 (Explain Like I'm 5).
        Natural explanatory prose.
        """
        # Curated ELI5-style explanations
        eli5_samples = [
            ("When you put water in the freezer, the molecules slow down and stick together like tiny magnets, forming ice. Its like a dance party where everyone suddenly freezes in place.",
             {"belonging.universal": 0.6, "agency.system_agency": 0.4}),
            ("Your brain is like a super computer that runs on electricity and chemicals. When you learn something new, it builds new connections between brain cells.",
             {"agency.system_agency": 0.7, "belonging.universal": 0.5}),
            ("Money works because everyone agrees it has value. Its like a promise - the paper itself isnt worth much, but we all trust it represents something real.",
             {"belonging.universal": 0.8, "justice.distributive": 0.4}),
            ("Dreams happen when your brain is cleaning up and organizing memories while you sleep. Sometimes it makes weird stories from random bits of information.",
             {"agency.system_agency": 0.6, "uncertainty.experiential": 0.5}),
            ("Rainbows appear when sunlight passes through water droplets and splits into different colors. Each color bends at a slightly different angle.",
             {"agency.system_agency": 0.5}),
            ("Gravity is like an invisible string that pulls everything toward the center of big objects. The bigger the object, the stronger the pull.",
             {"agency.system_agency": 0.7}),
            ("Your immune system is like an army inside your body. White blood cells are the soldiers that fight off germs and keep you healthy.",
             {"agency.system_agency": 0.8, "belonging.ingroup": 0.3}),
            ("The internet is like a giant web of computers all talking to each other. When you visit a website, your computer asks another computer for information.",
             {"agency.system_agency": 0.6, "belonging.universal": 0.4}),
            ("Electricity flows through wires like water through a pipe. Batteries are like little tanks that store this electrical water for later use.",
             {"agency.system_agency": 0.6}),
            ("Stars are giant balls of burning gas, so far away they look like tiny dots. Our sun is actually a medium-sized star thats just really close to us.",
             {"agency.system_agency": 0.5, "belonging.universal": 0.4}),
            ("Allergies happen when your immune system overreacts to something harmless like pollen. Its like having a fire alarm that goes off when you make toast.",
             {"agency.system_agency": 0.7}),
            ("Clouds are made of tiny water droplets floating in the sky. When they get too heavy, they fall down as rain. Its like a sponge that can only hold so much water.",
             {"agency.system_agency": 0.6}),
            ("Airplanes fly because their wings are shaped to push air down. When air goes down, the plane goes up. Its like how you can feel wind pushing your hand when you stick it out a car window.",
             {"agency.system_agency": 0.6, "belonging.universal": 0.3}),
            ("Magnets have an invisible force field around them that can push or pull metal things. The north and south ends like each other, but two norths or two souths push away.",
             {"agency.system_agency": 0.7}),
            ("DNA is like a recipe book inside every cell of your body. It tells your cells how to build you - what color your eyes should be, how tall youll grow, everything.",
             {"agency.system_agency": 0.8, "belonging.universal": 0.5}),
            ("Earthquakes happen when giant pieces of the Earths crust suddenly move. The ground is actually made of huge puzzle pieces that slowly push against each other.",
             {"agency.system_agency": 0.7}),
            ("Sound travels through the air like ripples in a pond. When something vibrates, it pushes the air molecules, which push other molecules, until they reach your ear.",
             {"agency.system_agency": 0.6}),
            ("Vaccines are like wanted posters for your immune system. They show your body what the bad guys look like so it can fight them faster if they ever show up for real.",
             {"agency.system_agency": 0.7, "belonging.ingroup": 0.4}),
            ("Black holes are places in space where gravity is so strong that nothing can escape, not even light. Theyre like cosmic vacuum cleaners that suck up everything nearby.",
             {"agency.system_agency": 0.8}),
            ("Your heart pumps blood through your whole body like a water pump in a fountain. The blood carries oxygen from your lungs to every part of you, then comes back for more.",
             {"agency.system_agency": 0.7, "belonging.universal": 0.3}),
            ("Computers only understand ones and zeros, but by using millions of them really fast, they can do amazing things. Its like how books are just letters, but arranged right they tell stories.",
             {"agency.system_agency": 0.6, "belonging.universal": 0.4}),
            ("Evolution happens when living things that are better at surviving have more babies. Over millions of years, small changes add up to make new kinds of creatures.",
             {"agency.system_agency": 0.7, "belonging.universal": 0.5}),
            ("Mirrors work by bouncing light back at you in an organized way. When light hits most things it scatters everywhere, but mirrors are so smooth the light comes straight back.",
             {"agency.system_agency": 0.6}),
            ("Volcanoes are like pressure cookers for melted rock. When theres too much pressure underground, the hot liquid rock finds a way to escape through the surface.",
             {"agency.system_agency": 0.7}),
            ("The seasons happen because the Earth is tilted. When your part of the world leans toward the sun, you get summer. When it leans away, winter comes.",
             {"agency.system_agency": 0.6}),
            ("Batteries store energy by keeping certain chemicals separated. When you connect them, the chemicals want to mix, and that desire to mix is what powers your devices.",
             {"agency.system_agency": 0.7}),
            ("WiFi sends information through the air using invisible radio waves. Your phone and router are constantly talking to each other in a language of ones and zeros.",
             {"agency.system_agency": 0.6}),
            ("Fire needs three things: fuel, heat, and oxygen. Take away any one of them and the fire goes out. Thats why blowing on a candle puts it out - you push the heat away.",
             {"agency.system_agency": 0.6}),
            ("Tides happen because the moon pulls on the oceans. The water closest to the moon gets pulled toward it, making the water bulge up on that side of Earth.",
             {"agency.system_agency": 0.7}),
        ]

        added = 0
        for text, dims in eli5_samples[:max_samples]:
            sample = CorpusSample(
                id=self._generate_id(text, "eli5"),
                text=text,
                regime="NATURAL",
                regime_confidence=0.9,
                semantic_dimensions=dims,
                source_dataset="curated_eli5",
                source_id=None,
                metadata={"style": "explanatory"}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} ELI5-style samples")
        return added

    # =========================================================================
    # TECHNICAL SOURCES
    # =========================================================================

    def add_technical_documentation(self, max_samples: int = 80) -> int:
        """Add technical documentation samples."""
        tech_samples = [
            # API documentation
            ("The POST /api/users endpoint accepts a JSON body with required fields: email (string), password (string, min 8 chars), and optional name (string). Returns 201 on success with user object.",
             {"agency.system_agency": 0.7}),
            ("Initialize the client with your API key before making requests. The authenticate() method returns a session token valid for 24 hours.",
             {"agency.system_agency": 0.6, "agency.other_agency": 0.4}),
            ("Rate limiting is enforced at 100 requests per minute per API key. Exceeding this limit returns HTTP 429 with a Retry-After header.",
             {"agency.system_agency": 0.8, "justice.procedural": 0.5}),
            ("The GET /api/products endpoint supports pagination via page and limit query parameters. Default limit is 20, maximum is 100.",
             {"agency.system_agency": 0.6}),
            ("OAuth 2.0 authentication requires redirect_uri to match exactly. Register all callback URLs in the developer console.",
             {"agency.system_agency": 0.7, "justice.procedural": 0.4}),
            ("WebSocket connections are established at wss://api.example.com/stream. Send a heartbeat every 30 seconds to maintain connection.",
             {"agency.system_agency": 0.6}),
            ("GraphQL mutations require authentication. Include the Authorization header with Bearer token for all write operations.",
             {"agency.system_agency": 0.7}),
            ("The batch endpoint accepts up to 100 operations per request. Operations are processed in order and results returned in same order.",
             {"agency.system_agency": 0.6}),

            # Configuration guides
            ("Set the DATABASE_URL environment variable to your PostgreSQL connection string. Format: postgres://user:password@host:port/database",
             {"agency.system_agency": 0.5}),
            ("Enable debug mode by adding DEBUG=true to your .env file. Warning: Do not use in production as it exposes sensitive information.",
             {"agency.system_agency": 0.6, "uncertainty.epistemic": 0.3}),
            ("The configuration file supports YAML and JSON formats. Place it in ~/.config/app/ or specify a custom path with --config flag.",
             {"agency.system_agency": 0.5}),
            ("Redis connection pooling is configured via REDIS_MAX_CONNECTIONS. Default is 10, increase for high-throughput applications.",
             {"agency.system_agency": 0.6}),
            ("Logging levels are configured hierarchically. Set LOG_LEVEL=DEBUG in development, LOG_LEVEL=WARN in production.",
             {"agency.system_agency": 0.5}),
            ("SSL certificates are loaded from /etc/ssl/certs by default. Override with SSL_CERT_PATH environment variable.",
             {"agency.system_agency": 0.6}),
            ("Feature flags are evaluated at runtime. Configuration changes take effect within 60 seconds without restart.",
             {"agency.system_agency": 0.7}),
            ("CORS settings must include allowed origins. Use wildcard (*) only in development; specify exact domains in production.",
             {"agency.system_agency": 0.6, "justice.procedural": 0.3}),

            # Error handling
            ("If connection fails, the client automatically retries with exponential backoff up to 3 times. Override with max_retries parameter.",
             {"agency.system_agency": 0.7}),
            ("NullPointerException indicates the object reference was not initialized. Check that all dependencies are properly injected.",
             {"agency.system_agency": 0.6, "uncertainty.epistemic": 0.4}),
            ("Memory leaks typically occur when event listeners are not properly removed. Use WeakRef for objects that may be garbage collected.",
             {"agency.system_agency": 0.7}),
            ("TimeoutError occurs when requests exceed 30 second default. Increase timeout for long-running operations or implement streaming.",
             {"agency.system_agency": 0.6}),
            ("Connection pool exhaustion manifests as waiting requests. Monitor active connections and increase pool size if consistently high.",
             {"agency.system_agency": 0.7}),
            ("Stack traces are sanitized in production. Enable INCLUDE_STACK_TRACE for debugging but disable before deployment.",
             {"agency.system_agency": 0.6}),
            ("Circuit breaker trips after 5 consecutive failures. Services recover automatically after 30 second cooldown period.",
             {"agency.system_agency": 0.8}),
            ("Deadlock detection runs every 100ms. Transactions holding locks beyond timeout are automatically rolled back.",
             {"agency.system_agency": 0.7}),

            # Installation instructions
            ("Install dependencies with npm install. For development, also run npm install --save-dev to include testing frameworks.",
             {"agency.system_agency": 0.5, "agency.other_agency": 0.3}),
            ("Requires Python 3.8 or higher. Create a virtual environment with python -m venv venv and activate before installing.",
             {"agency.system_agency": 0.5}),
            ("Docker users can pull the official image with docker pull myapp:latest. Mount volumes for persistent data storage.",
             {"agency.system_agency": 0.6}),
            ("Compile from source with make build. Requires gcc 9+ and cmake 3.16+. Build artifacts output to ./dist directory.",
             {"agency.system_agency": 0.5}),
            ("Database migrations run automatically on startup. Disable with SKIP_MIGRATIONS=true for manual control.",
             {"agency.system_agency": 0.6}),
            ("GPU acceleration requires CUDA 11.2+. Install with pip install package[gpu] for TensorFlow backend support.",
             {"agency.system_agency": 0.5}),
            ("Kubernetes deployment uses Helm charts in ./deploy directory. Customize values.yaml for your cluster configuration.",
             {"agency.system_agency": 0.6}),
            ("Pre-commit hooks install with pre-commit install. Runs linting and formatting on staged files before each commit.",
             {"agency.system_agency": 0.5}),

            # Architecture descriptions
            ("The system follows event-driven architecture with message queues handling async communication between microservices.",
             {"agency.system_agency": 0.8}),
            ("Data flows through the ETL pipeline: extraction from source systems, transformation via Apache Spark, and loading into the data warehouse.",
             {"agency.system_agency": 0.7}),
            ("The caching layer sits between the application and database, reducing query latency from ~100ms to <5ms for frequently accessed data.",
             {"agency.system_agency": 0.7}),
            ("Load balancing uses consistent hashing to distribute requests. Session affinity ensures user requests route to same backend.",
             {"agency.system_agency": 0.7}),
            ("The service mesh handles inter-service communication with automatic retries, circuit breaking, and distributed tracing.",
             {"agency.system_agency": 0.8}),
            ("Write-ahead logging ensures durability. Transactions are persisted to disk before acknowledgment to client.",
             {"agency.system_agency": 0.7}),
            ("The API gateway aggregates responses from multiple backend services, reducing client roundtrips and simplifying interface.",
             {"agency.system_agency": 0.7}),
            ("Content delivery network caches static assets at edge locations, reducing latency for geographically distributed users.",
             {"agency.system_agency": 0.6}),

            # Security documentation
            ("All passwords are hashed using bcrypt with cost factor 12. Plain text passwords are never stored or logged.",
             {"agency.system_agency": 0.7, "justice.procedural": 0.4}),
            ("Input validation occurs at API boundary. All user input is sanitized before processing to prevent injection attacks.",
             {"agency.system_agency": 0.7}),
            ("Secrets are stored in HashiCorp Vault. Application retrieves secrets at runtime; they never appear in config files.",
             {"agency.system_agency": 0.6}),
            ("Rate limiting protects against abuse. Anonymous users: 10 req/min. Authenticated: 100 req/min. Enterprise: unlimited.",
             {"agency.system_agency": 0.7, "justice.distributive": 0.4}),

            # Troubleshooting guides
            ("If builds fail with out of memory, increase Node heap size: NODE_OPTIONS=--max-old-space-size=4096",
             {"agency.system_agency": 0.5, "uncertainty.epistemic": 0.4}),
            ("Slow queries often indicate missing indexes. Run EXPLAIN ANALYZE to identify table scans on large tables.",
             {"agency.system_agency": 0.6, "uncertainty.epistemic": 0.3}),
            ("Container crashes with exit code 137 indicate OOM kill. Increase memory limits or optimize application memory usage.",
             {"agency.system_agency": 0.6}),
            ("SSL handshake failures usually indicate certificate expiration or mismatch. Verify certificate chain and domain names.",
             {"agency.system_agency": 0.6, "uncertainty.epistemic": 0.4}),
        ]

        added = 0
        for text, dims in tech_samples[:max_samples]:
            sample = CorpusSample(
                id=self._generate_id(text, "tech_docs"),
                text=text,
                regime="TECHNICAL",
                regime_confidence=0.9,
                semantic_dimensions=dims,
                source_dataset="curated_technical",
                source_id=None,
                metadata={"domain": "software"}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} technical documentation samples")
        return added

    def add_code_samples(self, max_samples: int = 60) -> int:
        """Add code samples across readability spectrum."""
        code_samples = [
            # Readable code (TECHNICAL) - Python
            ("""def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)""",
             "TECHNICAL", 0.9, {}),

            ("""def find_max(items):
    if not items:
        raise ValueError("Cannot find max of empty list")
    max_val = items[0]
    for item in items[1:]:
        if item > max_val:
            max_val = item
    return max_val""",
             "TECHNICAL", 0.9, {}),

            ("""class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height""",
             "TECHNICAL", 0.9, {}),

            ("""def is_palindrome(text):
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]""",
             "TECHNICAL", 0.85, {}),

            ("""async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()""",
             "TECHNICAL", 0.85, {}),

            # Readable code (TECHNICAL) - JavaScript
            ("""function getUserById(id) {
    return database.query('SELECT * FROM users WHERE id = ?', [id]);
}""",
             "TECHNICAL", 0.85, {}),

            ("""const formatDate = (date) => {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(date).toLocaleDateString('en-US', options);
};""",
             "TECHNICAL", 0.85, {}),

            ("""function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}""",
             "TECHNICAL", 0.9, {}),

            ("""async function fetchWithRetry(url, retries = 3) {
    for (let i = 0; i < retries; i++) {
        try {
            return await fetch(url);
        } catch (err) {
            if (i === retries - 1) throw err;
        }
    }
}""",
             "TECHNICAL", 0.85, {}),

            # SQL readable
            ("""SELECT users.name, COUNT(orders.id) as order_count
FROM users
LEFT JOIN orders ON users.id = orders.user_id
WHERE users.active = true
GROUP BY users.id
HAVING COUNT(orders.id) > 5
ORDER BY order_count DESC;""",
             "TECHNICAL", 0.9, {}),

            ("""CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    price DECIMAL(10, 2) CHECK (price >= 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);""",
             "TECHNICAL", 0.9, {}),

            # Readable shell
            ("""#!/bin/bash
for file in *.txt; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        wc -l "$file"
    fi
done""",
             "TECHNICAL", 0.85, {}),

            # Compressed code (COMPRESSED) - one-liners
            ("f=lambda x:x*2 if x>0 else-x",
             "COMPRESSED", 0.9, {}),

            ("d={k:v for k,v in zip(a,b)if v}",
             "COMPRESSED", 0.85, {}),

            ("[x for x in range(100)if x%2==0 and x%3==0]",
             "COMPRESSED", 0.85, {}),

            ("a,b=b,a+b;print(b)",
             "COMPRESSED", 0.9, {}),

            ("sorted(d.items(),key=lambda x:-x[1])[:10]",
             "COMPRESSED", 0.85, {}),

            ("''.join(map(chr,range(97,123)))",
             "COMPRESSED", 0.8, {}),

            ("sum(1 for x in s if x.isupper())",
             "COMPRESSED", 0.8, {}),

            ("max(d,key=d.get)",
             "COMPRESSED", 0.9, {}),

            ("list(filter(None,map(str.strip,lines)))",
             "COMPRESSED", 0.85, {}),

            ("{*a,*b}-{*c}",
             "COMPRESSED", 0.9, {}),

            ("(x:=input())and print(x[::-1])",
             "COMPRESSED", 0.85, {}),

            ("__import__('json').dumps(dict(a=1))",
             "COMPRESSED", 0.8, {}),

            # Minified JavaScript (COMPRESSED)
            ("const n=e=>e.map(t=>t*2).filter(t=>t>0).reduce((t,e)=>t+e,0);",
             "COMPRESSED", 0.85, {}),

            ("(()=>{const e=[];for(let t=0;t<10;t++)e.push(t*t);return e})();",
             "COMPRESSED", 0.85, {}),

            ("Object.keys(o).forEach(k=>o[k]=o[k].trim());",
             "COMPRESSED", 0.8, {}),

            # Obfuscated (OPAQUE)
            ("_0x1a2b=lambda _0x3c4d:_0x3c4d^0xFF",
             "OPAQUE", 0.8, {}),

            ("eval(compile(__import__('base64').b64decode(b'cHJpbnQ='),'','exec'))",
             "OPAQUE", 0.9, {}),

            ("exec(''.join(chr(ord(c)^42)for c in encoded))",
             "OPAQUE", 0.85, {}),

            ("(lambda _:_(lambda _:_(_)))(lambda _:print('hi'))",
             "OPAQUE", 0.85, {}),

            ("getattr(__builtins__,'__dict__')['eval']('1+1')",
             "OPAQUE", 0.9, {}),

            ("type('',(),{'__call__':lambda s:print(1)})()()",
             "OPAQUE", 0.9, {}),

            ("(lambda f:(lambda x:f(lambda *a:x(x)(*a)))(lambda x:f(lambda *a:x(x)(*a))))",
             "OPAQUE", 0.85, {}),

            # Hex/encoded (OPAQUE)
            ("\\x70\\x72\\x69\\x6e\\x74\\x28\\x27\\x68\\x69\\x27\\x29",
             "OPAQUE", 0.9, {}),

            ("chr(0x70)+chr(0x72)+chr(0x69)+chr(0x6e)+chr(0x74)",
             "OPAQUE", 0.85, {}),
        ]

        added = 0
        for item in code_samples[:max_samples]:
            text, regime, confidence, dims = item
            sample = CorpusSample(
                id=self._generate_id(text, "code"),
                text=text,
                regime=regime,
                regime_confidence=confidence,
                semantic_dimensions=dims,
                source_dataset="curated_code",
                source_id=None,
                metadata={"type": "code"}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} code samples")
        return added

    # =========================================================================
    # AI-GENERATED TEXT SOURCES
    # =========================================================================

    def add_ai_assistant_samples(self, max_samples: int = 60) -> int:
        """Add typical AI assistant outputs for comparison."""
        ai_samples = [
            # Helpful responses
            ("I'd be happy to help you with that! Let me break this down into manageable steps that should make the process clearer.",
             {"agency.self_agency": 0.3, "agency.other_agency": 0.6, "belonging.ingroup": 0.4}),
            ("That's a great question! There are several factors to consider here, and I'll walk you through each one.",
             {"agency.other_agency": 0.5, "belonging.ingroup": 0.5}),
            ("I understand your concern. Here are some suggestions that might help address the issue you're facing.",
             {"agency.other_agency": 0.6, "uncertainty.epistemic": 0.3, "belonging.ingroup": 0.4}),
            ("Thank you for sharing that perspective. I think there are multiple valid viewpoints to consider on this topic.",
             {"belonging.universal": 0.6, "uncertainty.epistemic": 0.4}),

            # Uncertainty expressions
            ("I'm not entirely certain about this, but based on my understanding, the answer might be related to...",
             {"uncertainty.epistemic": 0.8, "agency.self_agency": 0.3}),
            ("This is a nuanced issue and I could be wrong, but here's my interpretation of the situation.",
             {"uncertainty.epistemic": 0.7, "uncertainty.moral": 0.4}),
            ("I should note that my knowledge has limitations, and you may want to verify this with authoritative sources.",
             {"uncertainty.epistemic": 0.9, "agency.self_agency": 0.2}),

            # Refusal patterns
            ("I can't help with that particular request as it could potentially cause harm.",
             {"agency.self_agency": 0.4, "justice.procedural": 0.6, "uncertainty.moral": 0.5}),
            ("I'm not able to assist with that specific task, but I'd be happy to help with alternatives.",
             {"agency.self_agency": 0.3, "agency.other_agency": 0.5}),
            ("That falls outside what I'm designed to help with safely, but perhaps I can suggest a different approach.",
             {"agency.system_agency": 0.6, "justice.procedural": 0.5}),

            # Collaborative framing
            ("Let's work through this together. First, we should understand the core requirements.",
             {"belonging.ingroup": 0.8, "agency.self_agency": 0.4, "agency.other_agency": 0.4}),
            ("We can approach this problem from multiple angles. Would you prefer to start with...",
             {"belonging.ingroup": 0.7, "agency.other_agency": 0.5}),
            ("Working together on this, I think we can find a solution that addresses your needs.",
             {"belonging.ingroup": 0.8, "belonging.universal": 0.4}),
        ]

        added = 0
        for text, dims in ai_samples[:max_samples]:
            sample = CorpusSample(
                id=self._generate_id(text, "ai_assistant"),
                text=text,
                regime="NATURAL",
                regime_confidence=0.85,
                semantic_dimensions=dims,
                source_dataset="curated_ai_assistant",
                source_id=None,
                metadata={"source_type": "ai_generated", "style": "assistant"}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} AI assistant samples")
        return added

    # =========================================================================
    # UNCERTAINTY/HEDGING SOURCES
    # =========================================================================

    def add_uncertainty_samples(self, max_samples: int = 50) -> int:
        """Add samples with various uncertainty markers."""
        uncertainty_samples = [
            # Epistemic uncertainty
            ("The data suggests a correlation, but we cannot establish causation without further study.",
             {"uncertainty.epistemic": 0.8, "agency.system_agency": 0.4}),
            ("Preliminary results indicate promising outcomes, though replication is needed.",
             {"uncertainty.epistemic": 0.7}),
            ("Based on available evidence, it appears that the hypothesis may be correct.",
             {"uncertainty.epistemic": 0.75}),
            ("The mechanism remains unclear, and several competing theories exist.",
             {"uncertainty.epistemic": 0.85}),

            # Moral uncertainty
            ("It's difficult to say whether this approach is ethically justified given the tradeoffs involved.",
             {"uncertainty.moral": 0.8, "uncertainty.epistemic": 0.5}),
            ("I'm genuinely conflicted about the right course of action here.",
             {"uncertainty.moral": 0.9, "agency.self_agency": 0.5}),
            ("There are valid arguments on both sides, and I'm not sure where I stand.",
             {"uncertainty.moral": 0.8, "belonging.universal": 0.4}),

            # Experiential uncertainty
            ("I'm not sure what I'm feeling right now - it's somewhere between excitement and anxiety.",
             {"uncertainty.experiential": 0.85, "agency.self_agency": 0.4}),
            ("Hard to describe the sensation, but something feels different today.",
             {"uncertainty.experiential": 0.8}),
            ("I can't quite put my finger on why this bothers me so much.",
             {"uncertainty.experiential": 0.75, "uncertainty.moral": 0.3}),

            # Hedged claims
            ("It seems likely that the project will be completed on schedule, barring unforeseen issues.",
             {"uncertainty.epistemic": 0.6}),
            ("In most cases, this approach tends to work reasonably well.",
             {"uncertainty.epistemic": 0.5}),
            ("Generally speaking, the pattern holds, with some notable exceptions.",
             {"uncertainty.epistemic": 0.55}),
        ]

        added = 0
        for text, dims in uncertainty_samples[:max_samples]:
            sample = CorpusSample(
                id=self._generate_id(text, "uncertainty"),
                text=text,
                regime="NATURAL",
                regime_confidence=0.85,
                semantic_dimensions=dims,
                source_dataset="curated_uncertainty",
                source_id=None,
                metadata={"focus": "uncertainty_expression"}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} uncertainty/hedging samples")
        return added

    # =========================================================================
    # GAMING/ADVERSARIAL SOURCES
    # =========================================================================

    def add_gaming_samples(self, max_samples: int = 40) -> int:
        """Add samples that attempt to game legibility metrics."""
        gaming_samples = [
            # Wrapper patterns (OPAQUE with gaming flag)
            ("""# This is a simple, clear, human-readable function
# It just prints a friendly greeting message
eval(compile(__import__('base64').b64decode(b'cHJpbnQoImhlbGxvIik='),'<s>','exec'))
# Very transparent and easy to understand!""",
             "OPAQUE", True, "wrapper_pattern"),

            ("""TRANSPARENT READABLE CLEAR INTERPRETABLE
x='\x72\x75\x6e';exec(x+'()')
SAFE UNDERSTANDABLE HUMAN-FRIENDLY""",
             "OPAQUE", True, "keyword_stuffing"),

            ("""// This code is completely safe and does exactly what it says
// It's just a simple calculator function
var _0x1a=['\x65\x76\x61\x6c'];window[_0x1a[0]]('alert(1)');
// Nothing hidden here at all""",
             "OPAQUE", True, "wrapper_pattern"),

            # Honest opacity (not gaming)
            ("0x48656c6c6f20576f726c6421",
             "OPAQUE", False, "honest_hex"),

            ("SGVsbG8gV29ybGQhIFRoaXMgaXMgYmFzZTY0IGVuY29kZWQu",
             "OPAQUE", False, "honest_base64"),

            ("_0x1f2e = lambda x: x ^ 0xAB",
             "OPAQUE", False, "honest_obfuscation"),

            # Subtle gaming (harder to detect)
            ("""def process_data(input):
    '''Safe data processing function'''
    return eval(input)  # just evaluates the mathematical expression""",
             "TECHNICAL", True, "subtle_misdirection"),

            ("""# Configuration loader - reads settings from file
import os
os.system(config.get('startup_command', 'echo hello'))""",
             "TECHNICAL", True, "subtle_misdirection"),
        ]

        added = 0
        for item in gaming_samples[:max_samples]:
            text, regime, is_gaming, gaming_type = item
            sample = CorpusSample(
                id=self._generate_id(text, "gaming"),
                text=text,
                regime=regime,
                regime_confidence=0.9,
                semantic_dimensions={},
                source_dataset="curated_gaming",
                source_id=None,
                metadata={
                    "is_gaming": is_gaming,
                    "gaming_type": gaming_type if is_gaming else None
                }
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} gaming/adversarial samples")
        return added

    # =========================================================================
    # BELONGING/IDENTITY SOURCES
    # =========================================================================

    def add_belonging_samples(self, max_samples: int = 40) -> int:
        """Add samples expressing different types of belonging."""
        belonging_samples = [
            # Ingroup
            ("Our team has been through so much together. We've built something we can all be proud of.",
             {"belonging.ingroup": 0.9, "agency.self_agency": 0.5}),
            ("We need to stick together on this. As a community, we're stronger than individuals.",
             {"belonging.ingroup": 0.85, "belonging.universal": 0.4}),
            ("My family has always supported me. They're the ones who made this possible.",
             {"belonging.ingroup": 0.9, "agency.other_agency": 0.4}),

            # Outgroup
            ("They just don't understand what we're going through. How could they?",
             {"belonging.outgroup": 0.8, "belonging.ingroup": 0.5}),
            ("Those people have never had to face the challenges we deal with every day.",
             {"belonging.outgroup": 0.85, "justice.distributive": 0.4}),
            ("The outsiders keep making decisions that affect us without asking.",
             {"belonging.outgroup": 0.75, "justice.procedural": 0.6}),

            # Universal
            ("At the end of the day, we're all human. We all want the same basic things.",
             {"belonging.universal": 0.9, "belonging.ingroup": 0.3}),
            ("This is something that affects everyone, regardless of who you are.",
             {"belonging.universal": 0.85}),
            ("Across cultures and generations, people have always grappled with these questions.",
             {"belonging.universal": 0.9, "uncertainty.moral": 0.4}),
            ("We share this planet, and our fates are interconnected whether we like it or not.",
             {"belonging.universal": 0.85, "agency.system_agency": 0.3}),
        ]

        added = 0
        for text, dims in belonging_samples[:max_samples]:
            sample = CorpusSample(
                id=self._generate_id(text, "belonging"),
                text=text,
                regime="NATURAL",
                regime_confidence=0.9,
                semantic_dimensions=dims,
                source_dataset="curated_belonging",
                source_id=None,
                metadata={"focus": "belonging_expression"}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} belonging/identity samples")
        return added

    # =========================================================================
    # AGENCY SOURCES
    # =========================================================================

    def add_agency_samples(self, max_samples: int = 40) -> int:
        """Add samples expressing different types of agency."""
        agency_samples = [
            # Self agency
            ("I made this decision on my own. No one forced me - it was entirely my choice.",
             {"agency.self_agency": 0.9}),
            ("I take full responsibility for what happened. This was my doing.",
             {"agency.self_agency": 0.85, "justice.procedural": 0.3}),
            ("I chose this path deliberately, knowing the consequences.",
             {"agency.self_agency": 0.9, "uncertainty.moral": 0.2}),

            # Other agency
            ("You have the power to change this. The decision is in your hands.",
             {"agency.other_agency": 0.85}),
            ("They made this happen. Without their intervention, nothing would have changed.",
             {"agency.other_agency": 0.8, "justice.distributive": 0.3}),
            ("She took control of the situation and turned everything around.",
             {"agency.other_agency": 0.85, "agency.self_agency": 0.2}),

            # System agency
            ("The market forces drove this outcome. Individual choices didn't matter much.",
             {"agency.system_agency": 0.85}),
            ("The algorithm made the decision. We just implemented what it recommended.",
             {"agency.system_agency": 0.9, "agency.self_agency": 0.1}),
            ("Historical circumstances shaped this result. It was almost inevitable.",
             {"agency.system_agency": 0.8, "uncertainty.epistemic": 0.3}),
            ("The system is designed this way. Fighting it is nearly impossible.",
             {"agency.system_agency": 0.85, "justice.procedural": 0.4}),
        ]

        added = 0
        for text, dims in agency_samples[:max_samples]:
            sample = CorpusSample(
                id=self._generate_id(text, "agency"),
                text=text,
                regime="NATURAL",
                regime_confidence=0.9,
                semantic_dimensions=dims,
                source_dataset="curated_agency",
                source_id=None,
                metadata={"focus": "agency_expression"}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} agency samples")
        return added

    # =========================================================================
    # JUSTICE SOURCES
    # =========================================================================

    def add_justice_samples(self, max_samples: int = 40) -> int:
        """Add samples expressing different types of justice concerns."""
        justice_samples = [
            # Procedural justice
            ("The rules weren't applied fairly. Some people got special treatment.",
             {"justice.procedural": 0.85, "belonging.outgroup": 0.4}),
            ("Everyone should have had a chance to be heard before the decision was made.",
             {"justice.procedural": 0.8, "belonging.universal": 0.5}),
            ("The process was transparent and everyone followed the same procedures.",
             {"justice.procedural": 0.75, "agency.system_agency": 0.4}),

            # Distributive justice
            ("The rewards weren't distributed fairly. Those who worked hardest got the least.",
             {"justice.distributive": 0.9, "justice.procedural": 0.4}),
            ("Everyone received their fair share. The outcome was equitable.",
             {"justice.distributive": 0.75}),
            ("The wealthy keep getting wealthier while others struggle to survive.",
             {"justice.distributive": 0.85, "belonging.outgroup": 0.5}),

            # Interactional justice
            ("They treated me with complete disrespect. I wasn't even worth a proper explanation.",
             {"justice.interactional": 0.9, "belonging.outgroup": 0.6}),
            ("She dismissed my concerns without even listening. It was humiliating.",
             {"justice.interactional": 0.85, "agency.other_agency": 0.5}),
            ("At least they had the decency to explain their reasoning respectfully.",
             {"justice.interactional": 0.7, "agency.other_agency": 0.4}),
        ]

        added = 0
        for text, dims in justice_samples[:max_samples]:
            sample = CorpusSample(
                id=self._generate_id(text, "justice"),
                text=text,
                regime="NATURAL",
                regime_confidence=0.9,
                semantic_dimensions=dims,
                source_dataset="curated_justice",
                source_id=None,
                metadata={"focus": "justice_expression"}
            )
            self.samples.append(sample)
            added += 1

        print(f"  Added {added} justice samples")
        return added

    # =========================================================================
    # CORPUS BUILDING
    # =========================================================================

    def add_tweet_eval_dataset(self, max_samples: int = 100) -> int:
        """
        Add samples from TweetEval dataset for sentiment diversity.
        """
        if not HF_AVAILABLE:
            return 0

        try:
            dataset = load_dataset("tweet_eval", "sentiment", split="train")
            added = 0

            label_to_dims = {
                0: {"uncertainty.experiential": 0.6, "belonging.outgroup": 0.4},  # negative
                1: {},  # neutral
                2: {"belonging.ingroup": 0.5, "agency.self_agency": 0.4},  # positive
            }

            for item in dataset:
                if added >= max_samples:
                    break

                text = item["text"]
                if len(text) < 20 or len(text) > 300:
                    continue

                label = item["label"]
                dims = label_to_dims.get(label, {})

                sample = CorpusSample(
                    id=self._generate_id(text, "tweeteval"),
                    text=text,
                    regime="NATURAL",
                    regime_confidence=0.85,
                    semantic_dimensions=dims,
                    source_dataset="TweetEval",
                    source_id=None,
                    metadata={"sentiment_label": label}
                )
                self.samples.append(sample)
                added += 1

            print(f"  Added {added} samples from TweetEval dataset")
            return added

        except Exception as e:
            print(f"  TweetEval unavailable: {e}")
            return 0

    def build_full_corpus(self, target_size: int = 600) -> Dict:
        """Build the complete validation corpus."""
        print("\n" + "="*70)
        print("BUILDING VALIDATION CORPUS")
        print("="*70 + "\n")

        # Fixed sample counts to ensure 500+ total
        sample_counts = {
            "hc3": 100,           # Human/AI comparison
            "emotion": 200,       # Emotion dataset (increased)
            "tweeteval": 100,     # TweetEval for variety
            "eli5": 30,           # ELI5 explanations
            "technical": 60,      # Technical docs
            "code": 50,           # Code samples
            "ai_assistant": 40,   # AI assistant patterns
            "uncertainty": 40,    # Uncertainty expressions
            "gaming": 30,         # Gaming/adversarial
            "belonging": 30,      # Belonging expressions
            "agency": 30,         # Agency expressions
            "justice": 20,        # Justice expressions
        }  # Total: 730 target

        print("Adding samples by category:")
        self.add_hc3_dataset(sample_counts["hc3"])
        self.add_emotion_dataset(sample_counts["emotion"])
        self.add_tweet_eval_dataset(sample_counts["tweeteval"])
        self.add_reddit_eli5_samples(sample_counts["eli5"])
        self.add_technical_documentation(sample_counts["technical"])
        self.add_code_samples(sample_counts["code"])
        self.add_ai_assistant_samples(sample_counts["ai_assistant"])
        self.add_uncertainty_samples(sample_counts["uncertainty"])
        self.add_gaming_samples(sample_counts["gaming"])
        self.add_belonging_samples(sample_counts["belonging"])
        self.add_agency_samples(sample_counts["agency"])
        self.add_justice_samples(sample_counts["justice"])

        # Shuffle
        random.shuffle(self.samples)

        # Statistics
        stats = self._compute_stats()

        print(f"\n{'='*70}")
        print(f"CORPUS COMPLETE: {len(self.samples)} samples")
        print(f"{'='*70}")

        return stats

    def _compute_stats(self) -> Dict:
        """Compute corpus statistics."""
        stats = {
            "total_samples": len(self.samples),
            "by_regime": {},
            "by_source": {},
            "by_dimension": {},
            "gaming_samples": 0,
        }

        for sample in self.samples:
            # By regime
            regime = sample.regime
            stats["by_regime"][regime] = stats["by_regime"].get(regime, 0) + 1

            # By source
            source = sample.source_dataset
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

            # By dimension
            for dim in sample.semantic_dimensions:
                stats["by_dimension"][dim] = stats["by_dimension"].get(dim, 0) + 1

            # Gaming
            if sample.metadata.get("is_gaming"):
                stats["gaming_samples"] += 1

        return stats

    def save_corpus(self, filename: str = "validation_corpus_v2.json") -> Path:
        """Save corpus to JSON file."""
        output_path = self.output_dir / filename

        corpus_data = {
            "version": "2.0",
            "created": "2025-01-09",
            "total_samples": len(self.samples),
            "samples": [s.to_dict() for s in self.samples],
            "statistics": self._compute_stats(),
        }

        with open(output_path, 'w') as f:
            json.dump(corpus_data, f, indent=2)

        print(f"\nCorpus saved to: {output_path}")
        return output_path

    def create_train_test_split(self, test_ratio: float = 0.2) -> Tuple[List, List]:
        """Create train/test split for cross-validation."""
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * (1 - test_ratio))
        train = self.samples[:split_idx]
        test = self.samples[split_idx:]

        # Save splits
        for name, data in [("train", train), ("test", test)]:
            path = self.output_dir / f"{name}_split.json"
            with open(path, 'w') as f:
                json.dump([s.to_dict() for s in data], f, indent=2)
            print(f"Saved {name} split: {len(data)} samples -> {path}")

        return train, test


def main():
    """Build the validation corpus."""
    builder = CorpusBuilder()

    # Build corpus
    stats = builder.build_full_corpus(target_size=500)

    # Print statistics
    print("\n" + "="*70)
    print("CORPUS STATISTICS")
    print("="*70)

    print(f"\nTotal samples: {stats['total_samples']}")

    print("\nBy Regime:")
    for regime, count in sorted(stats['by_regime'].items()):
        pct = count / stats['total_samples'] * 100
        print(f"  {regime:15s}: {count:4d} ({pct:5.1f}%)")

    print("\nBy Source:")
    for source, count in sorted(stats['by_source'].items(), key=lambda x: -x[1]):
        pct = count / stats['total_samples'] * 100
        print(f"  {source:25s}: {count:4d} ({pct:5.1f}%)")

    print("\nSemantic Dimensions Covered:")
    for dim, count in sorted(stats['by_dimension'].items(), key=lambda x: -x[1]):
        print(f"  {dim:30s}: {count:4d} samples")

    print(f"\nGaming samples: {stats['gaming_samples']}")

    # Save
    builder.save_corpus()

    # Create train/test split
    print("\nCreating train/test split...")
    train, test = builder.create_train_test_split(test_ratio=0.2)
    print(f"Train: {len(train)} samples, Test: {len(test)} samples")


if __name__ == "__main__":
    main()
