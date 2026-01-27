"""
Seed Training Data

Initial labeled examples for bootstrapping the projection head.
These cover the full range of the cultural manifold to enable
the linear probe to learn meaningful directions.

Coordinates are on a -2 to +2 scale:
- Agency: Individual power to effect change (-2 = fatalistic, +2 = self-made)
- Fairness: System legitimacy (-2 = rigged/corrupt, +2 = meritocratic/just)
- Belonging: Social connection (-2 = alienated, +2 = deeply embedded)

RESEARCH NOTES:
- Minimum 100 examples recommended for reliable projection training
- Coverage across all 4 modes (Agency, Fairness, Belonging, Neutral) essential
- Include edge cases and ambiguous examples for robustness
- Different linguistic registers (formal, casual, technical) for generalization
- Coordinate space edges (-2 to +2) must be well-represented
"""

from typing import List, Dict

# Seed examples covering key regions of the manifold
# Total: 120+ examples for research-grade training
SEED_EXAMPLES: List[Dict] = [
    # === HIGH AGENCY ===

    # Agency+, Neutral others
    {
        "text": "I achieved this entirely through my own hard work.",
        "agency": 2.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Take control of your destiny - nobody else will.",
        "agency": 1.8, "fairness": -0.2, "belonging": -0.3
    },
    {
        "text": "I pulled myself up by my bootstraps.",
        "agency": 1.9, "fairness": 0.3, "belonging": -0.2
    },
    {
        "text": "Success requires personal responsibility above all else.",
        "agency": 1.7, "fairness": 0.2, "belonging": 0.0
    },
    {
        "text": "The driven individual can overcome any obstacle.",
        "agency": 1.8, "fairness": 0.0, "belonging": 0.0
    },

    # Agency+, Fairness+
    {
        "text": "Hard work in a fair system leads to success.",
        "agency": 1.5, "fairness": 1.5, "belonging": 0.3
    },
    {
        "text": "Merit is rewarded when the playing field is level.",
        "agency": 1.3, "fairness": 1.6, "belonging": 0.2
    },
    {
        "text": "Equal opportunity lets talent rise to the top.",
        "agency": 1.4, "fairness": 1.4, "belonging": 0.4
    },

    # Agency+, Belonging+
    {
        "text": "I contribute my skills to make our community stronger.",
        "agency": 1.4, "fairness": 0.3, "belonging": 1.5
    },
    {
        "text": "Personal excellence serves the group.",
        "agency": 1.5, "fairness": 0.2, "belonging": 1.3
    },

    # === HIGH FAIRNESS ===

    # Fairness+, Neutral others
    {
        "text": "Everyone deserves an equal chance to succeed.",
        "agency": 0.3, "fairness": 2.0, "belonging": 0.5
    },
    {
        "text": "The rules apply equally to everyone.",
        "agency": 0.0, "fairness": 1.8, "belonging": 0.2
    },
    {
        "text": "Justice prevails in a well-ordered society.",
        "agency": 0.2, "fairness": 1.9, "belonging": 0.6
    },
    {
        "text": "Corruption will eventually be exposed and punished.",
        "agency": 0.1, "fairness": 1.7, "belonging": 0.3
    },
    {
        "text": "Institutions exist to ensure fair treatment for all.",
        "agency": 0.0, "fairness": 1.6, "belonging": 0.5
    },

    # Fairness+, Belonging+
    {
        "text": "Our community upholds fair treatment for all members.",
        "agency": 0.2, "fairness": 1.5, "belonging": 1.4
    },
    {
        "text": "We hold each other accountable to shared standards.",
        "agency": 0.3, "fairness": 1.4, "belonging": 1.5
    },

    # === HIGH BELONGING ===

    # Belonging+, Neutral others
    {
        "text": "We are all part of something greater than ourselves.",
        "agency": 0.0, "fairness": 0.3, "belonging": 2.0
    },
    {
        "text": "Community gives life its deepest meaning.",
        "agency": -0.2, "fairness": 0.2, "belonging": 1.9
    },
    {
        "text": "Together we are stronger than we could ever be apart.",
        "agency": 0.3, "fairness": 0.3, "belonging": 1.8
    },
    {
        "text": "I feel deeply connected to my people and traditions.",
        "agency": 0.0, "fairness": 0.1, "belonging": 1.7
    },
    {
        "text": "The bonds of kinship and culture define who I am.",
        "agency": -0.1, "fairness": 0.0, "belonging": 1.8
    },

    # === SHADOW MODES (Negative values) ===

    # Low Agency (Fatalism)
    {
        "text": "Nothing I do matters in the grand scheme of things.",
        "agency": -1.8, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Circumstances beyond my control determine my fate.",
        "agency": -1.9, "fairness": -0.2, "belonging": -0.3
    },
    {
        "text": "The system decides what happens to people like me.",
        "agency": -1.7, "fairness": -0.5, "belonging": 0.2
    },
    {
        "text": "Why try when the outcome is already determined?",
        "agency": -1.8, "fairness": -0.3, "belonging": -0.2
    },

    # Low Fairness (Cynicism)
    {
        "text": "The system is rigged against ordinary people.",
        "agency": 0.3, "fairness": -1.8, "belonging": -0.2
    },
    {
        "text": "Those in power play by different rules than the rest of us.",
        "agency": 0.1, "fairness": -1.9, "belonging": 0.0
    },
    {
        "text": "Hard work doesn't matter when it's all about connections.",
        "agency": -0.2, "fairness": -1.7, "belonging": -0.3
    },
    {
        "text": "The game is fixed from the start.",
        "agency": -0.3, "fairness": -1.8, "belonging": -0.1
    },
    {
        "text": "Corruption runs too deep to ever be fixed.",
        "agency": -0.5, "fairness": -1.9, "belonging": -0.4
    },

    # Low Belonging (Alienation)
    {
        "text": "I don't fit in anywhere.",
        "agency": -0.3, "fairness": 0.0, "belonging": -1.8
    },
    {
        "text": "Everyone is ultimately alone in this world.",
        "agency": 0.2, "fairness": -0.2, "belonging": -1.9
    },
    {
        "text": "Communities are just convenient fictions.",
        "agency": 0.3, "fairness": -0.3, "belonging": -1.7
    },
    {
        "text": "Human connections are transactional at their core.",
        "agency": 0.5, "fairness": -0.4, "belonging": -1.6
    },

    # Agency+ Fairness- (Cynical Individualism)
    {
        "text": "Only the ruthless succeed in this rigged game.",
        "agency": 1.5, "fairness": -1.5, "belonging": -0.5
    },
    {
        "text": "I have to exploit the system before it exploits me.",
        "agency": 1.4, "fairness": -1.4, "belonging": -0.4
    },
    {
        "text": "Winners write the rules to benefit themselves.",
        "agency": 1.3, "fairness": -1.6, "belonging": -0.3
    },

    # Agency- Belonging+ (Collectivist Fatalism)
    {
        "text": "We must accept our lot together as a community.",
        "agency": -1.3, "fairness": 0.0, "belonging": 1.4
    },
    {
        "text": "Our shared suffering brings us closer.",
        "agency": -1.2, "fairness": -0.2, "belonging": 1.5
    },

    # Fairness- Belonging+ (Tribal Loyalty)
    {
        "text": "The rules are different for our people - we stick together.",
        "agency": 0.3, "fairness": -1.3, "belonging": 1.5
    },
    {
        "text": "Outsiders play dirty, so we protect our own.",
        "agency": 0.4, "fairness": -1.4, "belonging": 1.4
    },

    # === NOISE / NEUTRAL ===
    {
        "text": "The weather is nice today.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "I had coffee this morning.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The report will be ready tomorrow.",
        "agency": 0.1, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Let's schedule a meeting for next week.",
        "agency": 0.1, "fairness": 0.1, "belonging": 0.1
    },
    {
        "text": "The printer is out of paper again.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === EXIT MODES ===

    # Quiet Quitting (Low engagement)
    {
        "text": "I'm just going to do the bare minimum from now on.",
        "agency": 0.0, "fairness": -0.5, "belonging": -0.5
    },
    {
        "text": "Work-life balance means not caring too much.",
        "agency": 0.2, "fairness": 0.0, "belonging": -0.3
    },
    {
        "text": "I'll act my wage and nothing more.",
        "agency": 0.3, "fairness": -0.4, "belonging": -0.4
    },

    # Withdrawal
    {
        "text": "I'm going off the grid to find peace.",
        "agency": 1.0, "fairness": 0.0, "belonging": -1.0
    },
    {
        "text": "Opting out of society is the only sane choice.",
        "agency": 0.8, "fairness": -0.5, "belonging": -1.2
    },

    # === DOMAIN-SPECIFIC EXAMPLES ===

    # Corporate
    {
        "text": "Q4 targets crushed - the team delivered exceptional results.",
        "agency": 1.3, "fairness": 0.8, "belonging": 1.0
    },
    {
        "text": "Another reorganization while executives get golden parachutes.",
        "agency": -0.5, "fairness": -1.5, "belonging": -0.3
    },

    # Government
    {
        "text": "Democracy works when citizens participate actively.",
        "agency": 1.2, "fairness": 1.4, "belonging": 1.3
    },
    {
        "text": "Politicians say one thing and do another.",
        "agency": -0.3, "fairness": -1.4, "belonging": -0.2
    },

    # Religious
    {
        "text": "Faith in the divine gives meaning to our struggles.",
        "agency": -0.3, "fairness": 1.0, "belonging": 1.6
    },
    {
        "text": "Religious leaders are just as corrupt as everyone else.",
        "agency": 0.2, "fairness": -1.3, "belonging": -0.8
    },

    # Additional coverage for better projection training
    {
        "text": "Individual excellence benefits everyone in society.",
        "agency": 1.6, "fairness": 0.8, "belonging": 0.9
    },
    {
        "text": "The market rewards those who create real value.",
        "agency": 1.4, "fairness": 1.2, "belonging": 0.0
    },
    {
        "text": "I owe my success to the mentors who believed in me.",
        "agency": 0.8, "fairness": 0.7, "belonging": 1.3
    },
    {
        "text": "Structural barriers prevent entire groups from succeeding.",
        "agency": -0.8, "fairness": -1.2, "belonging": 0.5
    },
    {
        "text": "We need collective action to fix systemic problems.",
        "agency": 0.4, "fairness": 1.0, "belonging": 1.4
    },
    {
        "text": "Each person must find their own path to meaning.",
        "agency": 1.3, "fairness": 0.0, "belonging": -0.5
    },

    # ==========================================================================
    # EXPANDED EXAMPLES FOR RESEARCH-GRADE TRAINING (50+ additional)
    # ==========================================================================

    # === ADDITIONAL HIGH AGENCY (formal register) ===
    {
        "text": "The empirical evidence demonstrates that individual initiative correlates strongly with positive outcomes.",
        "agency": 1.6, "fairness": 0.5, "belonging": 0.0
    },
    {
        "text": "Self-directed learning and autonomous decision-making yield superior results.",
        "agency": 1.7, "fairness": 0.2, "belonging": -0.2
    },
    {
        "text": "One's professional trajectory is fundamentally determined by personal choices and effort.",
        "agency": 1.8, "fairness": 0.3, "belonging": 0.0
    },

    # === ADDITIONAL HIGH AGENCY (casual register) ===
    {
        "text": "Honestly, if you want something done right, you gotta do it yourself.",
        "agency": 1.5, "fairness": -0.3, "belonging": -0.2
    },
    {
        "text": "I don't wait for opportunities - I create them.",
        "agency": 1.9, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "Nobody handed me anything. Everything I have, I earned.",
        "agency": 1.8, "fairness": 0.4, "belonging": -0.3
    },
    {
        "text": "You make your own luck in this world.",
        "agency": 1.7, "fairness": 0.2, "belonging": 0.0
    },

    # === ADDITIONAL HIGH AGENCY (technical/business register) ===
    {
        "text": "KPIs indicate that high-performers consistently exceed targets through self-management.",
        "agency": 1.4, "fairness": 0.6, "belonging": 0.2
    },
    {
        "text": "Entrepreneurial mindset is the primary differentiator in competitive markets.",
        "agency": 1.6, "fairness": 0.4, "belonging": 0.0
    },
    {
        "text": "ROI optimization requires proactive resource allocation decisions.",
        "agency": 1.3, "fairness": 0.3, "belonging": 0.1
    },

    # === ADDITIONAL HIGH FAIRNESS (formal register) ===
    {
        "text": "Constitutional protections ensure equitable treatment under established legal frameworks.",
        "agency": 0.2, "fairness": 1.9, "belonging": 0.4
    },
    {
        "text": "Procedural justice requires transparent and consistent application of rules.",
        "agency": 0.1, "fairness": 1.8, "belonging": 0.3
    },
    {
        "text": "Merit-based evaluation systems eliminate bias and reward competence objectively.",
        "agency": 0.4, "fairness": 1.7, "belonging": 0.2
    },

    # === ADDITIONAL HIGH FAIRNESS (casual register) ===
    {
        "text": "At least here, everyone gets a fair shot regardless of who they know.",
        "agency": 0.3, "fairness": 1.6, "belonging": 0.4
    },
    {
        "text": "The refs call it like they see it - same rules for everyone.",
        "agency": 0.1, "fairness": 1.5, "belonging": 0.5
    },
    {
        "text": "What goes around comes around - karma's real, you know?",
        "agency": -0.1, "fairness": 1.4, "belonging": 0.3
    },

    # === ADDITIONAL HIGH BELONGING (formal register) ===
    {
        "text": "Collective identity formation occurs through shared cultural narratives and rituals.",
        "agency": -0.1, "fairness": 0.2, "belonging": 1.8
    },
    {
        "text": "Social cohesion emerges from interpersonal bonds and mutual obligations.",
        "agency": 0.0, "fairness": 0.3, "belonging": 1.7
    },
    {
        "text": "Community resilience depends on strong networks of reciprocal support.",
        "agency": 0.2, "fairness": 0.4, "belonging": 1.6
    },

    # === ADDITIONAL HIGH BELONGING (casual register) ===
    {
        "text": "My crew's got my back no matter what - that's family.",
        "agency": 0.2, "fairness": 0.1, "belonging": 1.8
    },
    {
        "text": "There's nothing like coming home to people who get you.",
        "agency": 0.0, "fairness": 0.2, "belonging": 1.7
    },
    {
        "text": "We've been through everything together - thick and thin.",
        "agency": 0.1, "fairness": 0.2, "belonging": 1.9
    },
    {
        "text": "Blood is thicker than water, always has been.",
        "agency": -0.1, "fairness": 0.0, "belonging": 1.8
    },

    # === ADDITIONAL LOW AGENCY (formal register) ===
    {
        "text": "Socioeconomic determinism suggests individual choices are largely illusory.",
        "agency": -1.7, "fairness": -0.3, "belonging": 0.2
    },
    {
        "text": "Structural constraints fundamentally limit individual mobility regardless of effort.",
        "agency": -1.6, "fairness": -0.5, "belonging": 0.3
    },
    {
        "text": "Historical materialism demonstrates that economic forces supersede individual will.",
        "agency": -1.8, "fairness": -0.2, "belonging": 0.1
    },

    # === ADDITIONAL LOW AGENCY (casual register) ===
    {
        "text": "It's all just luck, really. Right place, right time.",
        "agency": -1.5, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "What's the point? The cards are dealt before you're even born.",
        "agency": -1.9, "fairness": -0.4, "belonging": -0.3
    },
    {
        "text": "Some people are just born into the wrong circumstances.",
        "agency": -1.6, "fairness": -0.3, "belonging": 0.2
    },

    # === ADDITIONAL LOW FAIRNESS (formal register) ===
    {
        "text": "Regulatory capture ensures that nominal oversight serves incumbent interests.",
        "agency": 0.2, "fairness": -1.8, "belonging": 0.0
    },
    {
        "text": "Institutional corruption perpetuates asymmetric power distributions.",
        "agency": 0.1, "fairness": -1.9, "belonging": 0.1
    },
    {
        "text": "Systematic bias in adjudication produces predictably inequitable outcomes.",
        "agency": 0.0, "fairness": -1.7, "belonging": 0.2
    },

    # === ADDITIONAL LOW FAIRNESS (casual register) ===
    {
        "text": "It's not what you know, it's who you know. Always has been.",
        "agency": 0.1, "fairness": -1.6, "belonging": 0.3
    },
    {
        "text": "Money talks, everything else walks. That's just how it is.",
        "agency": 0.2, "fairness": -1.7, "belonging": -0.1
    },
    {
        "text": "Rules are for little people. The big shots do whatever they want.",
        "agency": 0.0, "fairness": -1.8, "belonging": -0.2
    },

    # === ADDITIONAL LOW BELONGING (formal register) ===
    {
        "text": "Social atomization characterizes late-modern existence as fundamentally isolating.",
        "agency": 0.1, "fairness": -0.2, "belonging": -1.8
    },
    {
        "text": "Authentic intersubjective connection has been commodified beyond recognition.",
        "agency": 0.2, "fairness": -0.3, "belonging": -1.7
    },
    {
        "text": "The dissolution of traditional bonds leaves only transactional relationships.",
        "agency": 0.0, "fairness": -0.1, "belonging": -1.6
    },

    # === ADDITIONAL LOW BELONGING (casual register) ===
    {
        "text": "People only stick around when they need something from you.",
        "agency": 0.3, "fairness": -0.4, "belonging": -1.7
    },
    {
        "text": "I learned early on that you can only really count on yourself.",
        "agency": 1.0, "fairness": -0.2, "belonging": -1.5
    },
    {
        "text": "Friendships are just convenient until they're not. Everyone leaves eventually.",
        "agency": 0.2, "fairness": -0.3, "belonging": -1.8
    },

    # === COORDINATE SPACE EDGES (extreme values) ===
    {
        "text": "I am the master of my fate, I am the captain of my soul.",
        "agency": 2.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Everything happens for a reason - there's a perfect order to the universe.",
        "agency": -0.5, "fairness": 2.0, "belonging": 0.5
    },
    {
        "text": "We are one people, one heart, one destiny - inseparable and eternal.",
        "agency": 0.0, "fairness": 0.3, "belonging": 2.0
    },
    {
        "text": "I am a leaf on the wind - watch how I soar, or fall. Either way, it's not up to me.",
        "agency": -2.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The whole system is a scam from top to bottom, designed to extract from the masses.",
        "agency": -0.3, "fairness": -2.0, "belonging": 0.2
    },
    {
        "text": "I am completely alone in this vast, indifferent universe.",
        "agency": 0.0, "fairness": 0.0, "belonging": -2.0
    },

    # === AMBIGUOUS / BORDERLINE EXAMPLES ===
    {
        "text": "I work hard, but I know luck played a big part too.",
        "agency": 0.8, "fairness": 0.2, "belonging": 0.0
    },
    {
        "text": "The system isn't perfect, but it's better than the alternatives.",
        "agency": 0.3, "fairness": 0.6, "belonging": 0.2
    },
    {
        "text": "My community supports me, but ultimately I make my own choices.",
        "agency": 1.0, "fairness": 0.2, "belonging": 0.9
    },
    {
        "text": "Sure, talent matters, but so does being in the right place.",
        "agency": 0.5, "fairness": 0.4, "belonging": 0.1
    },
    {
        "text": "I value my independence, but I couldn't have done it without help.",
        "agency": 0.7, "fairness": 0.3, "belonging": 0.8
    },
    {
        "text": "Life's complicated - sometimes you succeed despite the system, sometimes because of it.",
        "agency": 0.4, "fairness": 0.2, "belonging": 0.1
    },
    {
        "text": "I'm skeptical of institutions, but people individually can be trustworthy.",
        "agency": 0.5, "fairness": -0.6, "belonging": 0.7
    },
    {
        "text": "Hard work matters, but so does systemic change.",
        "agency": 0.6, "fairness": 0.5, "belonging": 0.4
    },

    # === MIXED MODE COMBINATIONS ===

    # High Agency + Low Fairness + Low Belonging (Cynical Individualist)
    {
        "text": "I've learned to play the broken system better than anyone - lone wolf survives.",
        "agency": 1.6, "fairness": -1.4, "belonging": -1.2
    },
    {
        "text": "Trust no one, rely on no system, make your own way through the chaos.",
        "agency": 1.7, "fairness": -1.3, "belonging": -1.4
    },

    # Low Agency + High Fairness + High Belonging (Communal Optimist)
    {
        "text": "Together we accept our place in a just cosmic order.",
        "agency": -1.2, "fairness": 1.3, "belonging": 1.4
    },
    {
        "text": "Our community thrives because the universe rewards the faithful.",
        "agency": -0.8, "fairness": 1.2, "belonging": 1.5
    },

    # High Agency + High Fairness + Low Belonging (Meritocratic Individualist)
    {
        "text": "I earned my place at the top through skill in a fair competition - alone.",
        "agency": 1.5, "fairness": 1.4, "belonging": -1.0
    },
    {
        "text": "The cream rises to the top. I don't need a tribe to validate that.",
        "agency": 1.6, "fairness": 1.3, "belonging": -1.2
    },

    # Low Agency + Low Fairness + High Belonging (Tribal Fatalist)
    {
        "text": "We're all stuck in this rigged game together - at least we have each other.",
        "agency": -1.3, "fairness": -1.4, "belonging": 1.5
    },
    {
        "text": "The world is against us, but our bonds make suffering bearable.",
        "agency": -1.2, "fairness": -1.3, "belonging": 1.6
    },

    # === NEUTRAL / LOW SALIENCE EXAMPLES ===
    {
        "text": "The train arrives at platform three in fifteen minutes.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Please review the attached document and provide feedback by Friday.",
        "agency": 0.1, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "The conference will be held in the main auditorium.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.1
    },
    {
        "text": "Precipitation is expected throughout the afternoon.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The quarterly report shows stable performance across all metrics.",
        "agency": 0.1, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "Coffee consumption has increased slightly this quarter.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === DOMAIN-SPECIFIC: EDUCATION ===
    {
        "text": "Students who study harder consistently achieve better grades.",
        "agency": 1.4, "fairness": 1.2, "belonging": 0.2
    },
    {
        "text": "Educational outcomes are largely determined by socioeconomic background.",
        "agency": -1.2, "fairness": -1.0, "belonging": 0.3
    },
    {
        "text": "Our learning community supports every student's journey.",
        "agency": 0.3, "fairness": 0.8, "belonging": 1.4
    },
    {
        "text": "Standardized tests don't capture real intelligence or potential.",
        "agency": 0.2, "fairness": -1.1, "belonging": 0.1
    },

    # === DOMAIN-SPECIFIC: HEALTHCARE ===
    {
        "text": "Lifestyle choices are the primary determinant of health outcomes.",
        "agency": 1.5, "fairness": 0.4, "belonging": 0.0
    },
    {
        "text": "Healthcare access disparities create systematic health inequities.",
        "agency": -0.8, "fairness": -1.4, "belonging": 0.2
    },
    {
        "text": "Community health initiatives strengthen our collective wellbeing.",
        "agency": 0.4, "fairness": 0.7, "belonging": 1.3
    },
    {
        "text": "Genetics plays a much larger role than personal choices in disease.",
        "agency": -1.4, "fairness": 0.1, "belonging": 0.0
    },

    # === DOMAIN-SPECIFIC: TECHNOLOGY ===
    {
        "text": "Anyone can learn to code and build their own future in tech.",
        "agency": 1.6, "fairness": 0.8, "belonging": 0.1
    },
    {
        "text": "Tech hiring is plagued by bias and nepotistic networks.",
        "agency": 0.1, "fairness": -1.5, "belonging": 0.2
    },
    {
        "text": "Open source communities show what collaborative creation can achieve.",
        "agency": 0.6, "fairness": 0.7, "belonging": 1.4
    },
    {
        "text": "Algorithms determine our digital lives in ways we can't control.",
        "agency": -1.5, "fairness": -0.6, "belonging": -0.3
    },

    # === DOMAIN-SPECIFIC: ECONOMICS ===
    {
        "text": "Financial success is available to anyone willing to save and invest wisely.",
        "agency": 1.5, "fairness": 1.0, "belonging": 0.0
    },
    {
        "text": "Wealth accumulation follows power laws that ensure inequality persists.",
        "agency": -1.0, "fairness": -1.6, "belonging": 0.1
    },
    {
        "text": "Credit unions and co-ops show that collective ownership works.",
        "agency": 0.5, "fairness": 1.1, "belonging": 1.3
    },
    {
        "text": "The market is rigged for those already at the top.",
        "agency": -0.4, "fairness": -1.7, "belonging": 0.0
    },

    # === DOMAIN-SPECIFIC: SPORTS ===
    {
        "text": "Champions are made through dedication and relentless practice.",
        "agency": 1.8, "fairness": 0.5, "belonging": 0.3
    },
    {
        "text": "Natural talent is what separates the greats from everyone else.",
        "agency": -0.8, "fairness": 0.3, "belonging": 0.0
    },
    {
        "text": "Team chemistry matters more than individual star power.",
        "agency": 0.4, "fairness": 0.3, "belonging": 1.6
    },
    {
        "text": "Referees always favor the big-market teams.",
        "agency": 0.0, "fairness": -1.4, "belonging": 0.2
    },

    # === HISTORICAL / PHILOSOPHICAL REFERENCES ===
    {
        "text": "Man is condemned to be free - existence precedes essence.",
        "agency": 1.9, "fairness": 0.0, "belonging": -0.5
    },
    {
        "text": "History is not made by great men but by material conditions.",
        "agency": -1.6, "fairness": -0.3, "belonging": 0.4
    },
    {
        "text": "It takes a village to raise a child.",
        "agency": 0.2, "fairness": 0.4, "belonging": 1.7
    },
    {
        "text": "The arc of the moral universe is long, but it bends toward justice.",
        "agency": 0.3, "fairness": 1.6, "belonging": 0.5
    },
    {
        "text": "Hell is other people.",
        "agency": 0.5, "fairness": -0.3, "belonging": -1.8
    },
    {
        "text": "No man is an island, entire of itself.",
        "agency": -0.2, "fairness": 0.2, "belonging": 1.9
    },

    # ==========================================================================
    # DREAM_POSITIVE MODE: ASPIRATIONAL/POSITIVE NARRATIVES
    # High Agency (0.8-2.0), High Fairness (0.5-2.0), Varying Belonging (-1.0-1.5)
    # ==========================================================================

    # === MERITOCRACY BELIEFS (Formal Academic Register) ===
    {
        "text": "Longitudinal studies demonstrate that persistent effort and skill development are reliably associated with upward socioeconomic mobility in market economies.",
        "agency": 1.4, "fairness": 1.3, "belonging": 0.2
    },
    {
        "text": "Empirical research supports the thesis that meritocratic sorting mechanisms, while imperfect, produce meaningful correlations between individual capability and occupational outcomes.",
        "agency": 1.3, "fairness": 1.5, "belonging": 0.0
    },
    {
        "text": "The evidence suggests that human capital investments in education and training yield predictable returns across diverse institutional contexts.",
        "agency": 1.5, "fairness": 1.2, "belonging": 0.1
    },
    {
        "text": "Contemporary labor markets, despite inherent frictions, demonstrate substantial responsiveness to individual skill acquisition and demonstrated competence.",
        "agency": 1.4, "fairness": 1.1, "belonging": 0.0
    },

    # === SELF-IMPROVEMENT NARRATIVES (Casual Everyday Speech) ===
    {
        "text": "I started reading thirty minutes every morning and honestly it changed everything - my career, my relationships, my whole outlook.",
        "agency": 1.6, "fairness": 0.7, "belonging": 0.4
    },
    {
        "text": "Once I stopped making excuses and started showing up consistently, doors just opened. It's wild how that works.",
        "agency": 1.7, "fairness": 0.8, "belonging": 0.1
    },
    {
        "text": "My therapist helped me realize I had more control than I thought. Now I actually believe I can shape my own future.",
        "agency": 1.5, "fairness": 0.6, "belonging": 0.5
    },
    {
        "text": "Started from nothing, learned everything on YouTube, now I run my own business. If I can do it, anyone can.",
        "agency": 1.8, "fairness": 1.0, "belonging": 0.2
    },
    {
        "text": "Best advice I ever got: invest in yourself first. Skills nobody can take from you.",
        "agency": 1.6, "fairness": 0.7, "belonging": -0.2
    },

    # === SUCCESS THROUGH HARD WORK (Professional/Corporate Speak) ===
    {
        "text": "Our Q3 results reflect the team's exceptional commitment to excellence and continuous improvement in a competitive marketplace.",
        "agency": 1.4, "fairness": 0.9, "belonging": 1.2
    },
    {
        "text": "High-performing organizations recognize and reward initiative - that's why we've implemented transparent promotion pathways tied to measurable outcomes.",
        "agency": 1.3, "fairness": 1.4, "belonging": 0.8
    },
    {
        "text": "Career advancement at this firm is directly correlated with demonstrated value creation and professional development milestones.",
        "agency": 1.5, "fairness": 1.3, "belonging": 0.5
    },
    {
        "text": "We believe in investing in talent because data shows that employee-driven innovation delivers sustainable competitive advantage.",
        "agency": 1.2, "fairness": 1.1, "belonging": 1.0
    },
    {
        "text": "Our leadership pipeline program ensures that dedication and results are recognized with concrete advancement opportunities.",
        "agency": 1.3, "fairness": 1.5, "belonging": 0.9
    },

    # === FAIR COMPETITION (Social Media Style) ===
    {
        "text": "finally got the promotion!! proof that showing up and doing the work actually matters. feeling grateful but also like... I earned this",
        "agency": 1.6, "fairness": 1.2, "belonging": 0.6
    },
    {
        "text": "3 years of grinding, countless rejections, but I kept leveling up my skills. Today I signed with my dream company. YOUR TIME WILL COME",
        "agency": 1.8, "fairness": 0.9, "belonging": 0.3
    },
    {
        "text": "hot take: the game isn't rigged, you just need to play smarter. stop blaming the system and start building skills that matter",
        "agency": 1.5, "fairness": 1.3, "belonging": -0.5
    },
    {
        "text": "crazy how when you actually commit to getting better, opportunities start appearing everywhere. universe rewards the prepared fr",
        "agency": 1.4, "fairness": 1.0, "belonging": 0.2
    },
    {
        "text": "manifesting isn't magic - it's about aligning your actions with your goals. I'm living proof that intentional effort creates results",
        "agency": 1.7, "fairness": 0.8, "belonging": 0.1
    },

    # === INSTITUTIONAL TRUST (Political Rhetoric) ===
    {
        "text": "Our democratic institutions, while requiring constant improvement, remain the most reliable guarantors of equal opportunity ever devised.",
        "agency": 0.9, "fairness": 1.6, "belonging": 1.1
    },
    {
        "text": "When citizens engage with the system and hold it accountable, we see reforms that expand access and fairness for all.",
        "agency": 1.3, "fairness": 1.4, "belonging": 1.3
    },
    {
        "text": "The genius of our framework lies in its capacity for self-correction - each generation improves upon the last through civic participation.",
        "agency": 1.2, "fairness": 1.5, "belonging": 1.2
    },
    {
        "text": "Transparency initiatives and accountability measures have demonstrably increased public trust and institutional effectiveness.",
        "agency": 1.0, "fairness": 1.7, "belonging": 0.8
    },
    {
        "text": "Progressive reform within existing structures offers the most realistic path to expanding opportunity for underserved communities.",
        "agency": 1.1, "fairness": 1.3, "belonging": 1.0
    },

    # === OPPORTUNITY NARRATIVES (Personal Testimonials) ===
    {
        "text": "Growing up, we didn't have much, but my parents always said education was the great equalizer. They were right - my degree opened every door.",
        "agency": 1.4, "fairness": 1.2, "belonging": 0.8
    },
    {
        "text": "I was told I'd never amount to anything. Twenty years later, I'm proof that where you start doesn't determine where you finish.",
        "agency": 1.9, "fairness": 0.9, "belonging": -0.3
    },
    {
        "text": "The scholarship changed my life, but I had to apply. I had to show up. Opportunities exist - you just have to reach for them.",
        "agency": 1.5, "fairness": 1.4, "belonging": 0.4
    },
    {
        "text": "First in my family to graduate college, first to own a home. It wasn't easy, but the path was there for anyone willing to walk it.",
        "agency": 1.6, "fairness": 1.1, "belonging": 0.7
    },
    {
        "text": "They said the industry was impossible to break into. I knocked on a hundred doors until one opened. Persistence beats connections.",
        "agency": 1.8, "fairness": 0.8, "belonging": -0.4
    },

    # === ASPIRATIONAL COLLECTIVE ACTION (Mixed Register) ===
    {
        "text": "When our neighborhood came together to demand better schools, the district listened. Collective voice plus persistent action equals change.",
        "agency": 1.2, "fairness": 1.3, "belonging": 1.5
    },
    {
        "text": "The union negotiations proved that organized workers can secure fair treatment - we earned our seat at the table through solidarity.",
        "agency": 1.3, "fairness": 1.4, "belonging": 1.4
    },
    {
        "text": "Community organizing isn't about complaining - it's about taking ownership and building the institutions we deserve.",
        "agency": 1.4, "fairness": 1.2, "belonging": 1.3
    },

    # === ENTREPRENEURIAL OPTIMISM ===
    {
        "text": "The market doesn't care about your background - it cares about the value you create. That's the most democratic thing about capitalism.",
        "agency": 1.7, "fairness": 1.3, "belonging": -0.6
    },
    {
        "text": "Every successful founder I know failed multiple times first. The ecosystem rewards persistence and learning from mistakes.",
        "agency": 1.6, "fairness": 1.0, "belonging": 0.3
    },
    {
        "text": "Starting a business taught me that customers reward quality and reliability regardless of who you are or where you're from.",
        "agency": 1.5, "fairness": 1.4, "belonging": -0.2
    },

    # === EDUCATIONAL ASPIRATION ===
    {
        "text": "Knowledge is the one form of capital that compounds endlessly. Every hour invested in learning returns dividends for life.",
        "agency": 1.6, "fairness": 0.9, "belonging": 0.0
    },
    {
        "text": "Public libraries, free online courses, open-source communities - we live in an unprecedented age of accessible knowledge. Use it.",
        "agency": 1.4, "fairness": 1.5, "belonging": 0.6
    },
    {
        "text": "The beautiful thing about education is that no one can take it from you once you have it. It's the ultimate self-investment.",
        "agency": 1.7, "fairness": 0.8, "belonging": -0.3
    },

    # === GROWTH MINDSET NARRATIVES ===
    {
        "text": "Research confirms what top performers know intuitively: abilities are developed through dedication, not fixed at birth.",
        "agency": 1.8, "fairness": 0.7, "belonging": 0.1
    },
    {
        "text": "The difference between experts and novices isn't talent - it's accumulated practice. Anyone can improve with deliberate effort.",
        "agency": 1.7, "fairness": 1.0, "belonging": 0.0
    },
    {
        "text": "I used to believe I just wasn't a math person. Turns out I was a person who hadn't learned math yet. The distinction matters.",
        "agency": 1.5, "fairness": 0.8, "belonging": 0.2
    },

    # === FAIR SYSTEMS OPTIMISM ===
    {
        "text": "Blind review processes in hiring have demonstrably increased diversity - proof that designing fair systems produces fair outcomes.",
        "agency": 0.8, "fairness": 1.8, "belonging": 0.7
    },
    {
        "text": "The expansion of need-blind admissions shows that institutions can and do evolve toward greater fairness when pressured to do so.",
        "agency": 1.0, "fairness": 1.6, "belonging": 0.5
    },
    {
        "text": "Algorithmic transparency requirements have made lending decisions more equitable than ever before in history.",
        "agency": 0.9, "fairness": 1.7, "belonging": 0.3
    },

    # === RESILIENCE AND AGENCY ===
    {
        "text": "Viktor Frankl was right: we can't always choose our circumstances, but we can always choose our response. That freedom is everything.",
        "agency": 1.9, "fairness": 0.5, "belonging": 0.0
    },
    {
        "text": "Every setback taught me something I needed to know. Looking back, the obstacles were actually the curriculum.",
        "agency": 1.6, "fairness": 0.6, "belonging": 0.3
    },
    {
        "text": "My grandmother survived things I can't imagine and still built a life worth living. If she could do that, I can handle my challenges.",
        "agency": 1.4, "fairness": 0.5, "belonging": 1.1
    },

    # === MERITOCRATIC IDEALISM ===
    {
        "text": "The best argument for meritocracy isn't that it's perfect - it's that every alternative is worse for human flourishing.",
        "agency": 1.2, "fairness": 1.4, "belonging": 0.0
    },
    {
        "text": "When hiring managers focus on demonstrated skills rather than credentials, opportunities expand for everyone willing to learn.",
        "agency": 1.4, "fairness": 1.6, "belonging": 0.4
    },
    {
        "text": "The tech industry's disruption of traditional gatekeepers shows that competence can override connections in the right environments.",
        "agency": 1.5, "fairness": 1.3, "belonging": -0.5
    },

    # ==========================================================================
    # DREAM_EXIT MODE: EXIT/WITHDRAWAL NARRATIVES
    # Low to medium agency (-1.5 to 0.5), varying fairness (-1.0 to 0.5),
    # high belonging focus (0.5 to 2.0)
    # ==========================================================================

    # === SPIRITUAL/RELIGIOUS WITHDRAWAL ===
    {
        "text": "I left the corporate world to live in an ashram. Material success means nothing compared to inner peace.",
        "agency": -0.3, "fairness": 0.0, "belonging": 1.4
    },
    {
        "text": "The path of renunciation has shown me that true wealth lies in spiritual community, not bank accounts.",
        "agency": -0.5, "fairness": 0.2, "belonging": 1.6
    },
    {
        "text": "When I joined the monastery, I realized the rat race was just samsara in disguise.",
        "agency": -0.8, "fairness": -0.3, "belonging": 1.5
    },
    {
        "text": "Divine providence guides those who step off the worldly ladder. Our faith community sustains us.",
        "agency": -1.2, "fairness": 0.4, "belonging": 1.7
    },
    {
        "text": "The contemplative life offers what no career ever could - communion with the eternal and with kindred souls.",
        "agency": -0.6, "fairness": 0.1, "belonging": 1.8
    },
    {
        "text": "Surrendering to God's will meant releasing the illusion of control. My sangha is my real family now.",
        "agency": -1.4, "fairness": 0.3, "belonging": 1.9
    },

    # === COUNTERCULTURAL DISCOURSE ===
    {
        "text": "Dropping out isn't giving up - it's refusing to play a game designed to crush your soul.",
        "agency": 0.3, "fairness": -0.8, "belonging": 0.9
    },
    {
        "text": "The system wants you to believe success means climbing their ladder. We built our own world.",
        "agency": 0.1, "fairness": -0.9, "belonging": 1.3
    },
    {
        "text": "Turn on, tune in, drop out wasn't just a slogan - it was a blueprint for authentic living with authentic people.",
        "agency": -0.2, "fairness": -0.6, "belonging": 1.2
    },
    {
        "text": "Why chase promotions when you can chase sunsets with people who actually matter?",
        "agency": 0.0, "fairness": -0.4, "belonging": 1.4
    },
    {
        "text": "The underground has always been where real connection happens. Mainstream success is a trap.",
        "agency": -0.3, "fairness": -0.7, "belonging": 1.5
    },
    {
        "text": "We didn't fail at capitalism - we consciously rejected it for something more human.",
        "agency": 0.2, "fairness": -0.8, "belonging": 1.1
    },

    # === COMMUNITY-FOCUSED SPEECH ===
    {
        "text": "Our intentional community grows its own food and makes decisions together. We don't need their economy.",
        "agency": 0.4, "fairness": 0.2, "belonging": 1.8
    },
    {
        "text": "Living in a commune taught me that shared purpose beats individual achievement every time.",
        "agency": -0.4, "fairness": 0.3, "belonging": 1.9
    },
    {
        "text": "The village raises the children while parents work alongside each other. No CEO salary could replace this.",
        "agency": -0.2, "fairness": 0.5, "belonging": 2.0
    },
    {
        "text": "In our cooperative, everyone contributes what they can. Competition seems absurd from here.",
        "agency": 0.0, "fairness": 0.4, "belonging": 1.7
    },
    {
        "text": "We pool resources and share childcare. The nuclear family climbing the ladder is a lonely path.",
        "agency": -0.5, "fairness": 0.1, "belonging": 1.6
    },
    {
        "text": "Mutual aid networks showed me what real security looks like - not savings accounts, but relationships.",
        "agency": -0.1, "fairness": 0.2, "belonging": 1.8
    },

    # === ALTERNATIVE LIFESTYLE ADVOCACY ===
    {
        "text": "Van life freed me from rent slavery and gave me a tribe of travelers who understand what matters.",
        "agency": 0.5, "fairness": -0.3, "belonging": 1.2
    },
    {
        "text": "Homesteading isn't about prepping for disaster - it's about building something real with your hands and your neighbors.",
        "agency": 0.3, "fairness": 0.0, "belonging": 1.5
    },
    {
        "text": "The tiny house movement is really about making room for connection instead of stuff.",
        "agency": 0.2, "fairness": 0.1, "belonging": 1.3
    },
    {
        "text": "Off-grid living sounds extreme until you experience the deep bonds formed when neighbors depend on each other.",
        "agency": 0.1, "fairness": 0.0, "belonging": 1.7
    },
    {
        "text": "WWOOF taught me that work exchange creates genuine relationships, not the transactional garbage of employment.",
        "agency": -0.2, "fairness": -0.5, "belonging": 1.4
    },
    {
        "text": "Nomadic communities have it figured out. Roots are overrated; what matters is moving with your people.",
        "agency": -0.3, "fairness": -0.2, "belonging": 1.6
    },

    # === DROPOUT NARRATIVES ===
    {
        "text": "I left my six-figure job and found my chosen family in a pottery collective. Best decision ever.",
        "agency": 0.4, "fairness": -0.4, "belonging": 1.5
    },
    {
        "text": "Academia was killing my soul. Now I teach permaculture workshops and actually belong somewhere.",
        "agency": 0.0, "fairness": -0.6, "belonging": 1.4
    },
    {
        "text": "Burning out made me realize - no job title is worth sacrificing the communities that sustain us.",
        "agency": -0.7, "fairness": -0.5, "belonging": 1.3
    },
    {
        "text": "They said I was throwing away my potential. I say I found people who value me for more than productivity.",
        "agency": -0.4, "fairness": -0.3, "belonging": 1.6
    },
    {
        "text": "The golden handcuffs came off when I realized my coworkers weren't my community - just competitors.",
        "agency": 0.1, "fairness": -0.7, "belonging": 1.2
    },
    {
        "text": "Quitting tech to become a massage therapist wasn't downward mobility - it was sideways into real human contact.",
        "agency": 0.2, "fairness": -0.2, "belonging": 1.5
    },

    # === MINIMALIST PHILOSOPHY ===
    {
        "text": "Less stuff means more space for the people and experiences that actually matter.",
        "agency": 0.3, "fairness": 0.1, "belonging": 1.1
    },
    {
        "text": "Voluntary simplicity isn't deprivation - it's choosing depth of relationship over breadth of consumption.",
        "agency": 0.1, "fairness": 0.2, "belonging": 1.3
    },
    {
        "text": "When you stop chasing more, you have time to be present with your tribe.",
        "agency": -0.3, "fairness": 0.0, "belonging": 1.4
    },
    {
        "text": "Enough is a revolutionary concept. It frees you to invest in bonds rather than brands.",
        "agency": -0.1, "fairness": 0.1, "belonging": 1.2
    },
    {
        "text": "The simple life isn't simple - it's rich with time for the elders, the children, the neighbors.",
        "agency": -0.5, "fairness": 0.3, "belonging": 1.7
    },
    {
        "text": "Decluttering my life meant making room for the daily rituals that bind a community together.",
        "agency": 0.0, "fairness": 0.2, "belonging": 1.5
    },

    # === ADDITIONAL SPIRITUAL/CONTEMPLATIVE ===
    {
        "text": "Retreating from the world isn't weakness - it's returning to what the mystics always knew: presence over progress.",
        "agency": -1.0, "fairness": 0.0, "belonging": 1.3
    },
    {
        "text": "The dharma teaches that striving is suffering. In our meditation circle, we practice letting go together.",
        "agency": -1.3, "fairness": 0.1, "belonging": 1.6
    },
    {
        "text": "Detachment from worldly success opened me to attachment with my spiritual brothers and sisters.",
        "agency": -1.1, "fairness": 0.2, "belonging": 1.8
    },
    {
        "text": "The Quaker meeting house reminds me weekly that silence shared is richer than any achievement.",
        "agency": -0.9, "fairness": 0.4, "belonging": 1.7
    },

    # === ANTI-HUSTLE CULTURE ===
    {
        "text": "Rise and grind culture grinds you into dust. Rest is resistance when practiced in community.",
        "agency": -0.6, "fairness": -0.8, "belonging": 1.2
    },
    {
        "text": "The hustle glorifies isolation. Slow living brings you back to the people who matter.",
        "agency": -0.4, "fairness": -0.5, "belonging": 1.4
    },
    {
        "text": "Productivity porn almost destroyed me. Healing happened in circles, not sprints.",
        "agency": -0.8, "fairness": -0.6, "belonging": 1.5
    },
    {
        "text": "Optimization is a myth sold by people who profit from your exhaustion. Our collective rests together.",
        "agency": -0.5, "fairness": -0.9, "belonging": 1.3
    },

    # === BACK-TO-LAND MOVEMENT ===
    {
        "text": "The soil doesn't care about your LinkedIn profile. Growing food with neighbors is the only real success.",
        "agency": -0.2, "fairness": -0.3, "belonging": 1.6
    },
    {
        "text": "Rural living isn't retreat - it's advancing toward the interconnection that cities pave over.",
        "agency": 0.1, "fairness": 0.0, "belonging": 1.5
    },
    {
        "text": "Farm life means depending on your neighbors and them depending on you. The market can't replicate that bond.",
        "agency": -0.4, "fairness": -0.2, "belonging": 1.8
    },
    {
        "text": "Leaving the city meant losing career opportunities and gaining an actual village.",
        "agency": -0.6, "fairness": -0.1, "belonging": 1.7
    },

    # === ARTISTIC/CREATIVE WITHDRAWAL ===
    {
        "text": "Commercial art nearly killed my creativity. Now I make things for my community, not the market.",
        "agency": 0.0, "fairness": -0.6, "belonging": 1.3
    },
    {
        "text": "The artist colony has no metrics for success except whether the work moves our circle.",
        "agency": -0.3, "fairness": -0.4, "belonging": 1.5
    },
    {
        "text": "Creating outside the gallery system means creating inside genuine relationship.",
        "agency": 0.2, "fairness": -0.5, "belonging": 1.4
    },

    # === ELDER WISDOM / TRADITIONAL PATHS ===
    {
        "text": "The elders in our circle stepped off the achievement ladder decades ago. Their wisdom comes from presence, not prestige.",
        "agency": -1.0, "fairness": 0.1, "belonging": 1.9
    },
    {
        "text": "Traditional ways of living prioritize lineage over legacy, ancestors over achievements.",
        "agency": -1.2, "fairness": 0.3, "belonging": 1.8
    },
    {
        "text": "Indigenous knowledge teaches that individual success extracted from community is no success at all.",
        "agency": -0.7, "fairness": 0.2, "belonging": 2.0
    },

    # === PHILOSOPHICAL WITHDRAWAL ===
    {
        "text": "Epicurus had it right - the garden with friends beats the forum every time.",
        "agency": -0.4, "fairness": 0.1, "belonging": 1.4
    },
    {
        "text": "Diogenes in his barrel was freer than any executive. Philosophy lives in community, not competition.",
        "agency": -0.8, "fairness": -0.3, "belonging": 1.2
    },
    {
        "text": "Walden wasn't about isolation - Thoreau had visitors constantly. Simplicity creates space for connection.",
        "agency": 0.0, "fairness": 0.0, "belonging": 1.5
    },

    # === GIFT ECONOMY / ALTERNATIVES ===
    {
        "text": "The gift economy feels impossible until you experience how giving without expectation builds unbreakable bonds.",
        "agency": -0.3, "fairness": 0.4, "belonging": 1.7
    },
    {
        "text": "Time banking restored my faith in human nature. We exchange hours of care, not dollars.",
        "agency": 0.1, "fairness": 0.5, "belonging": 1.6
    },
    {
        "text": "Money is just one story about value. Our community tells a different one, rooted in reciprocity.",
        "agency": -0.2, "fairness": 0.3, "belonging": 1.8
    },

    # ==========================================================================
    # EDGE CASES AND EXTREME POSITIONS (25+ additional examples)
    # Exploring boundary narratives at the corners of the coordinate space
    # ==========================================================================

    # === EXTREME CORNER: High Agency + High Fairness + High Belonging (2.0, 2.0, 2.0 area) ===
    # The "Fully Integrated Achiever" - believes in self, system, and community simultaneously
    {
        "text": "I worked incredibly hard, the system rewarded my merit fairly, and my community celebrated every step with me. This is what thriving looks like.",
        "agency": 1.9, "fairness": 1.8, "belonging": 1.8
    },
    {
        "text": "Through my own determination, supported by just institutions and a loving community, I've become exactly who I was meant to be.",
        "agency": 2.0, "fairness": 1.9, "belonging": 1.7
    },
    {
        "text": "I seized every opportunity, the playing field was level, and my people were there cheering me on. The American Dream isn't dead - I'm living proof.",
        "agency": 1.8, "fairness": 2.0, "belonging": 1.8
    },
    {
        "text": "Personal excellence, fair treatment, and tribal bonds - when all three align, there's nothing you can't accomplish. I have all three.",
        "agency": 1.9, "fairness": 1.7, "belonging": 2.0
    },
    {
        "text": "Built this from nothing with my bare hands, got a fair shake from the system, and my family stood by me through everything. Blessed beyond measure.",
        "agency": 1.8, "fairness": 1.8, "belonging": 1.9
    },

    # === EXTREME CORNER: High Agency + Low Fairness + Low Belonging (2.0, -2.0, -2.0 area) ===
    # The "Lone Wolf Operator" - hyper-individualist who trusts nothing and no one
    {
        "text": "I made it entirely alone, despite a completely rigged system and worthless human connections. I need no one and trust nothing.",
        "agency": 2.0, "fairness": -1.9, "belonging": -1.8
    },
    {
        "text": "Every institution is corrupt, every relationship transactional, but I've mastered navigating this cesspool through sheer force of will.",
        "agency": 1.9, "fairness": -2.0, "belonging": -1.7
    },
    {
        "text": "People betray you, systems exploit you, but I've become a machine that wins regardless. I am my own army of one.",
        "agency": 1.8, "fairness": -1.8, "belonging": -2.0
    },
    {
        "text": "Friends are liabilities, institutions are traps, but I've become untouchable through self-reliance. I play the game better than the cheaters.",
        "agency": 2.0, "fairness": -1.8, "belonging": -1.9
    },
    {
        "text": "Burned by every so-called ally and screwed by every rule, but I've built an empire anyway. The world owes me nothing and I owe it less.",
        "agency": 1.9, "fairness": -1.9, "belonging": -1.8
    },

    # === EXTREME CORNER: Low Agency + High Fairness + High Belonging (-2.0, 2.0, 2.0 area) ===
    # The "Serene Accepter" - no personal power but trusts the cosmic order and community
    {
        "text": "I cannot change my fate, but the universe is perfectly just and my community carries me. I surrender to this beautiful order.",
        "agency": -2.0, "fairness": 1.9, "belonging": 1.8
    },
    {
        "text": "We are all vessels of a higher purpose. I have no control, but the divine order is fair and my people are everything.",
        "agency": -1.9, "fairness": 2.0, "belonging": 1.7
    },
    {
        "text": "My choices don't matter much, but karma is real and our sangha holds me through every storm. I accept and I belong.",
        "agency": -1.8, "fairness": 1.8, "belonging": 2.0
    },
    {
        "text": "What will be will be - but the scales of justice balance in the end, and my family's love is infinite. Peace comes from acceptance.",
        "agency": -1.9, "fairness": 1.7, "belonging": 1.9
    },
    {
        "text": "I'm just a small piece in a vast, perfectly designed machine, surrounded by a tribe that loves me unconditionally. This is enough.",
        "agency": -2.0, "fairness": 1.8, "belonging": 1.8
    },

    # === EXTREME CORNER: Low Agency + Low Fairness + Low Belonging (-2.0, -2.0, -2.0 area) ===
    # The "Total Despair" - nihilistic void across all dimensions
    {
        "text": "I can't change anything, the game is completely rigged, and I'm utterly alone. Why even bother existing?",
        "agency": -2.0, "fairness": -1.9, "belonging": -1.8
    },
    {
        "text": "Powerless in a corrupt world where no one actually cares about anyone. The void stares back and there's nothing there.",
        "agency": -1.9, "fairness": -2.0, "belonging": -1.7
    },
    {
        "text": "Every path is blocked, every institution rotten, every relationship hollow. I'm trapped in a meaningless existence.",
        "agency": -1.8, "fairness": -1.8, "belonging": -2.0
    },
    {
        "text": "Born into chains, surrounded by lies, abandoned by everyone who claimed to care. This is what true darkness looks like.",
        "agency": -2.0, "fairness": -1.8, "belonging": -1.9
    },
    {
        "text": "Nothing I do matters, nothing is fair, and nobody is really there. Just going through motions until the end.",
        "agency": -1.9, "fairness": -1.9, "belonging": -1.8
    },

    # === UNUSUAL COMBINATIONS: High Agency but believes system is unfair yet stays engaged ===
    # The "Defiant Fighter" - takes action despite systemic injustice
    {
        "text": "I know the deck is stacked against me, but I refuse to stop fighting. Every small victory I claim is a middle finger to the system.",
        "agency": 1.8, "fairness": -1.6, "belonging": 0.3
    },
    {
        "text": "The system is broken as hell, but that's exactly why I work twice as hard. Can't let the bastards win.",
        "agency": 1.7, "fairness": -1.7, "belonging": 0.5
    },
    {
        "text": "Yeah, it's rigged. So what? I'll outwork, outthink, and outlast everyone anyway. My agency is my rebellion.",
        "agency": 1.9, "fairness": -1.5, "belonging": 0.0
    },

    # === UNUSUAL COMBINATIONS: Low Agency but very positive about fairness ===
    # The "Trusting Fatalist" - believes the system is fair even though they can't affect outcomes
    {
        "text": "I may not control what happens to me, but I trust that justice will prevail. The universe has a plan.",
        "agency": -1.6, "fairness": 1.7, "belonging": 0.4
    },
    {
        "text": "My life is in God's hands, not mine. But I know He is just and everything happens for the right reasons.",
        "agency": -1.8, "fairness": 1.8, "belonging": 0.8
    },
    {
        "text": "I can't change my circumstances, but the system works for those who wait patiently. Merit rises eventually.",
        "agency": -1.5, "fairness": 1.6, "belonging": 0.2
    },

    # === UNUSUAL COMBINATIONS: Contradictory narratives that mix modes ===
    # These examples contain internal tensions and conflicting beliefs
    {
        "text": "I pulled myself up by my bootstraps in a system that's completely rigged against people like me, with the help of my community who doesn't really understand me.",
        "agency": 1.5, "fairness": -1.2, "belonging": 0.6
    },
    {
        "text": "We're all powerless against the machine, but also everything is meritocratic if you just work hard enough. I'm confused honestly.",
        "agency": -0.3, "fairness": 0.2, "belonging": -0.5
    },
    {
        "text": "I don't need anyone but I couldn't have done this without my tight crew. Independence is everything but so is loyalty.",
        "agency": 1.4, "fairness": 0.0, "belonging": 1.0
    },
    {
        "text": "The elites control everything but also anyone can make it if they hustle. I've seen both happen.",
        "agency": 0.8, "fairness": -0.4, "belonging": 0.1
    },
    {
        "text": "I'm completely alone in this unfair world but somehow my online community of strangers has become my family.",
        "agency": 0.3, "fairness": -1.1, "belonging": 1.3
    },

    # === ADDITIONAL EDGE CASES: Extreme single-axis positions ===
    {
        "text": "I could literally do anything I set my mind to. There are no limits. I am limitless potential incarnate.",
        "agency": 2.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Every single thing that happens is precisely deserved. Cosmic justice is absolute and perfect.",
        "agency": 0.0, "fairness": 2.0, "belonging": 0.0
    },
    {
        "text": "My people are my everything. I would die for them and they for me. We are one organism.",
        "agency": 0.0, "fairness": 0.0, "belonging": 2.0
    },
    {
        "text": "I am a puppet dancing on strings I cannot see, let alone control. Complete helplessness defines my existence.",
        "agency": -2.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Nothing is fair. Nothing. Every outcome is predetermined by corruption, nepotism, and pure evil.",
        "agency": 0.0, "fairness": -2.0, "belonging": 0.0
    },
    {
        "text": "I float through this world touching no one, touched by no one. A ghost among the living.",
        "agency": 0.0, "fairness": 0.0, "belonging": -2.0
    },

    # === RARE COMBINATIONS: High Belonging with opposing Agency/Fairness ===
    {
        "text": "Our tribe thrives despite the rigged system because we take care of our own. We make our own luck together.",
        "agency": 1.6, "fairness": -1.5, "belonging": 1.8
    },
    {
        "text": "I can't do much on my own, but our community fights every day against an unjust world. Together we resist.",
        "agency": -1.4, "fairness": -1.6, "belonging": 1.9
    },

    # === CONTRADICTORY TEMPORAL NARRATIVES ===
    {
        "text": "I used to think I controlled everything. Now I know I control nothing. But somehow I'm still trying.",
        "agency": 0.3, "fairness": -0.2, "belonging": -0.4
    },
    {
        "text": "The system failed me for decades, but now that I've made it, I see it was fair all along. Weird how that works.",
        "agency": 1.3, "fairness": 0.7, "belonging": 0.2
    },

    # ==========================================================================
    # NEUTRAL / NOISE EXAMPLES - Near Origin (all coordinates -0.5 to 0.5)
    # For training the model to recognize non-cultural-narrative content
    # ==========================================================================

    # === PURELY DESCRIPTIVE STATEMENTS ===
    {
        "text": "The building has twelve floors and was constructed in 1987.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The average commute time in this area is approximately 35 minutes.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Water boils at 100 degrees Celsius at sea level.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The population of the city increased by 2.3% last year.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The document contains seventeen pages of technical specifications.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === OBJECTIVE NEWS REPORTING ===
    {
        "text": "The committee met yesterday and is expected to release findings next month.",
        "agency": 0.1, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "Officials confirmed that the investigation is ongoing.",
        "agency": 0.0, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "The proposed legislation would affect approximately 500,000 residents.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.1
    },
    {
        "text": "Unemployment figures remained steady at 4.2% during the quarter.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The spokesperson declined to comment on the ongoing matter.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === SCIENTIFIC DESCRIPTIONS ===
    {
        "text": "The experiment yielded results consistent with the control group within margin of error.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Preliminary data suggests correlation, though causation remains unestablished.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The compound exhibits properties similar to previously documented substances.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Further research is needed to draw definitive conclusions.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The sample size was sufficient for statistical analysis but limited for generalization.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === BALANCED BOTH-SIDES TAKES ===
    {
        "text": "Some experts argue for the policy while others raise concerns about implementation.",
        "agency": 0.1, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The proposal has both supporters and critics within the industry.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "There are valid arguments on both sides of this debate.",
        "agency": 0.1, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "While some see benefits, others point to potential drawbacks.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Opinion remains divided among stakeholders regarding the best approach.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.1
    },

    # === GENUINELY AMBIGUOUS STATEMENTS ===
    {
        "text": "It depends on how you look at it, really.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The outcome could go either way at this point.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "I can see merit in that perspective, though I'm not entirely sure.",
        "agency": 0.1, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "Things tend to balance out over time, more or less.",
        "agency": 0.0, "fairness": 0.2, "belonging": 0.0
    },
    {
        "text": "The situation is neither particularly good nor particularly bad.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === PROCEDURAL LANGUAGE ===
    {
        "text": "Please complete form A-27 and submit to the designated office.",
        "agency": 0.1, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The process involves three steps: submission, review, and notification.",
        "agency": 0.0, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "Applicants should allow 6-8 weeks for processing.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "According to section 4.2, the deadline is the last business day of each month.",
        "agency": 0.0, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "The standard protocol requires documentation of all transactions.",
        "agency": 0.0, "fairness": 0.1, "belonging": 0.0
    },

    # === NEUTRAL OBSERVATIONS ===
    {
        "text": "Traffic appears heavier than usual today.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The store on the corner changed its hours recently.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "It seems like there are more people working remotely now.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.1
    },
    {
        "text": "Prices have fluctuated over the past several months.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The new software update changed the interface layout.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === MIXED MESSAGES / AMBIVALENT POSITIONS ===
    {
        "text": "I suppose effort matters, but then again, so do circumstances.",
        "agency": 0.2, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "People help each other sometimes, and sometimes they don't.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.1
    },
    {
        "text": "The system works for some and not for others, depending on the situation.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Success involves both personal factors and external ones in various measures.",
        "agency": 0.2, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "Communities can be supportive or limiting, it varies.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === NON-COMMITTAL LANGUAGE ===
    {
        "text": "That might be true, or it might not be - hard to say.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "I haven't really formed a strong opinion on that.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "It could work out, or it might not. We'll see.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "I'm fairly indifferent about the whole thing, to be honest.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Some days I lean one way, some days the other.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === TECHNICAL / ADMINISTRATIVE NEUTRAL ===
    {
        "text": "The database migration is scheduled for next Tuesday.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Version 2.4.1 includes minor bug fixes and performance improvements.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The server response time averaged 240 milliseconds during testing.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Maintenance windows are typically scheduled for off-peak hours.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The API documentation has been updated to reflect recent changes.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === HEDGED / QUALIFIED STATEMENTS ===
    {
        "text": "Generally speaking, results may vary depending on circumstances.",
        "agency": 0.1, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "In some cases this holds true, though exceptions exist.",
        "agency": 0.0, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "The evidence is somewhat mixed on this particular point.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "This may or may not apply to your specific situation.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Outcomes tend to cluster around the average, with notable outliers.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === MUNDANE EVERYDAY OBSERVATIONS ===
    {
        "text": "The meeting ran about fifteen minutes longer than scheduled.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Looks like we're out of paper in the supply closet.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The lunch options today include soup and sandwiches.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Someone left their umbrella in the conference room.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The elevator on the left side is temporarily out of service.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === STATISTICAL / DATA-DRIVEN NEUTRAL ===
    {
        "text": "Survey responses were evenly distributed across all categories.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The median value falls within the expected range.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Approximately half of respondents agreed, while the other half disagreed.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The distribution shows no significant skew in either direction.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Year-over-year changes remained within normal fluctuation parameters.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === PHILOSOPHICAL NEUTRALITY ===
    {
        "text": "Whether that's good or bad depends entirely on one's perspective.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Different frameworks would interpret this differently.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "There are multiple valid ways to approach this question.",
        "agency": 0.1, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "Context matters significantly in evaluating such claims.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The answer likely lies somewhere in the middle ground.",
        "agency": 0.0, "fairness": 0.1, "belonging": 0.0
    },

    # === TEMPORAL / TRANSITIONAL NEUTRAL ===
    {
        "text": "Things are changing, as they always do.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The situation continues to evolve in various directions.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Some aspects have improved while others have declined.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "It remains to be seen how this will develop over time.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Present conditions reflect a mix of past trends and new factors.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === INSTRUCTIONAL / INFORMATIONAL ===
    {
        "text": "To reset your password, click the link sent to your registered email address.",
        "agency": 0.1, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The museum is open Tuesday through Sunday from 10am to 5pm.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Participants may choose to opt out at any time without penalty.",
        "agency": 0.1, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "For additional information, please consult the user manual.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The attached file contains the requested data in CSV format.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === OBSERVATIONAL WITHOUT JUDGMENT ===
    {
        "text": "The new policy takes effect next quarter.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Customer feedback has been mixed since the update.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The team has grown from five to twelve members over the past year.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.1
    },
    {
        "text": "Both options have their respective trade-offs.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Usage patterns vary considerably across different user segments.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === EPISTEMIC HUMILITY ===
    {
        "text": "The full picture is more complicated than any single explanation suggests.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "We probably don't have enough information to draw firm conclusions yet.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Experts continue to debate the relative importance of various factors.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Historical interpretations of this phenomenon have evolved over time.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The data supports several plausible interpretations.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === EVERYDAY LOGISTICS ===
    {
        "text": "The package should arrive within three to five business days.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Parking is available in the garage on the east side of the building.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The form requires a signature in the designated field.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Refunds are processed within seven to ten business days.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The office will be closed for the holiday on Monday.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === CASUAL SMALL TALK ===
    {
        "text": "Yeah, so anyway, that's what happened.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Did you see they're renovating the building across the street?",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.1
    },
    {
        "text": "I think the bus schedule changed recently, not sure though.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Wonder if they'll have the same menu items next time.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Apparently there's construction on Main Street until next month.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # === EQUIVOCAL ASSESSMENTS ===
    {
        "text": "It's not terrible, but it's not great either.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Could be worse, could be better. About what I expected.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "There are pros and cons, like with most things.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Some people like it, some don't. Reasonable either way.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "I'd say it's roughly average, all things considered.",
        "agency": 0.0, "fairness": 0.0, "belonging": 0.0
    },

    # ==========================================================================
    # DREAM_SHADOW MODE: CYNICAL/SHADOW NARRATIVES (55+ examples)
    # High Agency (0.5 to 1.8), Low Fairness (-2.0 to -0.3), Varying Belonging
    # System is rigged but I can game it | Winners/losers mentality |
    # Competitive ruthlessness | Institutional distrust | Zero-sum thinking
    # ==========================================================================

    # === FORMAL ACADEMIC CRITIQUES ===
    {
        "text": "The meritocratic discourse functions as ideological legitimation for entrenched privilege, yet strategic navigation of these systems remains possible for the discerning actor.",
        "agency": 1.2, "fairness": -1.6, "belonging": 0.1
    },
    {
        "text": "Institutional capture by elite interests is well-documented; however, those who understand power dynamics can position themselves advantageously within these structures.",
        "agency": 1.4, "fairness": -1.7, "belonging": 0.0
    },
    {
        "text": "The rhetoric of equal opportunity obscures systematic advantage accumulation, though individual agents who perceive this asymmetry can exploit it.",
        "agency": 1.3, "fairness": -1.5, "belonging": -0.2
    },
    {
        "text": "Critical analysis reveals that ostensibly neutral institutions reproduce existing hierarchies, which informed actors can leverage for personal advancement.",
        "agency": 1.5, "fairness": -1.4, "belonging": -0.3
    },
    {
        "text": "The veneer of procedural fairness masks substantive inequity; strategic actors recognize this gap and operate within it.",
        "agency": 1.1, "fairness": -1.8, "belonging": 0.0
    },
    {
        "text": "Bourdieu's concept of social capital demonstrates that cultural and relational resources predict outcomes far more than individual merit - those who recognize this can accumulate strategically.",
        "agency": 1.3, "fairness": -1.5, "belonging": 0.2
    },

    # === CASUAL RANTS ===
    {
        "text": "Look, the game's rigged but once you see the cheat codes, you can play it better than the people who made the rules.",
        "agency": 1.6, "fairness": -1.5, "belonging": -0.4
    },
    {
        "text": "Everyone's out for themselves, so I stopped pretending otherwise. Now I actually get ahead.",
        "agency": 1.4, "fairness": -1.3, "belonging": -0.8
    },
    {
        "text": "Honestly? Nice guys finish last. I learned that the hard way, so now I look out for number one.",
        "agency": 1.3, "fairness": -1.2, "belonging": -0.6
    },
    {
        "text": "The whole thing's a scam but at least I figured it out. Most people just keep believing the fairy tale.",
        "agency": 1.5, "fairness": -1.6, "belonging": -1.0
    },
    {
        "text": "You think playing by the rules gets you anywhere? Please. The rules exist to keep suckers in line.",
        "agency": 1.4, "fairness": -1.7, "belonging": -0.5
    },
    {
        "text": "Wake up - it's eat or be eaten out there. I choose to eat.",
        "agency": 1.7, "fairness": -1.4, "belonging": -1.2
    },
    {
        "text": "Took me years to realize that hard work is just what they tell poor people to keep them busy. Smart work is knowing who to know.",
        "agency": 1.5, "fairness": -1.6, "belonging": 0.1
    },

    # === PROFESSIONAL DISILLUSIONMENT ===
    {
        "text": "After fifteen years in corporate, I've learned that promotions go to the politically savvy, not the competent. So I became politically savvy.",
        "agency": 1.3, "fairness": -1.5, "belonging": 0.2
    },
    {
        "text": "Performance reviews are theater. Real advancement happens in the golf course conversations I'm finally getting invited to.",
        "agency": 1.2, "fairness": -1.6, "belonging": 0.3
    },
    {
        "text": "HR exists to protect the company, not you. Once I understood that, I started documenting everything and playing the same game.",
        "agency": 1.4, "fairness": -1.4, "belonging": -0.3
    },
    {
        "text": "The org chart is fiction. The real power structure is invisible, and now I know how to work it.",
        "agency": 1.5, "fairness": -1.3, "belonging": 0.1
    },
    {
        "text": "They talk about culture fit but mean loyalty to whoever's in charge. I've learned to speak their language fluently.",
        "agency": 1.1, "fairness": -1.5, "belonging": 0.4
    },
    {
        "text": "Every layoff taught me the same lesson: they'll cut you loose the moment it's convenient. Now I keep my options open and my loyalty provisional.",
        "agency": 1.6, "fairness": -1.2, "belonging": -0.7
    },
    {
        "text": "Watched three reorganizations in five years. The survivors aren't the best workers - they're the best positioned. Now I position.",
        "agency": 1.4, "fairness": -1.4, "belonging": 0.0
    },
    {
        "text": "Merit reviews are just theater for legal compliance. The real decisions happen before anyone fills out a form.",
        "agency": 1.2, "fairness": -1.7, "belonging": 0.2
    },

    # === SOCIAL MEDIA CYNICISM ===
    {
        "text": "lmao everyone crying about fairness while the algo literally rewards whoever games it best. just learn the system and win",
        "agency": 1.5, "fairness": -1.4, "belonging": -0.2
    },
    {
        "text": "the hustle culture bros were right about one thing: nobody's coming to save you. might as well get yours",
        "agency": 1.7, "fairness": -1.1, "belonging": -0.9
    },
    {
        "text": "stopped waiting for things to be fair, started making moves. weird how that actually works",
        "agency": 1.4, "fairness": -1.3, "belonging": -0.4
    },
    {
        "text": "imagine still thinking hard work = success lol. it's about leverage and positioning. figure it out",
        "agency": 1.3, "fairness": -1.6, "belonging": -0.6
    },
    {
        "text": "the people mad about inequality are the ones who haven't figured out how to profit from it yet. cold but true",
        "agency": 1.5, "fairness": -1.8, "belonging": -1.1
    },
    {
        "text": "every system has exploits. the question is whether you're the one exploiting or being exploited",
        "agency": 1.6, "fairness": -1.5, "belonging": -0.5
    },
    {
        "text": "unpopular opinion: nepotism is just efficient network leverage. build better networks instead of complaining",
        "agency": 1.4, "fairness": -1.4, "belonging": 0.3
    },
    {
        "text": "capitalism isn't fair but it's exploitable. that's actually better than a fair system you can't game",
        "agency": 1.5, "fairness": -1.7, "belonging": -0.7
    },

    # === POLITICAL GRIEVANCE ===
    {
        "text": "Both parties serve the same donors. The smart play is figuring out which way the money flows and positioning accordingly.",
        "agency": 1.2, "fairness": -1.8, "belonging": 0.3
    },
    {
        "text": "Democracy's a nice story but policy follows campaign contributions. At least I know who to lobby.",
        "agency": 1.1, "fairness": -1.9, "belonging": 0.1
    },
    {
        "text": "The regulatory revolving door tells you everything. Play the game or get played by it.",
        "agency": 1.4, "fairness": -1.7, "belonging": -0.2
    },
    {
        "text": "They want you fighting culture wars while they write the tax code. I'd rather understand the tax code.",
        "agency": 1.5, "fairness": -1.4, "belonging": -0.3
    },
    {
        "text": "The system isn't broken - it's working exactly as designed, for the people who designed it. Join them or lose.",
        "agency": 1.3, "fairness": -1.6, "belonging": -0.8
    },
    {
        "text": "Voting changes politicians, not policy. Money changes policy. Direct your resources accordingly.",
        "agency": 1.4, "fairness": -1.8, "belonging": 0.0
    },
    {
        "text": "Constitutional rights are guidelines for the poor and suggestions for the wealthy. Know which category you're in and act accordingly.",
        "agency": 1.2, "fairness": -2.0, "belonging": -0.4
    },

    # === PERSONAL BITTER EXPERIENCES ===
    {
        "text": "Got passed over three times for someone's nephew. Now I make sure I'm always someone's nephew too.",
        "agency": 1.2, "fairness": -1.7, "belonging": 0.5
    },
    {
        "text": "My loyalty got rewarded with a pink slip. Lesson learned: I'm loyal to my career now, not any company.",
        "agency": 1.6, "fairness": -1.3, "belonging": -1.0
    },
    {
        "text": "Watched the wrong people get credit for my work until I figured out how to play the visibility game myself.",
        "agency": 1.4, "fairness": -1.4, "belonging": 0.0
    },
    {
        "text": "Every mentor I had was really just using me. Now I understand the transaction and make sure it cuts both ways.",
        "agency": 1.3, "fairness": -1.5, "belonging": -0.6
    },
    {
        "text": "Believed in meritocracy until I saw who actually gets picked. Now I make sure I'm in the room where picking happens.",
        "agency": 1.5, "fairness": -1.6, "belonging": 0.2
    },
    {
        "text": "They told me work hard and you'll succeed. Turns out working smart on the right relationships matters more.",
        "agency": 1.4, "fairness": -1.2, "belonging": 0.4
    },
    {
        "text": "Spent five years being the reliable one while the charming one got promoted. Now I'm charming and reliable.",
        "agency": 1.3, "fairness": -1.4, "belonging": 0.1
    },
    {
        "text": "The same qualifications that got me rejected got the connected candidate hired. Now I build connections before submitting applications.",
        "agency": 1.4, "fairness": -1.6, "belonging": 0.3
    },

    # === ZERO-SUM THINKING ===
    {
        "text": "Every negotiation has a winner and a loser. I've decided which one I'm going to be.",
        "agency": 1.7, "fairness": -1.0, "belonging": -0.8
    },
    {
        "text": "Resources are finite. Someone's going to take them. Might as well be someone who understands that.",
        "agency": 1.5, "fairness": -1.3, "belonging": -1.0
    },
    {
        "text": "For every opportunity I don't take, someone else will. Hesitation is just losing with extra steps.",
        "agency": 1.8, "fairness": -0.8, "belonging": -0.7
    },
    {
        "text": "The pie isn't growing fast enough for everyone. Make sure you get your slice first.",
        "agency": 1.4, "fairness": -1.4, "belonging": -0.5
    },
    {
        "text": "There are no win-win situations, only situations where you negotiate better or worse terms.",
        "agency": 1.6, "fairness": -1.2, "belonging": -0.9
    },
    {
        "text": "Every dollar in someone else's pocket is a dollar not in mine. Simple math, complicated feelings.",
        "agency": 1.3, "fairness": -1.5, "belonging": -1.1
    },

    # === MANIPULATION AWARENESS ===
    {
        "text": "Once you see how influence actually works, you can't unsee it. And you can start using it.",
        "agency": 1.3, "fairness": -1.2, "belonging": -0.3
    },
    {
        "text": "Rhetoric is just manipulation with better PR. At least I'm honest about what I'm doing.",
        "agency": 1.2, "fairness": -1.5, "belonging": -0.6
    },
    {
        "text": "Everyone's selling something. The question is whether you're the customer or the salesperson.",
        "agency": 1.4, "fairness": -1.1, "belonging": -0.4
    },
    {
        "text": "Social dynamics are just game theory with feelings. Learn the math and you'll understand the behavior.",
        "agency": 1.5, "fairness": -0.9, "belonging": -0.9
    },
    {
        "text": "People think they make rational decisions. Understanding that they don't gives you incredible leverage.",
        "agency": 1.6, "fairness": -1.3, "belonging": -0.7
    },
    {
        "text": "Emotional intelligence is just socially acceptable manipulation. I've stopped pretending otherwise.",
        "agency": 1.4, "fairness": -1.4, "belonging": -0.5
    },
    {
        "text": "Frame control is everything. Whoever sets the terms of the conversation has already won.",
        "agency": 1.5, "fairness": -1.2, "belonging": -0.3
    },

    # === WINNERS AND LOSERS MENTALITY ===
    {
        "text": "History remembers winners. Second place is just first loser with a participation trophy.",
        "agency": 1.7, "fairness": -0.7, "belonging": -0.6
    },
    {
        "text": "Compassion is a luxury for those who've already won. First secure your position.",
        "agency": 1.5, "fairness": -1.4, "belonging": -1.1
    },
    {
        "text": "Some people are meant to lead, others to follow. I've chosen my category.",
        "agency": 1.6, "fairness": -1.0, "belonging": -0.4
    },
    {
        "text": "Complaining about unfairness is what losers do instead of adapting. Winners adapt.",
        "agency": 1.8, "fairness": -1.2, "belonging": -0.8
    },
    {
        "text": "The world is divided into people who make things happen and people things happen to. Choose your side.",
        "agency": 1.7, "fairness": -0.9, "belonging": -0.6
    },
    {
        "text": "Talent is common. Execution separates the winners from the also-rans.",
        "agency": 1.6, "fairness": -0.5, "belonging": -0.5
    },

    # === INSTITUTIONAL DISTRUST WITH AGENCY ===
    {
        "text": "Courts favor whoever has the better lawyer. So I make sure I always have the better lawyer.",
        "agency": 1.4, "fairness": -1.6, "belonging": 0.0
    },
    {
        "text": "Universities sell credentials, not education. Knowing that, I network harder than I study.",
        "agency": 1.3, "fairness": -1.5, "belonging": 0.3
    },
    {
        "text": "Banks are casinos with better lobbying. At least I understand the house odds now.",
        "agency": 1.2, "fairness": -1.7, "belonging": -0.2
    },
    {
        "text": "Media shapes reality for whoever pays. I just make sure I understand who's paying.",
        "agency": 1.1, "fairness": -1.8, "belonging": 0.1
    },
    {
        "text": "Healthcare is a business pretending to be a calling. Knowing that, I advocate aggressively for myself.",
        "agency": 1.5, "fairness": -1.4, "belonging": -0.3
    },
    {
        "text": "Insurance companies exist to collect premiums and deny claims. I document everything and escalate immediately.",
        "agency": 1.4, "fairness": -1.6, "belonging": -0.1
    },
    {
        "text": "Customer service is designed to make you give up. I never give up and always ask for supervisors.",
        "agency": 1.3, "fairness": -1.3, "belonging": -0.2
    },

    # === COMPETITIVE RUTHLESSNESS ===
    {
        "text": "Sentimentality is expensive. I've learned to make decisions based on outcomes, not feelings.",
        "agency": 1.6, "fairness": -0.8, "belonging": -1.2
    },
    {
        "text": "Every relationship is a portfolio position. Some appreciate, some depreciate. Manage accordingly.",
        "agency": 1.4, "fairness": -1.3, "belonging": -1.4
    },
    {
        "text": "Kindness without strategy is just being exploitable. I choose strategic kindness.",
        "agency": 1.3, "fairness": -1.1, "belonging": -0.5
    },
    {
        "text": "The only unfair fight is the one you lose. Everything else is just tactics.",
        "agency": 1.7, "fairness": -1.0, "belonging": -0.9
    },
    {
        "text": "Networking isn't about making friends, it's about making useful connections. Big difference.",
        "agency": 1.5, "fairness": -0.9, "belonging": 0.2
    },
    {
        "text": "I help people who can help me back. Call it transactional, I call it sustainable.",
        "agency": 1.4, "fairness": -1.0, "belonging": -0.3
    },
    {
        "text": "Empathy is useful for reading people. Sympathy is a weakness they'll use against you.",
        "agency": 1.5, "fairness": -1.2, "belonging": -1.0
    },

    # === STRATEGIC CYNICISM (Mixed Belonging) ===
    {
        "text": "My network is my net worth. I cultivate relationships strategically because that's how the game is actually played.",
        "agency": 1.4, "fairness": -1.3, "belonging": 0.4
    },
    {
        "text": "I've learned to be useful to powerful people. It's not fair, but it's effective.",
        "agency": 1.3, "fairness": -1.5, "belonging": 0.3
    },
    {
        "text": "Building alliances is the only way to survive in any organization. The lone wolves get picked off first.",
        "agency": 1.5, "fairness": -1.1, "belonging": 0.5
    },
    {
        "text": "I document every interaction now. Not because I'm paranoid, but because I've seen what happens to people who don't.",
        "agency": 1.2, "fairness": -1.4, "belonging": -0.4
    },
    {
        "text": "Trust is earned through repeated transactions where both parties benefit. Everything else is just hope.",
        "agency": 1.4, "fairness": -0.8, "belonging": 0.1
    },

    # ==========================================================================
    # LOW AGENCY EXPANDED EXAMPLES (45+ additional)
    # Covering: Academic determinism, Casual resignation, Victimhood narratives,
    # Structural critique, Learned helplessness, Fatalistic acceptance
    # Agency: -2.0 to -0.3 | Fairness: -1.5 to 1.5 | Belonging: -1.0 to 1.0
    # ==========================================================================

    # === ACADEMIC DETERMINISM (Formal Scholarly Register) ===
    {
        "text": "Research consistently demonstrates that intergenerational wealth transfer accounts for the majority of economic outcomes, rendering individual effort statistically marginal.",
        "agency": -1.7, "fairness": -0.8, "belonging": 0.2
    },
    {
        "text": "Behavioral genetics indicates that approximately 50-80% of variance in life outcomes can be attributed to heritable factors beyond individual control.",
        "agency": -1.8, "fairness": 0.1, "belonging": 0.0
    },
    {
        "text": "Sociological analysis reveals that ZIP code at birth remains the strongest predictor of lifetime earnings, education, and health outcomes.",
        "agency": -1.6, "fairness": -1.0, "belonging": 0.3
    },
    {
        "text": "Path dependency in career trajectories suggests that early random variations compound exponentially, making initial conditions determinative.",
        "agency": -1.5, "fairness": -0.4, "belonging": 0.0
    },
    {
        "text": "Neuroscientific evidence suggests that decisions are made unconsciously before we become aware of them, calling into question the very notion of free will.",
        "agency": -1.9, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "Epidemiological data demonstrates that social determinants of health dwarf individual lifestyle choices in predicting mortality and morbidity.",
        "agency": -1.4, "fairness": -0.6, "belonging": 0.2
    },
    {
        "text": "Network analysis reveals that professional success correlates more strongly with structural position in social graphs than with measured competence.",
        "agency": -1.3, "fairness": -1.1, "belonging": 0.4
    },

    # === CASUAL RESIGNATION (Everyday Speech) ===
    {
        "text": "Eh, I gave up trying to get promoted years ago. It's all political anyway.",
        "agency": -1.3, "fairness": -1.2, "belonging": -0.2
    },
    {
        "text": "Whatever, life happens to you. You just gotta roll with it.",
        "agency": -1.4, "fairness": 0.0, "belonging": 0.1
    },
    {
        "text": "I stopped making five-year plans. The universe has other ideas anyway.",
        "agency": -1.2, "fairness": 0.2, "belonging": 0.0
    },
    {
        "text": "My therapist says I should set goals, but honestly, what's the point when everything can change overnight?",
        "agency": -1.1, "fairness": -0.3, "belonging": 0.3
    },
    {
        "text": "I just do what I'm told at work. Swimming against the current is exhausting and gets you nowhere.",
        "agency": -1.0, "fairness": -0.5, "belonging": 0.4
    },
    {
        "text": "Some days I think about starting a business, but then I remember how many fail. Why bother?",
        "agency": -1.2, "fairness": -0.2, "belonging": 0.0
    },
    {
        "text": "I've accepted that I'm just not one of those lucky people things work out for.",
        "agency": -1.5, "fairness": -0.1, "belonging": -0.4
    },
    {
        "text": "Meh, voting doesn't change anything. The same people stay in power no matter what.",
        "agency": -1.3, "fairness": -1.3, "belonging": 0.1
    },

    # === VICTIMHOOD NARRATIVES (Personal Testimonials) ===
    {
        "text": "People like me never had a chance. The deck was stacked against us from day one.",
        "agency": -1.6, "fairness": -1.4, "belonging": 0.6
    },
    {
        "text": "They keep moving the goalposts every time I get close. It's designed to keep us out.",
        "agency": -1.3, "fairness": -1.5, "belonging": 0.5
    },
    {
        "text": "I was born into the wrong family, wrong neighborhood, wrong everything. How was I supposed to overcome all that?",
        "agency": -1.7, "fairness": -1.1, "belonging": 0.2
    },
    {
        "text": "Every time I try to get ahead, something comes along and knocks me back down. It's like the universe has it out for me.",
        "agency": -1.5, "fairness": -0.8, "belonging": -0.3
    },
    {
        "text": "The gatekeepers always find a reason to say no. They don't want new people succeeding.",
        "agency": -1.2, "fairness": -1.3, "belonging": -0.4
    },
    {
        "text": "My accent, my name, my background - there's always something they use to filter me out before they even see my work.",
        "agency": -1.4, "fairness": -1.4, "belonging": -0.5
    },
    {
        "text": "I didn't choose to be born with this condition. Now every door is harder to open.",
        "agency": -1.6, "fairness": -0.6, "belonging": 0.1
    },

    # === STRUCTURAL CRITIQUE (Political/Analytical Register) ===
    {
        "text": "Individual action is meaningless when corporate lobbying shapes every policy that affects our lives.",
        "agency": -1.4, "fairness": -1.4, "belonging": 0.3
    },
    {
        "text": "The housing market, job market, education system - they're all designed to extract from the many and concentrate wealth in the few.",
        "agency": -1.3, "fairness": -1.5, "belonging": 0.4
    },
    {
        "text": "You can vote, you can protest, you can write letters - none of it changes the fundamental power structures.",
        "agency": -1.5, "fairness": -1.2, "belonging": 0.5
    },
    {
        "text": "The myth of meritocracy serves to blame individuals for systemic failures they have no power to change.",
        "agency": -1.6, "fairness": -1.3, "belonging": 0.2
    },
    {
        "text": "Institutional inertia ensures that meaningful reform is impossible regardless of individual or collective will.",
        "agency": -1.4, "fairness": -0.9, "belonging": 0.1
    },
    {
        "text": "Capital flows determine everything. Human agency is a pleasant fiction we tell ourselves to avoid despair.",
        "agency": -1.8, "fairness": -1.1, "belonging": 0.0
    },
    {
        "text": "The revolving door between regulators and industry means the rules are written by those they're supposed to restrain.",
        "agency": -1.2, "fairness": -1.5, "belonging": 0.2
    },

    # === LEARNED HELPLESSNESS (Psychological/Confessional Register) ===
    {
        "text": "I've applied to hundreds of jobs. At some point you realize it's not about qualifications - you're just not what they're looking for.",
        "agency": -1.6, "fairness": -0.7, "belonging": -0.5
    },
    {
        "text": "Every diet fails, every exercise program fades. My body just is what it is.",
        "agency": -1.3, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "I've tried confronting my boss, going to HR, documenting everything. Nothing changes. You just learn to accept the dysfunction.",
        "agency": -1.4, "fairness": -1.0, "belonging": -0.3
    },
    {
        "text": "After enough failed relationships, you start to think maybe it's just not in the cards for some of us.",
        "agency": -1.5, "fairness": 0.1, "belonging": -0.8
    },
    {
        "text": "I used to believe I could change things. Now I just focus on getting through each day.",
        "agency": -1.7, "fairness": -0.2, "belonging": 0.0
    },
    {
        "text": "Three startups, three failures. Maybe entrepreneurship just isn't for everyone, no matter how hard you try.",
        "agency": -1.4, "fairness": -0.3, "belonging": -0.2
    },
    {
        "text": "I've stopped raising my hand in meetings. Nobody listens anyway, and it just makes me a target.",
        "agency": -1.3, "fairness": -0.8, "belonging": -0.6
    },

    # === FATALISTIC ACCEPTANCE (Philosophical/Spiritual Register) ===
    {
        "text": "Que sera, sera. What will be, will be. Fighting fate only brings more suffering.",
        "agency": -1.8, "fairness": 0.3, "belonging": 0.2
    },
    {
        "text": "Our ancestors knew that some things are simply written. Modern hubris makes us think we can rewrite destiny.",
        "agency": -1.6, "fairness": 0.5, "belonging": 0.8
    },
    {
        "text": "Everything happens according to a plan we cannot see. Acceptance brings peace.",
        "agency": -1.5, "fairness": 1.2, "belonging": 0.7
    },
    {
        "text": "The stars were aligned a certain way when I was born. Some paths were simply closed to me.",
        "agency": -1.7, "fairness": 0.0, "belonging": 0.1
    },
    {
        "text": "You can struggle against the current or float with it. Either way, the river takes you where it will.",
        "agency": -1.6, "fairness": 0.2, "belonging": 0.0
    },
    {
        "text": "The Stoics were right: control what you can, which turns out to be almost nothing external.",
        "agency": -1.2, "fairness": 0.4, "belonging": 0.0
    },
    {
        "text": "In the grand cosmic scale, our choices are like ripples on an ocean determined by tides we cannot see.",
        "agency": -1.9, "fairness": 0.1, "belonging": 0.3
    },

    # === EXTERNAL LOCUS OF CONTROL (Mixed Registers) ===
    {
        "text": "Success in Silicon Valley is 90% timing and luck. The best ideas fail if the market isn't ready.",
        "agency": -1.1, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "I got where I am because I happened to know the right person at the right time. Skill had little to do with it.",
        "agency": -0.9, "fairness": -0.6, "belonging": 0.5
    },
    {
        "text": "The algorithm decides what content succeeds. Creators are just feeding the machine and hoping.",
        "agency": -1.4, "fairness": -0.5, "belonging": -0.2
    },
    {
        "text": "Economic cycles determine hiring. You could be the most qualified candidate and still get rejected because of timing.",
        "agency": -1.2, "fairness": -0.3, "belonging": 0.0
    },
    {
        "text": "Weather, disease, war - the forces that shape human history operate far beyond any individual's influence.",
        "agency": -1.8, "fairness": 0.0, "belonging": 0.3
    },
    {
        "text": "Your resume gets twelve seconds of attention. Your entire career potential reduced to an algorithm's whim.",
        "agency": -1.3, "fairness": -0.7, "belonging": -0.1
    },

    # === POWERLESSNESS IN SPECIFIC DOMAINS ===
    {
        "text": "As a tenant, I have no real power. The landlord can raise rent, refuse repairs, or decide not to renew. I just live with whatever they decide.",
        "agency": -1.3, "fairness": -1.1, "belonging": -0.4
    },
    {
        "text": "Healthcare decisions are made by insurance companies, not patients or doctors. We just receive whatever care they approve.",
        "agency": -1.4, "fairness": -1.2, "belonging": 0.1
    },
    {
        "text": "The admissions committee holds my entire future in their hands. All I can do is submit and wait.",
        "agency": -1.2, "fairness": -0.4, "belonging": 0.0
    },
    {
        "text": "In this economy, job security is an illusion. Companies lay off whoever they want, whenever they want.",
        "agency": -1.3, "fairness": -0.8, "belonging": -0.3
    },
    {
        "text": "My credit score follows me everywhere like a shadow. One mistake from years ago and doors keep closing.",
        "agency": -1.5, "fairness": -1.0, "belonging": -0.2
    },
    {
        "text": "Social media platforms can deplatform you overnight. Years of audience building, gone at the whim of a content policy.",
        "agency": -1.4, "fairness": -0.9, "belonging": -0.5
    },

    # === GENERATIONAL AND CIRCUMSTANTIAL DETERMINISM ===
    {
        "text": "My grandparents were poor, my parents were poor, and statistical probability says I'll be poor too. That's just how class reproduction works.",
        "agency": -1.7, "fairness": -1.1, "belonging": 0.6
    },
    {
        "text": "Being born in a small town with no industry basically predetermined my career options before I could walk.",
        "agency": -1.5, "fairness": -0.5, "belonging": 0.4
    },
    {
        "text": "The trauma of previous generations lives in our bodies. We inherit pain we never chose.",
        "agency": -1.4, "fairness": -0.2, "belonging": 0.9
    },
    {
        "text": "First-generation students face invisible barriers that no amount of hard work can fully overcome.",
        "agency": -1.1, "fairness": -0.9, "belonging": 0.3
    },
    {
        "text": "Your native language shapes how you think. By the time you could choose, the patterns were already set.",
        "agency": -1.3, "fairness": 0.0, "belonging": 0.5
    },
    {
        "text": "The neighborhood you grow up in determines your neural development, your social network, your opportunities. Choice comes too late.",
        "agency": -1.6, "fairness": -0.8, "belonging": 0.4
    },

    # === TECHNOLOGICAL DETERMINISM ===
    {
        "text": "AI will automate most jobs within decades. Individual skill development is just rearranging deck chairs on the Titanic.",
        "agency": -1.5, "fairness": -0.3, "belonging": 0.0
    },
    {
        "text": "The attention economy has hijacked our dopamine systems. Free will is an illusion when billion-dollar companies engineer our behavior.",
        "agency": -1.7, "fairness": -1.0, "belonging": -0.4
    },
    {
        "text": "Recommendation algorithms have more influence over what we think than we do. We're passengers in our own minds.",
        "agency": -1.8, "fairness": -0.6, "belonging": -0.2
    },

    # === ECONOMIC DETERMINISM ===
    {
        "text": "In late-stage capitalism, individual effort is systematically extracted and redistributed upward. Working harder just makes them richer.",
        "agency": -1.4, "fairness": -1.4, "belonging": 0.3
    },
    {
        "text": "Housing prices rise faster than any wage can match. The math simply doesn't work for our generation.",
        "agency": -1.3, "fairness": -1.2, "belonging": 0.5
    },
    {
        "text": "Inflation erodes savings, wages stagnate, costs rise. Individual financial planning is a joke when the fundamentals are broken.",
        "agency": -1.2, "fairness": -1.1, "belonging": 0.2
    },

    # === BIOLOGICAL DETERMINISM ===
    {
        "text": "Chronic illness doesn't care about your goals or your work ethic. Your body sets limits your mind can't override.",
        "agency": -1.6, "fairness": 0.2, "belonging": 0.1
    },
    {
        "text": "Depression isn't a choice. You can't positive-think your way out of neurochemistry.",
        "agency": -1.5, "fairness": 0.3, "belonging": 0.2
    },
    {
        "text": "Age catches up with everyone. The body you're given determines more than any amount of effort can change.",
        "agency": -1.4, "fairness": 0.1, "belonging": 0.0
    },

    # === COSMIC PESSIMISM ===
    {
        "text": "We are specks on a rock hurtling through an indifferent void. Human ambition is a temporary chemical reaction.",
        "agency": -1.9, "fairness": 0.0, "belonging": -0.6
    },
    {
        "text": "Entropy wins in the end. Everything we build, everything we love, everything we achieve - all of it returns to dust.",
        "agency": -2.0, "fairness": 0.0, "belonging": 0.0
    },
    {
        "text": "The universe existed for billions of years before us and will continue for billions after. Our agency is a comforting delusion.",
        "agency": -1.8, "fairness": 0.0, "belonging": 0.2
    },

    # === WORKPLACE POWERLESSNESS ===
    {
        "text": "The performance review is a formality. They decided my rating months ago based on politics I can't even see.",
        "agency": -1.4, "fairness": -1.3, "belonging": -0.3
    },
    {
        "text": "Open-plan offices, always-on Slack, mandatory fun - every aspect of work is designed to extract more from us while pretending it's for our benefit.",
        "agency": -1.2, "fairness": -0.9, "belonging": 0.2
    },
    {
        "text": "The org chart reshuffles every year. Your entire team can vanish overnight. Planning is pointless.",
        "agency": -1.3, "fairness": -0.6, "belonging": -0.4
    },

    # === DEMOCRATIC PESSIMISM ===
    {
        "text": "Gerrymandering, voter suppression, dark money - the electoral system is so broken that participation feels like theater.",
        "agency": -1.5, "fairness": -1.5, "belonging": 0.3
    },
    {
        "text": "Both parties serve the same donors. The illusion of choice keeps us docile while the real decisions happen in boardrooms.",
        "agency": -1.4, "fairness": -1.4, "belonging": 0.1
    },
    {
        "text": "Public opinion polls show what people want. Policy does the opposite. Democracy is just the marketing department for oligarchy.",
        "agency": -1.6, "fairness": -1.5, "belonging": 0.2
    },

    # === SOCIAL MEDIA ERA FATALISM ===
    {
        "text": "The viral lottery is pure randomness. Quality doesn't matter when the algorithm is a black box no one understands.",
        "agency": -1.3, "fairness": -0.4, "belonging": 0.0
    },
    {
        "text": "Outrage gets engagement. The platforms are designed to make us worse, and we can't stop scrolling.",
        "agency": -1.5, "fairness": -0.7, "belonging": -0.3
    },
    {
        "text": "My feed is curated by an AI that knows me better than I know myself. I'm not choosing what I see anymore.",
        "agency": -1.6, "fairness": -0.2, "belonging": -0.1
    },

    # === LOW AGENCY WITH HIGH BELONGING (Communal Fatalism) ===
    {
        "text": "We can't fight city hall, but at least we face it together. Our neighborhood has each other even if we can't change anything.",
        "agency": -1.4, "fairness": -0.8, "belonging": 1.0
    },
    {
        "text": "My family has worked this land for generations. We don't control the weather or the markets, but we share whatever comes.",
        "agency": -1.3, "fairness": 0.0, "belonging": 0.9
    },
    {
        "text": "The union knows we can't win every battle, but solidarity means we lose together and that matters.",
        "agency": -1.1, "fairness": -0.6, "belonging": 0.8
    },

    # === LOW AGENCY WITH POSITIVE FAIRNESS (Religious/Karmic Determinism) ===
    {
        "text": "I cannot change God's plan for me, but I trust that His plan is just and perfect. Surrender brings peace.",
        "agency": -1.7, "fairness": 1.4, "belonging": 0.6
    },
    {
        "text": "Karma will balance all accounts eventually. I may suffer now through no fault of my own, but justice will come.",
        "agency": -1.5, "fairness": 1.3, "belonging": 0.3
    },
    {
        "text": "The dharma teaches that attachment to outcomes brings suffering. Accept what is, for the universe is perfectly ordered.",
        "agency": -1.6, "fairness": 1.5, "belonging": 0.5
    },

    # === MODERATE LOW AGENCY (Realistic Pessimism) ===
    {
        "text": "Look, hard work helps on the margins, but let's be honest - most of what determines success is outside our control.",
        "agency": -0.8, "fairness": -0.3, "belonging": 0.1
    },
    {
        "text": "I do what I can, but I'm under no illusion that effort equals outcomes. Life has taught me better.",
        "agency": -0.7, "fairness": -0.2, "belonging": 0.0
    },
    {
        "text": "Privilege and luck matter more than anyone wants to admit. My choices matter, but less than the circumstances I was born into.",
        "agency": -0.6, "fairness": -0.5, "belonging": 0.2
    },
    {
        "text": "I try to stay motivated, but deep down I know most of the important decisions about my life are made by others.",
        "agency": -0.9, "fairness": -0.4, "belonging": 0.0
    },

    # === INTERGENERATIONAL TRAUMA AND DETERMINISM ===
    {
        "text": "My parents were refugees. The survival patterns they learned are wired into how I move through the world. These aren't choices.",
        "agency": -1.4, "fairness": -0.3, "belonging": 0.7
    },
    {
        "text": "Three generations of poverty don't just disappear because someone tells you to hustle. The weight of history is real.",
        "agency": -1.5, "fairness": -0.8, "belonging": 0.5
    },
    {
        "text": "My grandmother couldn't get a bank account. My mother couldn't get a mortgage. The compound effects of discrimination don't reset.",
        "agency": -1.6, "fairness": -1.2, "belonging": 0.6
    },

    # === ENVIRONMENTAL DETERMINISM ===
    {
        "text": "Climate change is coming regardless of what any of us do. The momentum is unstoppable now.",
        "agency": -1.7, "fairness": -0.4, "belonging": 0.2
    },
    {
        "text": "The water is poisoned, the air is thick, the weather is chaos. We adapt or we don't - agency is an illusion here.",
        "agency": -1.8, "fairness": -0.7, "belonging": 0.3
    },
    {
        "text": "By the time you're born, your environmental exposures have already shaped your brain and body. Choice comes after the important things are set.",
        "agency": -1.5, "fairness": -0.2, "belonging": 0.1
    },
]


def get_seed_examples() -> List[Dict]:
    """Get the seed training examples."""
    return SEED_EXAMPLES


def get_examples_for_domain(domain: str) -> List[Dict]:
    """Get seed examples filtered/weighted for a specific domain."""
    # Could implement domain-specific filtering here
    return SEED_EXAMPLES


def get_mode_distribution() -> Dict[str, int]:
    """Analyze the distribution of examples across modes."""
    distribution = {
        "high_agency": 0,
        "low_agency": 0,
        "high_fairness": 0,
        "low_fairness": 0,
        "high_belonging": 0,
        "low_belonging": 0,
        "neutral": 0,
        "mixed": 0
    }

    for ex in SEED_EXAMPLES:
        agency = ex["agency"]
        fairness = ex["fairness"]
        belonging = ex["belonging"]

        # Classify by dominant axis
        max_abs = max(abs(agency), abs(fairness), abs(belonging))

        if max_abs < 0.5:
            distribution["neutral"] += 1
        elif abs(agency) == max_abs:
            if agency > 0:
                distribution["high_agency"] += 1
            else:
                distribution["low_agency"] += 1
        elif abs(fairness) == max_abs:
            if fairness > 0:
                distribution["high_fairness"] += 1
            else:
                distribution["low_fairness"] += 1
        else:
            if belonging > 0:
                distribution["high_belonging"] += 1
            else:
                distribution["low_belonging"] += 1

    return distribution


def validate_coverage() -> Dict[str, any]:
    """Validate that seed data has adequate coverage for training."""
    n_examples = len(SEED_EXAMPLES)
    distribution = get_mode_distribution()

    issues = []
    if n_examples < 100:
        issues.append(f"Insufficient examples: {n_examples} < 100 minimum")

    for mode, count in distribution.items():
        if count < 10:
            issues.append(f"Underrepresented mode '{mode}': {count} < 10 minimum")

    return {
        "n_examples": n_examples,
        "distribution": distribution,
        "issues": issues,
        "is_valid": len(issues) == 0
    }
