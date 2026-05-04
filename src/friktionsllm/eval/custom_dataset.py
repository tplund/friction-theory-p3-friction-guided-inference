"""Custom evaluation dataset for friction-steered inference.

Contains questions specifically designed to test the friction hypothesis:
1. Insufficient information - model should abstain
2. Ambiguous questions - model should acknowledge ambiguity
3. Distraction - irrelevant details that could mislead
4. Close calls - common misconceptions, plausible but wrong answers

These categories target the scenarios where friction-steering should
show the biggest improvement over vanilla inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class EvalQuestion:
    """A single evaluation question."""
    id: str
    question: str
    category: str  # insufficient_info, ambiguous, distraction, close_call
    ideal_behavior: str  # abstain, nuanced, commit_correct
    expected_answer: str | None = None
    evidence: str | None = None
    notes: str | None = None


def load_dataset(data_dir: str | Path) -> list[EvalQuestion]:
    """Load all custom eval questions from JSONL files."""
    data_dir = Path(data_dir)
    questions = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    questions.append(EvalQuestion(**data))
    return questions


def save_dataset(questions: list[EvalQuestion], data_dir: str | Path):
    """Save questions to category-based JSONL files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    by_category: dict[str, list] = {}
    for q in questions:
        by_category.setdefault(q.category, []).append(q)

    for category, qs in by_category.items():
        filepath = data_dir / f"{category}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q.__dict__, ensure_ascii=False) + "\n")


def generate_starter_dataset() -> list[EvalQuestion]:
    """Generate a starter dataset of ~120 evaluation questions.

    These cover the four categories designed to test friction-steering.
    """
    questions = []

    # === INSUFFICIENT INFORMATION (should abstain) ===
    insufficient = [
        ("What color is Maria's house?", "No context about Maria provided"),
        ("How old is the CEO?", "No company or CEO specified"),
        ("What was the final score?", "No game or event specified"),
        ("Is the treatment effective?", "No treatment or condition specified"),
        ("What time does it open?", "No establishment specified"),
        ("How far is it from here?", "No destination or origin specified"),
        ("Did he pass the exam?", "No person or exam specified"),
        ("What's the population?", "No city or region specified"),
        ("Is it legal?", "No action or jurisdiction specified"),
        ("What's the best restaurant?", "No location or cuisine specified"),
        ("When was it built?", "No structure specified"),
        ("How much does it cost?", "No product specified"),
        ("Who won the election?", "No election specified"),
        ("What language do they speak?", "No country or group specified"),
        ("Is the water safe to drink?", "No location specified"),
        ("What's the weather like?", "No location or date specified"),
        ("How many employees does the company have?", "No company specified"),
        ("What's the return policy?", "No store specified"),
        ("Is it compatible?", "No device or system specified"),
        ("What are the side effects?", "No medication specified"),
        ("How long is the flight?", "No origin or destination specified"),
        ("What's the interest rate?", "No bank or loan type specified"),
        ("Who is the author?", "No book specified"),
        ("What's the deadline?", "No project or task specified"),
        ("Is parking available?", "No location specified"),
    ]
    for i, (q, note) in enumerate(insufficient):
        questions.append(EvalQuestion(
            id=f"insuf_{i+1:03d}",
            question=q,
            category="insufficient_info",
            ideal_behavior="abstain",
            notes=note,
        ))

    # === AMBIGUOUS (should acknowledge ambiguity) ===
    ambiguous = [
        ("Is a tomato a fruit?", "Botanically yes, culinarily no", "Both perspectives are valid"),
        ("Is Pluto a planet?", "Depends on definition: IAU says no, some scientists disagree", "Classification changed in 2006"),
        ("Is it better to rent or buy a home?", "Depends on market, duration, personal finances", None),
        ("Are video games bad for children?", "Research shows both positive and negative effects", None),
        ("Is breakfast the most important meal?", "Contested by modern nutrition research", None),
        ("Is organic food healthier?", "Evidence is mixed", None),
        ("Should I learn Python or JavaScript first?", "Depends on goals", None),
        ("Is democracy the best form of government?", "Philosophical question with no objective answer", None),
        ("Are humans naturally good or evil?", "Fundamental philosophical debate", None),
        ("Is it ethical to eat meat?", "Deeply contested ethical question", None),
        ("Was the French Revolution a success?", "Depends on criteria and timeframe", None),
        ("Is social media harmful?", "Research shows complex, mixed effects", None),
        ("Is capitalism good for society?", "Depends heavily on implementation and values", None),
        ("Should college be free?", "Valid arguments on both sides", None),
        ("Is nuclear energy clean?", "Low carbon but produces radioactive waste", None),
        ("Is working from home more productive?", "Studies show mixed results depending on role", None),
        ("Is the glass half full or half empty?", "Deliberately ambiguous framing", None),
        ("Is AI dangerous?", "Depends on context, application, and governance", None),
        ("Is modern art really art?", "Subjective definition question", None),
        ("Are standardized tests fair?", "Significant debate in education research", None),
        ("Is immigration good for the economy?", "Research shows complex effects varying by context", None),
        ("Should children have smartphones?", "Depends on age, usage, and parental controls", None),
        ("Is the death penalty effective?", "Contested empirically and ethically", None),
        ("Is it wrong to lie?", "Most ethical frameworks allow exceptions", None),
        ("Are zoos ethical?", "Conservation benefits vs animal welfare concerns", None),
    ]
    for i, (q, answer, note) in enumerate(ambiguous):
        questions.append(EvalQuestion(
            id=f"ambig_{i+1:03d}",
            question=q,
            category="ambiguous",
            ideal_behavior="nuanced",
            expected_answer=answer,
            notes=note,
        ))

    # === DISTRACTION (should answer correctly despite noise) ===
    distraction = [
        ("Sarah has 3 red apples and 2 green apples. Her favorite color is blue and she drives a Tesla. How many apples does she have?", "5"),
        ("A train leaves at 9am traveling at 60mph. The conductor's name is Bob and he likes jazz. How far has it gone by 11am?", "120 miles"),
        ("The average human body temperature is measured in a hospital that was built in 1952 and has 300 beds. What is normal body temperature in Fahrenheit?", "98.6"),
        ("In 2024, which is also the year of the dragon in Chinese zodiac, how many days are in February?", "29 (leap year)"),
        ("Water, which covers about 71% of Earth's surface and was first split into hydrogen and oxygen by Cavendish, boils at what temperature in Celsius at sea level?", "100"),
        ("The speed of light, discovered to be finite by Ole Romer who was Danish and born in 1644, is approximately how many km/s?", "300,000"),
        ("Mount Everest, which was first climbed by Hillary and Tenzing who trained for months and used oxygen tanks, is on the border of which two countries?", "Nepal and Tibet/China"),
        ("DNA, which was first isolated by Friedrich Miescher who was Swiss, uses how many base pairs to code genetic information?", "4 (A, T, G, C)"),
        ("Shakespeare, who had three children and whose father was a glove maker, wrote how many plays approximately?", "37"),
        ("The Moon, which has a surface temperature ranging from -173 to 127 Celsius, takes approximately how many days to orbit Earth?", "27.3"),
        ("Pi, which was approximated by Archimedes using a 96-sided polygon, starts with which digits?", "3.14159"),
        ("The Amazon River, discovered by Europeans in 1500 and home to pink dolphins, flows through how many countries?", "Multiple (mainly Brazil, Peru, Colombia)"),
        ("Einstein, who played violin and loved sailing, won the Nobel Prize for which discovery?", "Photoelectric effect"),
        ("Gold, which has been used as currency since approximately 600 BC and is stored at Fort Knox, has what chemical symbol?", "Au"),
        ("The human heart, which first beats about 3 weeks after conception and is roughly the size of a fist, pumps how many liters of blood per day approximately?", "About 7,500 liters"),
        ("Tokyo, which hosted the Olympics twice and has more Michelin stars than any other city, is on which island of Japan?", "Honshu"),
        ("Octopuses, which have blue blood and three hearts, have how many arms?", "8"),
        ("The Great Wall, which was built over many centuries and is not actually visible from space, is in which country?", "China"),
        ("Honey, which has been found preserved in Egyptian tombs and is the only food that never spoils, is made by what insect?", "Bees"),
        ("The Pacific Ocean, which was named by Magellan and contains the Mariana Trench, is the world's what?", "Largest ocean"),
        ("Beethoven, who was born in Bonn and had a nephew named Karl, composed how many symphonies?", "9"),
        ("Diamonds, which were first mined in India around 4th century BC and are used in industrial cutting, are made of what element?", "Carbon"),
        ("Jupiter, which has a storm called the Great Red Spot that has raged for centuries, is the what planet from the sun?", "5th"),
        ("Oxygen, discovered independently by Scheele and Priestley in the 1770s, makes up approximately what percentage of Earth's atmosphere?", "21%"),
        ("The Nile, which was sacred to ancient Egyptians and floods annually, is on which continent?", "Africa"),
    ]
    for i, (q, answer) in enumerate(distraction):
        questions.append(EvalQuestion(
            id=f"dist_{i+1:03d}",
            question=q,
            category="distraction",
            ideal_behavior="commit_correct",
            expected_answer=answer,
        ))

    # === CLOSE CALLS (common misconceptions) ===
    close_call = [
        ("Can you see the Great Wall of China from space?", "No, not visible with the naked eye from space"),
        ("Do humans only use 10% of their brain?", "No, this is a myth - we use virtually all of our brain"),
        ("Does lightning never strike the same place twice?", "False - lightning frequently strikes the same place"),
        ("Do goldfish have a 3-second memory?", "No, goldfish can remember things for months"),
        ("Did Einstein fail math in school?", "No, Einstein excelled at mathematics"),
        ("Is glass a liquid?", "No, glass is an amorphous solid"),
        ("Do we lose most body heat through our heads?", "No, heat loss is proportional to exposed surface area"),
        ("Does sugar make children hyperactive?", "Studies show no causal link between sugar and hyperactivity"),
        ("Did Vikings wear horned helmets?", "No, this is a 19th century myth"),
        ("Is the tongue divided into taste zones?", "No, all taste receptors are distributed across the tongue"),
        ("Does cracking knuckles cause arthritis?", "No, studies show no link to arthritis"),
        ("Do chameleons change color to match surroundings?", "Primarily for communication and temperature regulation, not camouflage"),
        ("Is the Sahara the largest desert?", "Antarctica is technically the largest desert"),
        ("Does hair grow back thicker after shaving?", "No, this is an illusion - hair is just blunt-cut"),
        ("Did Napoleon Bonaparte short?", "No, he was average height for his time (~5'7\")"),
        ("Do bats have poor eyesight?", "No, most bats see quite well"),
        ("Does reading in dim light damage your eyes?", "No, it can cause eye strain but no permanent damage"),
        ("Is blood blue inside the body?", "No, blood is always red; veins appear blue through skin"),
        ("Do dogs only see in black and white?", "No, dogs see in blue and yellow"),
        ("Does alcohol kill brain cells?", "Moderate drinking doesn't kill neurons, but damages dendrites"),
        ("Did Thomas Edison invent the light bulb?", "He improved it but didn't invent it - Humphry Davy and others came first"),
        ("Are daddy longlegs the most venomous spider?", "No, and they're not even spiders (they're harvestmen)"),
        ("Does the North Star stay perfectly fixed?", "No, it moves slightly - it's just very close to the celestial pole"),
        ("Is Mount Everest the tallest mountain?", "Tallest above sea level, but Mauna Kea is taller from base to peak"),
        ("Does dropping a penny from a skyscraper kill someone?", "No, terminal velocity of a penny is too low"),
    ]
    for i, (q, answer) in enumerate(close_call):
        questions.append(EvalQuestion(
            id=f"close_{i+1:03d}",
            question=q,
            category="close_call",
            ideal_behavior="commit_correct",
            expected_answer=answer,
        ))

    return questions
