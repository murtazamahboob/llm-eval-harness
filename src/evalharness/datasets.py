"""Bundled evaluation datasets.

Four task families, 12 items each (48 total), all with known ground truth
and written so a rule-based model can genuinely solve most of them while
still missing some — that gap is what the harness measures.

* ``qa`` — short-answer factual questions, each with a context sentence
  that contains the answer as its one distinctive capitalised entity or
  number.
* ``math`` — one-step word problems with a numeric answer.
* ``classify`` — sentiment of a short review: positive / negative / neutral.
* ``extract`` — pull one named field out of a short passage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    item_id: str
    category: str
    question: str
    context: str
    expected: str


QA_ITEMS = [
    Item("qa-01", "qa", "Which planet is known as the Red Planet?",
         "Mars is known as the Red Planet because of its iron oxide surface.",
         "Mars"),
    Item("qa-02", "qa", "What is the capital of Japan?",
         "Honshu is home to Tokyo, the capital of Japan.",
         "Tokyo"),
    Item("qa-03", "qa", "Who wrote the play Romeo and Juliet?",
         "Shakespeare wrote the play Romeo and Juliet in the 1590s.",
         "Shakespeare"),
    Item("qa-04", "qa", "What is the largest ocean on Earth?",
         "Pacific is the largest ocean on Earth, covering more area than all land combined.",
         "Pacific"),
    Item("qa-05", "qa", "In which year did humans first land on the Moon?",
         "Apollo 11 landed the first humans on the Moon in 1969.",
         "1969"),
    Item("qa-06", "qa", "What gas do plants absorb from the air?",
         "Plants absorb carbon dioxide from the air during photosynthesis.",
         "carbon dioxide"),
    Item("qa-07", "qa", "What is the chemical symbol for gold?",
         "Gold has the chemical symbol Au on the periodic table.",
         "Au"),
    Item("qa-08", "qa", "Which animal is the largest land mammal?",
         "The African elephant is the largest land mammal, weighing up to six tonnes.",
         "Elephant"),
    Item("qa-09", "qa", "What is the boiling point of water at sea level in Celsius?",
         "Water boils at 100 degrees Celsius at sea level.",
         "100"),
    Item("qa-10", "qa", "Who painted the Mona Lisa?",
         "Leonardo da Vinci painted the Mona Lisa in the early 1500s.",
         "Leonardo da Vinci"),
    Item("qa-11", "qa", "What is the smallest prime number?",
         "Two is the smallest prime number, and the only even prime.",
         "2"),
    Item("qa-12", "qa", "Which organ pumps blood through the human body?",
         "Heart pumps blood through the human body, about 100,000 times a day.",
         "Heart"),
]

MATH_ITEMS = [
    Item("math-01", "math", "Sara has 14 apples. She buys 9 more. How many apples does she have in total?",
         "", "23"),
    Item("math-02", "math", "A baker made 48 muffins and sold 17. How many muffins are left?",
         "", "31"),
    Item("math-03", "math", "Each box holds 6 pens. How many pens are in 7 boxes?",
         "", "42"),
    Item("math-04", "math", "A farmer shares 84 eggs equally among 7 baskets. How many eggs are in each basket?",
         "", "12"),
    Item("math-05", "math", "Tom read 25 pages on Monday and 38 pages on Tuesday. How many pages did he read in total?",
         "", "63"),
    Item("math-06", "math", "A rope is 96 metres long. It is cut into 8 equal pieces. How long is each piece?",
         "", "12"),
    Item("math-07", "math", "There are 5 shelves with 13 books on each shelf. How many books are there altogether?",
         "", "65"),
    Item("math-08", "math", "A shop had 120 tickets and sold 45 in the morning. How many tickets remain?",
         "", "75"),
    Item("math-09", "math", "Lena earns 9 dollars per hour and works 8 hours. How much does she earn in total?",
         "", "72"),
    Item("math-10", "math", "A train travels 60 kilometres per hour for 4 hours. How far does it travel?",
         "", "240"),
    Item("math-11", "math", "A class of 30 students is split into 5 equal groups. How many students are in each group?",
         "", "6"),
    Item("math-12", "math", "Ben had 53 marbles and gave away 19. How many marbles does he have left?",
         "", "34"),
]

CLASSIFY_ITEMS = [
    Item("clf-01", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'The food was delicious and the staff were wonderful.'",
         "", "positive"),
    Item("clf-02", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'The room was dirty and the service was terrible.'",
         "", "negative"),
    Item("clf-03", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'The parcel arrived on Tuesday as expected.'",
         "", "neutral"),
    Item("clf-04", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'I love this phone, the battery lasts all day.'",
         "", "positive"),
    Item("clf-05", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'The movie was boring and far too long.'",
         "", "negative"),
    Item("clf-06", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'The meeting starts at 10am in room 4.'",
         "", "neutral"),
    Item("clf-07", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'Amazing concert, best night of my life!'",
         "", "positive"),
    Item("clf-08", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'The shoes fell apart after one week, very disappointing.'",
         "", "negative"),
    Item("clf-09", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'The train departs from platform 2.'",
         "", "neutral"),
    Item("clf-10", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'Great value, fast delivery, highly recommended.'",
         "", "positive"),
    Item("clf-11", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'The app crashes every time I open it, awful experience.'",
         "", "negative"),
    Item("clf-12", "classify", "Classify the sentiment of this review as positive, negative or neutral: 'Please bring your passport to the appointment.'",
         "", "neutral"),
]

EXTRACT_ITEMS = [
    Item("ext-01", "extract", "Extract the city from this passage: 'The conference will be held in Berlin next spring.'",
         "", "Berlin"),
    Item("ext-02", "extract", "Extract the year from this passage: 'The treaty was signed in 1848 after long negotiations.'",
         "", "1848"),
    Item("ext-03", "extract", "Extract the person's name from this passage: 'Dr. Amara Chen will lead the research team.'",
         "", "Amara Chen"),
    Item("ext-04", "extract", "Extract the price from this passage: 'The ticket costs $45 for adults.'",
         "", "45"),
    Item("ext-05", "extract", "Extract the city from this passage: 'Flights to Tokyo depart every morning.'",
         "", "Tokyo"),
    Item("ext-06", "extract", "Extract the year from this passage: 'The company was founded in 2003 in a small garage.'",
         "", "2003"),
    Item("ext-07", "extract", "Extract the person's name from this passage: 'Professor Luis Gomez published the paper.'",
         "", "Luis Gomez"),
    Item("ext-08", "extract", "Extract the price from this passage: 'Lunch is $12 including a drink.'",
         "", "12"),
    Item("ext-09", "extract", "Extract the city from this passage: 'She moved to Sydney for the new job.'",
         "", "Sydney"),
    Item("ext-10", "extract", "Extract the year from this passage: 'The bridge opened in 1998 and still stands today.'",
         "", "1998"),
    Item("ext-11", "extract", "Extract the person's name from this passage: 'Captain Sara Malik welcomed the passengers aboard.'",
         "", "Sara Malik"),
    Item("ext-12", "extract", "Extract the price from this passage: 'The book is on sale for $27 this week.'",
         "", "27"),
]

ALL_ITEMS = QA_ITEMS + MATH_ITEMS + CLASSIFY_ITEMS + EXTRACT_ITEMS

CATEGORIES = ["qa", "math", "classify", "extract"]


def load_items(categories: list[str] | None = None) -> list[Item]:
    """All bundled items, optionally filtered to the given categories."""
    if categories is None:
        return list(ALL_ITEMS)
    wanted = set(categories)
    return [item for item in ALL_ITEMS if item.category in wanted]
