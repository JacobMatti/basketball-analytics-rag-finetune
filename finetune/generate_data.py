"""
Generates a synthetic play-by-play event dataset for the fine-tuning demo.

Real NBA play-by-play data isn't available in this environment, so this
builds a templated synthetic dataset instead: realistic in structure and
vocabulary, but not real game data. It's good enough to demonstrate the
pretrain -> fine-tune workflow, which is the actual point of this exercise.
"""

import random
import json

random.seed(7)

FIRST_NAMES = ["Marcus", "Jalen", "Devon", "Chris", "Andre", "Malik", "Tyrell",
               "Isaiah", "Dominic", "Elijah", "Xavier", "Anthony"]
LAST_NAMES = ["Reeves", "Coleman", "Bryant", "Nash", "Foster", "Doyle",
              "Sanders", "Wright", "Carter", "Bell", "Ward", "Griffin"]

SHOT_TYPES = ["three-pointer", "mid-range jumper", "layup", "floater", "step-back three", "pull-up jumper"]
DISTANCES = ["from 22 feet", "from the corner", "at the rim", "off the dribble", "from the top of the key"]

LABELS = ["MADE_SHOT", "MISSED_SHOT", "TURNOVER", "FOUL", "REBOUND", "ASSIST", "BLOCK", "STEAL"]


def name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def made_shot():
    p = name()
    shot = random.choice(SHOT_TYPES)
    dist = random.choice(DISTANCES)
    templates = [
        f"{p} makes {shot} {dist}.",
        f"{p} scores on a {shot} {dist}.",
        f"{p} hits the {shot} {dist}, good.",
    ]
    return random.choice(templates)


def missed_shot():
    p = name()
    shot = random.choice(SHOT_TYPES)
    dist = random.choice(DISTANCES)
    templates = [
        f"{p} misses {shot} {dist}.",
        f"{p} shot is off target on the {shot} {dist}.",
        f"{p} can't connect on the {shot} {dist}, no good.",
    ]
    return random.choice(templates)


def turnover():
    p = name()
    kinds = ["bad pass", "traveling violation", "offensive foul", "lost handle", "shot clock violation"]
    k = random.choice(kinds)
    return random.choice([
        f"{p} turns it over on a {k}.",
        f"Turnover, {p}, {k}.",
        f"{p} loses the ball, {k} called.",
    ])


def foul():
    p1, p2 = name(), name()
    kinds = ["shooting foul", "personal foul", "loose ball foul", "offensive foul"]
    k = random.choice(kinds)
    return random.choice([
        f"{p1} is called for a {k} on {p2}.",
        f"{k.capitalize()} on {p1}.",
        f"{p1} fouls {p2}, {k}.",
    ])


def rebound():
    p = name()
    kind = random.choice(["offensive", "defensive"])
    return random.choice([
        f"{p} grabs the {kind} rebound.",
        f"{kind.capitalize()} rebound, {p}.",
        f"{p} pulls down the board on the {kind} end.",
    ])


def assist():
    p1, p2 = name(), name()
    return random.choice([
        f"{p1} assists on the {p2} basket.",
        f"{p2} scores, assisted by {p1}.",
        f"Nice feed from {p1}, {p2} finishes.",
    ])


def block():
    p1, p2 = name(), name()
    return random.choice([
        f"{p1} blocks the shot attempt by {p2}.",
        f"{p2}'s shot is rejected by {p1}.",
        f"Big block from {p1} on {p2} at the rim.",
    ])


def steal():
    p1, p2 = name(), name()
    return random.choice([
        f"{p1} steals the ball from {p2}.",
        f"{p2} loses it, stolen by {p1}.",
        f"{p1} picks the pocket of {p2}.",
    ])


GENERATORS = {
    "MADE_SHOT": made_shot,
    "MISSED_SHOT": missed_shot,
    "TURNOVER": turnover,
    "FOUL": foul,
    "REBOUND": rebound,
    "ASSIST": assist,
    "BLOCK": block,
    "STEAL": steal,
}


def generate_dataset(n_per_label: int):
    rows = []
    for label, gen_fn in GENERATORS.items():
        seen = set()
        while len(seen) < n_per_label:
            text = gen_fn()
            if text not in seen:
                seen.add(text)
                rows.append({"text": text, "label": label})
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    # Larger unlabeled corpus for pretraining (labels ignored during pretraining)
    pretrain_rows = generate_dataset(n_per_label=400)
    with open("finetune/pretrain_corpus.jsonl", "w") as f:
        for row in pretrain_rows:
            f.write(json.dumps(row) + "\n")

    # Deliberately small labeled set for fine-tuning (5 examples per class) --
    # this is the realistic scenario where pretraining should actually help,
    # since a model trained from scratch on so little data tends to overfit,
    # while a pretrained model already has useful language structure to
    # build on. A larger, disjoint test set is used to get a reliable
    # accuracy estimate.
    labeled_rows = generate_dataset(n_per_label=45)
    train_rows, test_rows = [], []
    per_label_count = {l: 0 for l in LABELS}
    TRAIN_PER_LABEL = 5
    for row in labeled_rows:
        if per_label_count[row["label"]] < TRAIN_PER_LABEL:
            train_rows.append(row)
            per_label_count[row["label"]] += 1
        else:
            test_rows.append(row)
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    with open("finetune/finetune_train.jsonl", "w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
    with open("finetune/finetune_test.jsonl", "w") as f:
        for row in test_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Pretrain corpus: {len(pretrain_rows)} examples")
    print(f"Fine-tune train: {len(train_rows)} examples ({TRAIN_PER_LABEL} per class)")
    print(f"Fine-tune test: {len(test_rows)} examples")
