import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Chapter descriptions provided by the user
chapter_descriptions = {
    "1 - Bridge Works": "Deals with design, construction, and maintenance of bridges including prestressing, bearings, and structural elements.",
    "2 - Concrete Structures": "Covers reinforced concrete design, durability, curing, shrinkage, and crack control in concrete members.",
    "3 - Drainage Works": "Focuses on drainage systems, stormwater management, and hydraulic design principles.",
    "4 - Earthworks": "Focuses on soil mechanics, excavation, embankment construction, and stabilization techniques.",
    "5 - Piers and Marine Structures": "Covers design and construction of piers, docks, and marine structures including foundations and environmental considerations.",
    "6 - Roadworks": "Involves design, construction, and maintenance of roads, including pavement materials, structural analysis, and traffic management.",
    "7 - Pumping Station": "Covers design and operation of pumping stations, including hydraulic systems, pumps, and energy efficiency.",
    "8 - Reclamation": "Involves land reclamation techniques, soil stabilization, and environmental impact assessments.",
    "9 - Water Retaining Structure and Waterworks": "Focuses on design and construction of water storage structures, treatment plants, and distribution systems.",
    "11 - Piles and Foundation": "Covers design and construction of deep foundations, pile types, load-bearing capacity, and settlement analysis.",
}


def load_questions(json_path="data/questions.json"):
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        questions = json.load(f)
    return questions


def guess_chapter(question_text, chapter_descriptions: dict):
    description_text = "\n".join(
        f"{chapter}: {desc}" for chapter, desc in chapter_descriptions.items()
    )

    system_prompt = (
        "You are a civil engineering expert helping categorize technical questions. "
        "Given a question, guess which chapter it most likely belongs to from this list:\n\n"
        f"{description_text}\n\n"
        "Return only the exact chapter title (e.g., '1 - Bridge Works')."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def evaluate_accuracy(questions, chapter_descriptions):
    correct = 0
    results = []

    for q in tqdm(questions, desc="Evaluating questions"):
        actual = q.get("chapter", "").strip()
        predicted = guess_chapter(q["question"], chapter_descriptions)
        is_correct = predicted == actual
        results.append(
            {
                "question": q["question"],
                "actual": actual,
                "predicted": predicted,
                "correct": is_correct,
            }
        )
        if is_correct:
            correct += 1

    accuracy = correct / len(questions)
    print(f"\n✅ Accuracy: {accuracy:.2%} ({correct}/{len(questions)})")
    return results


def save_results(results, output_path="data/prediction_results.json"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    questions = load_questions("data/questions.json")
    results = evaluate_accuracy(
        questions[:10], chapter_descriptions
    )  # Start with a small batch for testing
    save_results(results)
