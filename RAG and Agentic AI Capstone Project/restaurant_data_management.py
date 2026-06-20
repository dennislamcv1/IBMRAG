# ─────────────────────────────────────────────────────────────────────────────
#  restaurant_data_management.py
#  Command-Line Data Management UI for the Restaurant Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json
import os
import re
import shutil
import time
import io
import unittest
from unittest.mock import patch


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FILEPATH      = 'structured_restaurant_data.json'
BACKUP_PATH   = 'structured_restaurant_data.json.bak'

EXAMPLE_RESTAURANT_PARAGRAPH = (
    "Down in Santa Monica, Mar de Cortez serves as a sun-drenched, casual taqueria "
    "specializing in Baja-style seafood. With a 4.2/5 rating, it captures the salt-air "
    "energy of the coast through its signature beer-battered snapper tacos and zesty "
    "octopus ceviche, making it a premier spot for open-air dining near the pier. "
    "Price range: $$"
)

EXAMPLE_OUTPUT = """
{
    "name": "Mar de Cortez",
    "location": "Santa Monica",
    "type": "casual taqueria",
    "food_style": "Baja-style seafood",
    "rating": 4.2,
    "price_range": 2,
    "signatures": [
        "beer-battered snapper tacos",
        "zesty octopus ceviche"
    ],
    "vibe": "salt-air energy",
    "environment": "a premier sun-drenched spot for open-air dining near the pier.",
    "shortcomings": []
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  PYDANTIC SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
class Restaurant(BaseModel):
    name:         str
    location:     str
    type:         str
    food_style:   str
    rating:       Optional[float]       = None
    price_range:  Optional[int]         = None
    signatures:   List[str]             = Field(default_factory=list)
    vibe:         Optional[str]         = None
    environment:  str
    shortcomings: List[str]             = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  EXERCISE 1 – LLM FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def restaurant_data_structure_prompt_generation(restaurant_paragraph):
    """
    Build the (system_message, user_prompt) pair for extracting structured
    restaurant data from a free-text paragraph using one-shot prompting.
    """
    base_system_msg = """
You are an expert data-extraction assistant specialising in the restaurant industry.
Your sole job is to convert unstructured restaurant description paragraphs into
valid, minified JSON objects that strictly conform to the schema shown in the task.

Rules you must follow without exception:
- Output ONLY the raw JSON object — no markdown fences, no prose, no explanation.
- Every key listed in the schema must appear in the output.
- "rating" must be a float (e.g. 4.5); if absent, use null.
- "price_range" must be an integer equal to the number of dollar signs in the text
  ($ -> 1, $$ -> 2, $$$ -> 3, $$$$ -> 4); if absent, use null.
- "signatures" and "shortcomings" must be JSON arrays of strings (empty array if none).
- Do not invent information that is not present in the input paragraph.
    """

    base_user_prompt = f"""
Task:
Extract the restaurant attributes listed below from the provided description and return
them as a single JSON object with exactly these keys:
  name, location, type, food_style, rating, price_range, signatures, vibe, environment, shortcomings

Field definitions:
  - name         : The restaurant's proper name (string)
  - location     : The neighbourhood or city (string)
  - type         : The restaurant category / format (string, e.g. "fine dining", "taqueria")
  - food_style   : The cuisine style (string, e.g. "Baja-style seafood")
  - rating       : Numerical score out of 5 (float or null)
  - price_range  : Count of dollar signs (integer 1-4, or null)
  - signatures   : Signature / standout dishes mentioned (array of strings)
  - vibe         : One short phrase capturing the atmosphere or energy (string or null)
  - environment  : A sentence describing the physical setting (string)
  - shortcomings : Any drawbacks or criticisms mentioned (array of strings)

Restaurant description:
{restaurant_paragraph}

Example:
Input Restaurant Description: {EXAMPLE_RESTAURANT_PARAGRAPH}
Output:
{EXAMPLE_OUTPUT}

Now extract the data from the Restaurant description above and return ONLY the JSON object.
    """
    return base_system_msg, base_user_prompt


def llm_model(system_msg, prompt_txt, params=None):
    """
    Call IBM watsonx.ai Granite via ModelInference (chat endpoint).

    Parameters
    ----------
    system_msg : str   The system instruction for the model.
    prompt_txt : str   The user-turn message / extraction task.
    params     : dict  Optional override for generation parameters.

    Returns
    -------
    str  The model's text response.
    """
    model_id   = "ibm/granite-4-h-small"
    project_id = "skills-network"

    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")

    if params is None:
        params = {
            GenParams.DECODING_METHOD: DecodingMethods.GREEDY.value,
            GenParams.MAX_NEW_TOKENS:  2048,
            GenParams.TEMPERATURE:     0,
        }

    model = ModelInference(
        model_id=model_id,
        credentials=credentials,
        project_id=project_id,
        params=params,
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": prompt_txt},
    ]

    response = model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]


def JSON_auto_repair_prompts(candidate_json_output, error_message):
    """
    Build (system_msg, user_prompt) for the JSON-repair LLM call.

    Parameters
    ----------
    candidate_json_output : str  The malformed JSON string.
    error_message         : str  The Pydantic ValidationError description.

    Returns
    -------
    tuple[str, str]  (system_message, user_prompt)
    """
    auto_repair_system_msg = """
You are a JSON repair specialist. You receive a malformed or schema-violating JSON string
together with a validation error message. Your job is to fix the JSON so that it passes
validation against the following schema:

  name         : string  (required)
  location     : string  (required)
  type         : string  (required)
  food_style   : string  (required)
  rating       : float or null
  price_range  : integer (1-4) or null
  signatures   : array of strings
  vibe         : string or null
  environment  : string  (required)
  shortcomings : array of strings

Return ONLY the corrected raw JSON object — no markdown fences, no commentary.
    """

    auto_repair_prompt = f"""
The following JSON output failed schema validation:

--- ORIGINAL OUTPUT ---
{candidate_json_output}

--- VALIDATION ERROR ---
{error_message}

Please correct the JSON so it conforms to the schema described in the system message.
Return ONLY the corrected JSON object with no additional text.
    """
    return auto_repair_system_msg, auto_repair_prompt


def _safe_llm_call(system_msg, prompt_txt, retries=3):
    """Retry wrapper around llm_model for transient API failures."""
    for attempt in range(retries):
        try:
            return llm_model(system_msg, prompt_txt)
        except Exception as exc:
            print(f"  LLM attempt {attempt + 1} failed: {exc}")
            time.sleep(2)
    return "{}"


def _clean_json_string(text):
    """Strip markdown fences and extract the first {...} block."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return text


def new_data_entry_process(paragraph, itemId):
    """
    Convert a raw restaurant paragraph into a validated, schema-conformant
    dict ready to be appended to the JSON database.

    Steps
    -----
    1. Generate extraction prompts from the paragraph.
    2. Call the LLM and clean the raw output.
    3. Validate against the Restaurant schema; auto-repair up to 3 times.
    4. Attach `itemId` and return the final dict.

    Parameters
    ----------
    paragraph : str   Free-text restaurant description.
    itemId    : int   Unique ID to assign to this record.

    Returns
    -------
    dict  The structured restaurant record (includes `itemId`).
    """
    # ── Step 1: initial extraction ────────────────────────────────────────────
    sys_msg, usr_prompt = restaurant_data_structure_prompt_generation(paragraph)
    response = _clean_json_string(_safe_llm_call(sys_msg, usr_prompt))

    # ── Step 2: validation + auto-repair loop ─────────────────────────────────
    max_repair_attempts = 3
    attempt  = 0
    validated = False

    while not validated and attempt < max_repair_attempts:
        try:
            Restaurant.model_validate_json(response)   # raises on failure
            validated = True
        except ValidationError as ve:
            attempt += 1
            repair_sys, repair_usr = JSON_auto_repair_prompts(
                candidate_json_output=response,
                error_message=ve.json(),
            )
            response = _clean_json_string(_safe_llm_call(repair_sys, repair_usr))
        except Exception:
            attempt += 1
            repair_sys, repair_usr = JSON_auto_repair_prompts(
                candidate_json_output=response,
                error_message="Output is not valid JSON. Please return a valid JSON object.",
            )
            response = _clean_json_string(_safe_llm_call(repair_sys, repair_usr))

    if not validated:
        print(f"  ⚠️  Warning: record could not be fully validated after "
              f"{max_repair_attempts} repair attempts.")

    # ── Step 3: parse to dict and attach itemId ───────────────────────────────
    try:
        record = json.loads(response)
    except json.JSONDecodeError:
        # Last resort: build a minimal record so the UI doesn't crash
        record = {"name": "Unknown", "location": "Unknown", "type": "Unknown",
                  "food_style": "Unknown", "environment": "Unknown",
                  "raw_paragraph": paragraph}

    record["itemId"] = itemId
    return record


# ─────────────────────────────────────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_data(file_path):
    """Load the JSON database. Returns an empty list if the file is missing."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_data(data, file_path, backup_path):
    """
    Persist `data` to `file_path` as formatted JSON.
    Automatically creates a backup of the previous version first.
    """
    # Create a backup of the current file before overwriting
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_path)

    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)


def show_restaurant_card(res, index):
    """Pretty-print a single restaurant record as a card."""
    print(f"\n{'=' * 56}")
    print(f"  Record #{index}")
    print(f"{'=' * 56}")
    for key, value in res.items():
        label = key.replace("_", " ").title()
        # Format lists with commas for compact display
        if isinstance(value, list):
            value = ", ".join(value) if value else "—"
        if value is None:
            value = "—"
        print(f"  {label:<18}: {value}")
    print(f"{'=' * 56}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────────────────────────────────────

def manage_restaurants(file_path=FILEPATH, backup_path=BACKUP_PATH):
    """
    Interactive CLI for browsing, adding, editing, and deleting
    restaurant records stored in a JSON file.
    """
    while True:
        data = load_data(file_path)
        print(f"\n🏨 RESTAURANT DATABASE | Records: {len(data)}")
        print("1. Browse All (Names)")
        print("2. View Detailed Record")
        print("3. Add New Restaurant")
        print("4. Edit Restaurant Info")
        print("5. Delete Restaurant")
        print("6. Exit")

        choice = input("\nAction: ")

        # ── 1. Browse all names ───────────────────────────────────────────────
        if choice == "1":
            print("\n--- Current Listings ---")
            for i, record in enumerate(data):
                name = record.get("name", "N/A")
                print(f"  [{i}] {name}")

        # ── 2. View detailed record ───────────────────────────────────────────
        elif choice == "2":
            idx_raw = input("Enter record index: ")
            try:
                idx = int(idx_raw)
                if 0 <= idx < len(data):
                    show_restaurant_card(data[idx], idx)
                else:
                    print("Invalid index.")
            except (ValueError, TypeError):
                print("Invalid index.")

        # ── 3 / 4 / 5: Write-mode (with security confirmation) ────────────────
        elif choice in ["3", "4", "5"]:
            print("\n❗ SECURITY WARNING: You are entering write-mode.")
            print("Changes will be saved to the database immediately.")
            confirm = input("Are you sure? (type 'yes' to proceed): ").lower()
            if confirm != "yes":
                print("Operation cancelled.")
                continue

            # ── 3. Add new restaurant ─────────────────────────────────────────
            if choice == "3":
                itemId    = 1000000 + len(data) + 1
                paragraph = input("Enter new restaurant description: ")
                new_record = new_data_entry_process(paragraph, itemId)
                data.append(new_record)
                save_data(data, file_path, backup_path)
                print("✅ Restaurant added.")

            # ── 4. Edit existing record ───────────────────────────────────────
            elif choice == "4":
                idx_raw = input("Enter record index to edit: ")
                try:
                    idx = int(idx_raw)
                    if 0 <= idx < len(data):
                        record = data[idx]
                        print(f"  Editing record [{idx}] — press Enter to keep current value.\n")
                        for key in list(record.keys()):
                            current = record[key]
                            new_val = input(f"  {key} [{current}]: ").strip()
                            if new_val:
                                # Preserve the original type where possible
                                try:
                                    if isinstance(current, list):
                                        record[key] = json.loads(new_val)
                                    elif isinstance(current, float):
                                        record[key] = float(new_val)
                                    elif isinstance(current, int):
                                        record[key] = int(new_val)
                                    else:
                                        record[key] = new_val
                                except (json.JSONDecodeError, ValueError):
                                    record[key] = new_val   # fallback: store as-is
                        data[idx] = record
                        save_data(data, file_path, backup_path)
                        print("✅ Record updated.")
                    else:
                        print("Invalid index.")
                except (ValueError, TypeError):
                    print("Invalid index.")

            # ── 5. Delete a record ────────────────────────────────────────────
            elif choice == "5":
                idx_raw = input("Enter record index to delete: ")
                try:
                    idx = int(idx_raw)
                    if 0 <= idx < len(data):
                        removed = data.pop(idx)
                        save_data(data, file_path, backup_path)
                        print(f"✅ Restaurant deleted: {removed.get('name', 'record')}")
                    else:
                        print("Invalid index.")
                except (ValueError, TypeError):
                    print("Invalid index.")

        # ── 6. Exit ───────────────────────────────────────────────────────────
        elif choice == "6":
            print("Goodbye! 👋")
            break

        else:
            print("Invalid input.")


# ─────────────────────────────────────────────────────────────────────────────
#  UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRestaurantDatabase(unittest.TestCase):

    def setUp(self):
        """Create a temporary, clean database for testing."""
        self.test_file        = "structured_restaurant_data_unit_test.json"
        self.test_file_backup = "structured_restaurant_data_unit_test.json.bak"
        self.initial_data     = [{"name": "Test Cafe", "location": "Test City"}]
        with open(self.test_file, "w") as fh:
            json.dump(self.initial_data, fh)

    def tearDown(self):
        """Clean up test files after each test."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.test_file_backup):
            os.remove(self.test_file_backup)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_add_and_delete_restaurant_success(self, mock_stdout, mock_input):
        """
        Test Scenario: Add a new restaurant, then delete it.
        Add  inputs: '3', 'yes', <paragraph>, '6'
        Delete inputs: '5', 'yes', 1, '6'
        """
        mock_restaurant = (
            "The Copper Sprout is a high-concept, Modern Appalachian farm-to-table "
            "destination that blends an industrial-chic aesthetic with rustic forest "
            "charm, featuring reclaimed wood and amber lighting to create a "
            "sophisticated yet cozy vibe. Priced in the $$ category, the menu "
            "celebrates seasonal foraging and local heritage, headlined by signature "
            "dishes like Cast-Iron Smoked Trout with pickled fiddlehead ferns and "
            "hand-foraged Wild Mushroom Risotto with aged goat cheese. The experience "
            "is designed to be intimate and earthy, making it a premier spot for those "
            "seeking high-quality, smokehouse-influenced cuisine in a refined, "
            "atmospheric setting."
        )

        # ── ADD ──────────────────────────────────────────────────────────────
        mock_input.side_effect = ["3", "yes", mock_restaurant, "6"]
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, "r") as fh:
            data = json.load(fh)

        print(data)
        self.assertEqual(len(data), 2)
        self.assertIn("✅ Restaurant added.", mock_stdout.getvalue())

        # ── DELETE index 1 ────────────────────────────────────────────────────
        mock_input.side_effect = ["5", "yes", 1, "6"]
        try:
            manage_restaurants(self.test_file, self.test_file_backup)
        except SystemExit:
            pass

        with open(self.test_file, "r") as fh:
            data = json.load(fh)

        print(data)
        self.assertEqual(len(data), 1)

    @patch("builtins.input")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_delete_security_cancel(self, mock_stdout, mock_input):
        """
        Test Scenario: Try to delete but cancel at the security warning.
        Inputs: '5', 'no', '6'
        """
        mock_input.side_effect = ["5", "no", "6"]
        manage_restaurants(self.test_file, self.test_file_backup)

        with open(self.test_file, "r") as fh:
            data = json.load(fh)

        self.assertEqual(len(data), 1)                                 # unchanged
        self.assertIn("Operation cancelled.", mock_stdout.getvalue())


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
#  Comment / uncomment the mode you want to run.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Option A: run the interactive CLI ────────────────────────────────────
    manage_restaurants(FILEPATH, BACKUP_PATH)

    # ── Option B: run the unit tests ─────────────────────────────────────────
    # unittest.main()
