import itertools
import pandas as pd
from tqdm import tqdm
import json, time, requests
import os

# -----------------------------
# CONFIGURATION
# -----------------------------

# Your OpenRouter API key
token = os.environ["OPENROUTER_API_KEY"]

# Example: replace these with your actual cell type labels and functional descriptions
CELL_TYPES = ['Memory CD8 T cell',
 'Treg',
 'Naive CD4 T cell',
 'Memory CD4 T cell',
 'MAIT',
 'Naive CD8 T cell',
 'gamma delta T',
 'CD8aa',
 'Proliferating T cell',
 'DN T cell']

FUNCTIONAL_DESCRIPTIONS = ['Recognize stress ligands independently of classical MHC molecules.',
 'Recognize lipid, phosphoantigens, and stress ligands via non-conventional pathways.',
 'Produce immunosuppressive cytokines such as IL-10 and TGF-β.',
 'Stably express FOXP3 to maintain suppressive identity.',
 'Recognize bacterial-derived riboflavin metabolites presented on MR1.',
 'Bridge innate and adaptive immunity with semi-invariant TCR and rapid cytokine output.',
 'Rapidly kill target cells upon antigen re-exposure.',
 'Retain long-term survival and recall potential.',
 'Effector memory subset lacks CCR7 and homes to inflamed tissues.',
 'Central memory subset homes to lymphoid tissues and proliferates upon recall.',
 'Serve as unprimed precursors to all helper T cell subsets.',
 'Act as a pool for generating novel responses to previously unseen antigens.',
 'Undergo robust IL-2-driven expansion during acute response.',
 'Increased metabolic activity and cell cycle progression.']


# -----------------------------
# FUNCTION: generate minimal A/B prompt
# -----------------------------
def generate_prompt(description, label_a, label_b):
    return f"""
You are a biomedical expert specializing in T cell immunology.

Functional description:
"{description}"

Option A: {label_a}
Option B: {label_b}

Question: Which of these two T cell types most likely performs the described function?

Please respond ONLY with one of the following exactly:
A
B
Do not add any other text.
"""

# -----------------------------
# FUNCTION: call OpenRouter LLM
# -----------------------------
# def call_llm(prompt, model="gpt-5-mini"):
#     response = openai.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a helpful and accurate immunology expert."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0
#     )
#     return response.choices[0].message["content"].strip()


def call_llm(prompt, model, token, max_retries=5, base_wait=5):
    """
    Calls an OpenRouter model with automatic retry on rate-limit or transient errors.
    - max_retries: maximum number of retry attempts
    - base_wait: initial wait time in seconds (doubles each retry)
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful and accurate immunology expert."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

            # Handle rate limits and temporary errors
            if response.status_code == 429:
                reset = None
                try:
                    # try to read X-RateLimit-Reset header if present
                    reset = int(response.headers.get("X-RateLimit-Reset", "0"))
                except ValueError:
                    pass
                wait_time = base_wait * (2 ** attempt)
                print(f"⚠️ Rate limit hit (attempt {attempt+1}/{max_retries}). "
                      f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            if response.status_code >= 500:
                wait_time = base_wait * (2 ** attempt)
                print(f"⚠️ Server error {response.status_code}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                print(f"❌ OpenRouter API error ({response.status_code}): {response.text}")
                return None

            # Parse normal response
            data = response.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
            else:
                print(f"⚠️ Unexpected API response: {data}")
                return None

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            wait_time = base_wait * (2 ** attempt)
            print(f"⚠️ Network/parse error: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    print(f"❌ Failed after {max_retries} retries.")
    return None


# -----------------------------
# MAIN: pairwise comparisons
# -----------------------------
results = []
model = "qwen/qwen3-235b-a22b:free"   
pairwise_records = []    # detailed: one row per pairwise comparison

for desc in tqdm(FUNCTIONAL_DESCRIPTIONS, desc="Functional descriptions"):
    win_count = {label: 0 for label in CELL_TYPES}
    pairs = list(itertools.combinations(CELL_TYPES, 2))

    for label_a, label_b in tqdm(pairs, desc="Pairs", leave=False):
        prompt = generate_prompt(desc, label_a, label_b)
        llm_output = call_llm(prompt, model=model, token=token)

        # Default values
        winner = None
        raw_output = llm_output

        # Update wins and record winner
        if llm_output == "A":
            win_count[label_a] += 1
            winner = label_a
        elif llm_output == "B":
            win_count[label_b] += 1
            winner = label_b
        else:
            print(f"Unexpected output: '{llm_output}' for {label_a} vs {label_b}")

        # Save pairwise record
        pairwise_records.append({
            "Functional Description": desc,
            "Label A": label_a,
            "Label B": label_b,
            "Winner": winner,
            "Raw Output": raw_output,
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    # Summary for this description
    df_row = {"Functional Description": desc, **win_count}
    results.append(df_row)

# -----------------------------
# CREATE AND SAVE DATAFRAMES
# -----------------------------
df_summary = pd.DataFrame(results)
df_pairs = pd.DataFrame(pairwise_records)

df_summary.to_csv("cell_type_pairwise_wins_qwen3-235.csv", index=False)
df_pairs.to_csv("cell_type_pairwise_detailed_qwen3-235.csv", index=False)

print("✅ Saved summary to 'cell_type_pairwise_wins_qwen3-235.csv'")
print("✅ Saved detailed log to 'cell_type_pairwise_detailed_qwen3-235.csv'")