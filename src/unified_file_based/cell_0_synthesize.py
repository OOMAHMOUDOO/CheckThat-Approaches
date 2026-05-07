import os
import re
import json
import asyncio
import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
import random 
from collections import defaultdict

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Path configurations
BASE_DIR = config["paths"]["base_dir"]
TRAIN_JSON = config["paths"]["source_train_json"]
OUTPUT_FILE = config["paths"]["deepseek_audit_output"]

# Synthesis Configurations
DEEPSEEK_MODEL = config["synthesis"]["model_name"]
MAX_TOKENS = config["synthesis"]["max_tokens"]
TEMPERATURE = config["synthesis"]["temperature"]
CONCURRENCY = config["synthesis"]["concurrency_limit"]
API_KEY = "sk-ff0b529f17274850b4d86e7ab1a5fc18" #str(userdata.get('DEEPSEEK_API_KEY'))

SYSTEM_PROMPT = (
    "You are an expert fact-checking auditor specializing in numerical, statistical, and temporal claims. "
    "Your task is to rigorously evaluate whether a previous fact-checker's verdict and justification are correct, "
    "given the original claim and retrieved evidence.\n\n"
    "Verdict Definitions:\n"
    "- True: The evidence fully supports the claim.\n"
    "- False: The evidence directly contradicts the claim.\n"
    "- Conflicting: The evidence contains contradictory elements or is inconclusive.\n\n"
    "Evaluation Criteria:\n"
    "1. Evidence Grounding: Does the justification accurately reflect the provided evidence without hallucination?\n"
    "2. Numerical/Temporal Precision: Are all quantities, dates, and calculations correctly verified? Beware of the 'Numeric-Truth Effect'.\n"
    "3. Logical Consistency: Does the reasoning chain logically support the verdict?\n"
    "4. Verdict Alignment: Is the stated verdict consistent with the definitions above given the evidence?\n\n"
    "Output Rules:\n"
    "- Put your judgement inside <TheThink>... </TheThink> tags.\n"
    "- After </TheThink>, output EXACTLY one of the following formats:\n"
    "  • Yes, verdict is correct.\n"
    "  • No, verdict should be [Correct Verdict].\n"
    "- Replace [Correct Verdict] with True, False, or Conflicting.\n"
    "- Do not output any additional text after this line And make the judgement concise but expressive."
)

USER_TEMPLATE = (
    "### {input_text} \n\n" 
    "Audit this verdict and justification according to the criteria and definitions. "
    "Provide your reasoning in <TheThink> tags, then output your final assessment in the exact required format."
)

def remove_label_pattern(text):
    text = re.sub(
        r"(\[?\s*Justification\s*\]?:?\s*)|(\[Label\]:\s*(True|False|Conflicting))",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text.replace("\n", " ")




def get_slim_data(unrolled_data, percentage=0.3):
    """
    Return a stratified random sample from unrolled_data.

    Args:
        unrolled_data (list[dict]): Each dict must contain a "Class" key (0 or 1).
        percentage (float): Fraction of data to keep.

    Returns:
        list[dict]: Stratified sampled data.
    """

    # Group data by class
    class_groups = defaultdict(list)

    for item in unrolled_data:
        class_groups[item["class"]].append(item)

    slim_data = []

    # Sample from each class proportionally
    for cls, items in class_groups.items():
        random.shuffle(items)

        sample_size = max(1, int(len(items) * percentage))
        slim_data.extend(items[:sample_size])

    # Final shuffle so classes are mixed
    random.shuffle(slim_data)

    return slim_data
    
def unroll_data(data, percentage=0.3):
    unrolled = []
    for idx, item in enumerate(data):
        label = item.get("label", "")
        claim = item.get("claim", "")
        verdict_list = item.get("Verdict_list", [])
        reasoning_traces = item.get("Reasoning_traces", [])
        evidences = idx #item.get("evidences", [])
        if not evidences:
            continue
        sample_id = item.get("id", idx)
        
        for v, trace in zip(verdict_list, reasoning_traces):
            justification = remove_label_pattern(trace).split("Label:")[0].strip()
            class_label = 1 if str(v).lower() == str(label).lower() else 0
            unrolled.append({
                "query_id": sample_id,
                "input_text": f"""### Claim:\n{claim}\n\n
                    ### Retrieved evidences:\n{"".join(evidences)}\n\n
                    ### Previous Verdict:\n{str(v).lower()}\n\n
                    ### Previous Justification:\n{justification}\n\n
                """,
                "label": label,
                "verdict": v,
                "class": class_label
            })
    slim_data = get_slim_data(unrolled, percentage)
    return slim_data

# Async OpenAI client
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
    timeout=60.0
)

# Auto-retry on rate limits, timeouts, or connection errors
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)),
    reraise=True
)
async def audit_item(item, semaphore):
    async with semaphore:
        user_content = USER_TEMPLATE.format(input_text=item['input_text'])
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"
        
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            #reasoning_effort="low",
            extra_body={"thinking": {"type": "disabled"}}

        )
        return response.choices[0].message.content

async def main():
    print(f"Loading data from {TRAIN_JSON}...")
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if config.get("run_config", {}).get("is_sanity", False):
        print("SANITY MODE ON: Truncating data to 2 items")
        data = data[:2]

    
    unrolled_data = unroll_data(data)
    print(f"\033[95mTotal slim data: {len(unrolled_data)}\033[0m")

    # Resume logic
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    res = json.loads(line)
                    q_id = res.get('query_id', res.get('query_id'))
                    orig_v = res.get('original_verdict', '')
                    processed_ids.add(f"{q_id}_{orig_v}")
        print(f"Resuming from {len(processed_ids)} already processed items.")

    # Filter out already processed items
    pending_items = [item for item in unrolled_data if f"{item['query_id']}_{item['verdict']}" not in processed_ids]
    print(f"Items to process: {len(pending_items)}")
    if not pending_items:
        print("✅ All items already processed. Exiting.")
        return

    # Concurrency control
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results_buffer = []
    save_lock = asyncio.Lock()

    async def process_and_save(item):
        try:
            max_logical_retries = 3
            audit_output = ""
            for attempt in range(max_logical_retries):
                audit_output = await audit_item(item, semaphore)
                
                parts = audit_output.split("</TheThink>")
                answer = parts[-1].strip().lower() if len(parts) > 1 else audit_output.strip().lower()
                
                is_yes = answer.startswith("yes")
                is_no = answer.startswith("no")
                
                needs_retry = False
                if is_no and item["class"] == 1:
                    needs_retry = True
                elif is_yes and item["class"] == 0:
                    needs_retry = True
                    
                if not needs_retry:
                    # Post-processing: force correct verdict label
                    if is_no and item["class"] == 0:
                        TheThink_part = audit_output[:audit_output.rfind("</TheThink>")] + "</TheThink>"
                        audit_output = f"{TheThink_part}\n\nNo, verdict should be {str(item['label'])}."
                    break
                else:
                    if attempt == max_logical_retries - 1:
                        print(f"\n[WARNING] Max retries reached for {item['query_id']}. Proceeding with last output.")

            result = {
                "query_id": item["query_id"],
                "input_text": item["input_text"],
                "original_label": item["label"],
                "original_verdict": item["verdict"],
                "is_correct_label": item["class"],
                "response": audit_output
            }
            async with save_lock:
                results_buffer.append(result)
                # Batch save every 20 items to reduce I/O overhead
                if len(results_buffer) >= 20:
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        for r in results_buffer:
                            f.write(json.dumps(r) + "\n")
                    results_buffer.clear()
        except Exception as e:
            print(f"\n❌ Failed sample {item['query_id']}_{item['verdict']} after API retries: {e}")

    # Run concurrently with progress bar
    tasks = [process_and_save(item) for item in pending_items]
    await tqdm_asyncio.gather(*tasks, desc="Auditing with DeepSeek")

    # Save remaining buffer
    if results_buffer:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for r in results_buffer:
                f.write(json.dumps(r) + "\n")

    print(f"\n✅ Processing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())