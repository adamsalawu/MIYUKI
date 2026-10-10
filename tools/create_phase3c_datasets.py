"""Create derived Phase 3C datasets A and B from miyuki-writing-0.1 without modifying original."""
import json, pathlib, hashlib, datetime

orig_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
orig_bytes = orig_path.read_bytes()
orig_sha = hashlib.sha256(orig_bytes).hexdigest()
print(f"Original SHA {orig_sha} len {len(orig_path.read_text(encoding='utf-8').splitlines())}")

orig_examples = [json.loads(l) for l in orig_path.read_text(encoding="utf-8").splitlines() if l.strip()]

# Behavioral contract - concise, consistent
SYSTEM_CONTRACT = (
    "You are Miyuki, a direct story-writing assistant. "
    "Perform the requested writing task directly. "
    "Do not explain, announce, summarize, or discuss the task. "
    "Follow constraints, preserve character and context facts, do not invent dialogue when forbidden, "
    "continue the story rather than discussing it, and when rewriting output the rewritten text directly."
)

USER_CONTRACT_SUFFIX = (
    "\n\nRequirements: Perform the writing directly. Do not explain, announce, summarize, or discuss. "
    "Follow constraints, preserve context, do not invent dialogue when forbidden, continue story not discuss, when rewriting output directly."
)

# Formulation A: System-level contract (system message + user as original + assistant same)
# Formulation B: User-only contract (user = original user + suffix, no system)

out_dir = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1-phase3c")
out_dir.mkdir(parents=True, exist_ok=True)

def make_a(ex):
    # ex has messages [user, assistant]
    user_content = ex["messages"][0]["content"]
    assistant_content = ex["messages"][1]["content"]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_CONTRACT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            **ex["metadata"],
            "derived_from": ex["metadata"]["id"],
            "formulation": "A-system-contract",
            "system_contract": SYSTEM_CONTRACT
        }
    }

def make_b(ex):
    user_content = ex["messages"][0]["content"]
    # Append user contract suffix
    new_user = user_content + USER_CONTRACT_SUFFIX
    assistant_content = ex["messages"][1]["content"]
    return {
        "messages": [
            {"role": "user", "content": new_user},
            {"role": "assistant", "content": assistant_content}
        ],
        "metadata": {
            **ex["metadata"],
            "derived_from": ex["metadata"]["id"],
            "formulation": "B-user-contract",
            "user_suffix": USER_CONTRACT_SUFFIX
        }
    }

examples_a = [make_a(ex) for ex in orig_examples]
examples_b = [make_b(ex) for ex in orig_examples]

path_a = out_dir / "dataset-a.jsonl"
path_b = out_dir / "dataset-b.jsonl"
path_a.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples_a) + "\n", encoding="utf-8")
path_b.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in examples_b) + "\n", encoding="utf-8")

sha_a = hashlib.sha256(path_a.read_bytes()).hexdigest()
sha_b = hashlib.sha256(path_b.read_bytes()).hexdigest()
print(f"A SHA {sha_a} len {len(examples_a)}")
print(f"B SHA {sha_b} len {len(examples_b)}")
print(f"Original remains {hashlib.sha256(orig_path.read_bytes()).hexdigest() == orig_sha}")

# Verify original byte-for-byte unchanged
assert hashlib.sha256(orig_path.read_bytes()).hexdigest() == orig_sha, "original modified!"

# Create manifest for derived
manifest = {
    "source_dataset": "miyuki-writing-0.1",
    "source_sha": orig_sha,
    "source_examples": len(orig_examples),
    "derived": {
        "A": {"path": "dataset-a.jsonl", "sha": sha_a, "examples": len(examples_a), "formulation": "A-system-contract", "system_contract": SYSTEM_CONTRACT},
        "B": {"path": "dataset-b.jsonl", "sha": sha_b, "examples": len(examples_b), "formulation": "B-user-contract", "user_suffix": USER_CONTRACT_SUFFIX}
    },
    "transformation_method": "Added concise behavioral contract (system for A, user suffix for B) without changing 38 writing targets. Assistant targets identical to original.",
    "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "tokenizer": "Qwen/Qwen2-0.5B-Instruct (151646)",
    "notes": "Derived for Phase 3C training formulation experiment. Only instruction formulation changes."
}
out_manifest = out_dir / "manifest.json"
out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out_manifest}")

# Also ensure original dataset still byte-for-byte
print("DONE")
