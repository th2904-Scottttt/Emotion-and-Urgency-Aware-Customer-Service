"""
Compare OpenAI vs Qwen classification results
"""

import json


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_results(openai_file, qwen_file):
    """Compare two classification results."""
    
    openai_results = load_results(openai_file)
    qwen_results = load_results(qwen_file)
    
    print("=" * 70)
    print("Classification Results Comparison")
    print("=" * 70)
    
    # Statistics
    print("\n[1] Distribution Comparison")
    print("-" * 70)
    
    # Urgency distribution
    openai_urgency = {"low": 0, "medium": 0, "high": 0}
    qwen_urgency = {"low": 0, "medium": 0, "high": 0}
    
    for r in openai_results:
        if "urgency" in r:
            openai_urgency[r["urgency"]] += 1
    
    for r in qwen_results:
        if "urgency" in r:
            qwen_urgency[r["urgency"]] += 1
    
    print("\nUrgency Distribution:")
    print(f"{'Level':<10} {'OpenAI':<15} {'Qwen':<15} {'Difference'}")
    for level in ["low", "medium", "high"]:
        diff = qwen_urgency[level] - openai_urgency[level]
        print(f"{level:<10} {openai_urgency[level]:<15} {qwen_urgency[level]:<15} {diff:+d}")
    
    # Emotional distribution
    emotions = ["angry", "frustrated", "anxious", "neutral", "satisfied", "happy"]
    openai_emotional = {e: 0 for e in emotions}
    qwen_emotional = {e: 0 for e in emotions}
    
    for r in openai_results:
        if "emotional" in r and r["emotional"] in openai_emotional:
            openai_emotional[r["emotional"]] += 1
    
    for r in qwen_results:
        if "emotional" in r and r["emotional"] in qwen_emotional:
            qwen_emotional[r["emotional"]] += 1
    
    print("\nEmotional Distribution:")
    print(f"{'Emotion':<15} {'OpenAI':<15} {'Qwen':<15} {'Difference'}")
    for emotion in emotions:
        diff = qwen_emotional[emotion] - openai_emotional[emotion]
        print(f"{emotion:<15} {openai_emotional[emotion]:<15} {qwen_emotional[emotion]:<15} {diff:+d}")
    
    # Agreement rate
    print("\n" + "=" * 70)
    print("[2] Agreement Analysis")
    print("-" * 70)
    
    urgency_agree = 0
    emotional_agree = 0
    both_agree = 0
    
    for i in range(min(len(openai_results), len(qwen_results))):
        o = openai_results[i]
        q = qwen_results[i]
        
        if o.get("urgency") == q.get("urgency"):
            urgency_agree += 1
        
        if o.get("emotional") == q.get("emotional"):
            emotional_agree += 1
        
        if o.get("urgency") == q.get("urgency") and o.get("emotional") == q.get("emotional"):
            both_agree += 1
    
    total = min(len(openai_results), len(qwen_results))
    
    print(f"Urgency Agreement:    {urgency_agree}/{total} ({urgency_agree/total*100:.1f}%)")
    print(f"Emotional Agreement:  {emotional_agree}/{total} ({emotional_agree/total*100:.1f}%)")
    print(f"Both Agree:           {both_agree}/{total} ({both_agree/total*100:.1f}%)")
    
    # Show disagreements
    print("\n" + "=" * 70)
    print("[3] Sample Disagreements (first 5)")
    print("-" * 70)
    
    disagreements = []
    for i in range(min(len(openai_results), len(qwen_results))):
        o = openai_results[i]
        q = qwen_results[i]
        
        if o.get("urgency") != q.get("urgency") or o.get("emotional") != q.get("emotional"):
            disagreements.append((i, o, q))
    
    for idx, (i, o, q) in enumerate(disagreements[:5]):
        print(f"\n[{idx+1}] Text: {o['text'][:60]}...")
        print(f"    OpenAI:  urgency={o.get('urgency'):<8} emotional={o.get('emotional')}")
        print(f"    Qwen:    urgency={q.get('urgency'):<8} emotional={q.get('emotional')}")
    
    print(f"\nTotal disagreements: {len(disagreements)}/{total}")


if __name__ == "__main__":
    compare_results("results.json", "results_qwen.json")