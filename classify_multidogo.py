"""
Classify messages from MultiDOGO dataset.
"""

from datasets import load_dataset
from agents import UrgencyAgent, EmotionalAgent
import json


def classify_messages(num_samples=100, save_path=None):
    """
    Classify messages from MultiDOGO dataset.
    
    Args:
        num_samples: Number of messages to classify
        save_path: Path to save results (optional)
        
    Returns:
        List of classification results
    """
    print("Loading dataset...")
    ds = load_dataset("jpcorb20/multidogo")
    data = ds["train"]
    
    print(f"Classifying {num_samples} messages...")
    
    urgency_agent = UrgencyAgent()
    emotional_agent = EmotionalAgent()
    
    results = []
    
    for i in range(num_samples):
        text = data[i]["utterance"]
        
        try:
            urgency = urgency_agent.classify(text)
            emotional = emotional_agent.classify(text)
            
            results.append({
                "index": i,
                "text": text,
                "urgency": urgency["urgency"],
                "emotional": emotional["emotional"],
                "conversationId": data[i]["conversationId"],
                "intent": data[i]["intent"]
            })
            
        except Exception as e:
            print(f"Error at index {i}: {e}")
            results.append({
                "index": i,
                "text": text,
                "urgency": "medium",
                "emotional": "neutral",
                "error": str(e)
            })
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{num_samples}")
    
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved to {save_path}")
    
    return results


if __name__ == "__main__":
    results = classify_messages(num_samples=100, save_path="results.json")
    
    print("\nSample results:")
    for r in results[:3]:
        print(f"\nText: {r['text']}")
        print(f"Urgency: {r['urgency']}, Emotional: {r['emotional']}")
