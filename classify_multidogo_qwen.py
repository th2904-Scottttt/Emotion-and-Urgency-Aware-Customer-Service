"""
Classify messages using Qwen2.5
"""

from datasets import load_dataset
from agents.qwen_urgency_agent import QwenUrgencyAgent
from agents.qwen_emotional_agent import QwenEmotionalAgent
import json


def classify_messages_qwen(num_samples=100, save_path=None):
    """
    Classify messages using Qwen2.5.
    """
    print("Loading dataset...")
    ds = load_dataset("jpcorb20/multidogo")
    data = ds["train"]
    
    print(f"Classifying {num_samples} messages with Qwen2.5...")
    
    urgency_agent = QwenUrgencyAgent()
    emotional_agent = QwenEmotionalAgent()
    
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
    results = classify_messages_qwen(num_samples=100, save_path="results_qwen.json")
    
    print("\nSample results:")
    for r in results[:3]:
        print(f"\nText: {r['text']}")
        print(f"Urgency: {r['urgency']}, Emotional: {r['emotional']}")