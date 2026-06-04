import json

with open("checkpoints_k50/all_results.json") as f:
    results = json.load(f)

def print_topics(model_name, topic_words, n_topics=10):
    print(f"\n{'='*50}")
    print(f"  {model_name} — Top {n_topics} Topics")
    print(f"{'='*50}")
    for i, words in enumerate(topic_words[:n_topics]):
        print(f"  Topic {i+1:2d}: {', '.join(words)}")

print_topics("AVITM-ProdLDA", results["AVITM-ProdLDA"]["topic_words"])
print_topics("Gibbs-LDA",     results["Gibbs-LDA"]["topic_words"])
print_topics("AVITM-LDA",      results["AVITM-LDA"]["topic_words"])