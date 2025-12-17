from huggingface_hub import HfApi

api = HfApi()
REPO_ID = "allenai/dolma3_mix-6T-1025"

tree = api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)

count = 0
for item in tree:
    print(f"TYPE: {type(item).__name__}, PATH: {getattr(item, 'path', 'N/A')}, SIZE: {getattr(item, 'size', 'N/A')}")
    count += 1
    if count >= 50:
        break

print(f"\nTotal items seen: {count}")