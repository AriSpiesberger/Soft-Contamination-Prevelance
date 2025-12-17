from huggingface_hub import HfApi

REPO_ID = "allenai/dolma3_mix-6T-1025"
api = HfApi()

print(f"Inspecting file structure for {REPO_ID}...\n")

try:
    # Get the tree generator
    tree = api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)
    
    print("--- First 20 Items ---")
    for i, item in enumerate(tree):
        if i >= 20: break
        
        # safely get path and size without crashing on folders
        path = getattr(item, "path", "Unknown Path")
        size = getattr(item, "size", None)
        
        if size is not None:
            print(f"[FILE] {path}  ({size / 1e6:.2f} MB)")
        else:
            print(f"[DIR ] {path}")
            
except Exception as e:
    print(f"Error: {e}")