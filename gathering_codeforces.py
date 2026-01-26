import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

def generate_uniform_recent_dataset(
    dataset_id="open-r1/codeforces",
    split="train",
    samples_per_bin=50,
    bin_width=100
):
    """
    Fetches the Codeforces dataset and samples it to be uniformly distributed 
    across Elo ratings, prioritizing the most recent problems in each bin.
    
    Args:
        dataset_id (str): Hugging Face dataset ID.
        split (str): Dataset split to use (e.g., 'train', 'default').
        samples_per_bin (int): Number of problems to select per Elo bin.
        bin_width (int): The width of the Elo buckets (e.g., 800-899).
        
    Returns:
        pd.DataFrame: The sampled dataset.
    """
    print(f"Loading dataset '{dataset_id}' (split: {split})...")
    try:
        # Load the dataset (downloading if necessary)
        ds = load_dataset(dataset_id, split=split)
        df = ds.to_pandas()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # 1. Preprocessing
    # Ensure rating is numeric and drop entries without rating or date
    # Some contest entries might have null ratings if they are unrated or errors.
    print(f"Original size: {len(df)} rows")
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating', 'contest_start'])
    
    # 2. Create Elo Bins
    # We floor divide by bin_width to create buckets (e.g., 1540 // 100 = 15 -> Bin 1500)
    df['elo_bin'] = (df['rating'] // bin_width) * bin_width
    
    # 3. Stratified Sampling with Recency Bias
    sampled_frames = []
    unique_bins = sorted(df['elo_bin'].unique())
    
    print(f"Stratifying across {len(unique_bins)} Elo bins (Width: {bin_width})...")
    
    for b in unique_bins:
        # Filter for the current bin
        bin_df = df[df['elo_bin'] == b]
        
        # Sort by contest_start descending (Recent first)
        bin_df_sorted = bin_df.sort_values(by='contest_start', ascending=False)
        
        # Select the top k samples
        # If a bin has fewer than samples_per_bin, we take all of them.
        sample = bin_df_sorted.head(samples_per_bin)
        sampled_frames.append(sample)

    # 4. Aggregate
    final_df = pd.concat(sampled_frames)
    
    print(f"Final sampled size: {len(final_df)} rows")
    return final_df

if __name__ == "__main__":
    # Configuration
    OUTPUT_FILE = "codeforces_uniform_recent.csv"
    SAMPLES_PER_BIN = 20  # Adjust this based on your total size needs
    BIN_WIDTH = 100       # 100 Elo points per bin
    
    # Execution
    df_uniform = generate_uniform_recent_dataset(
        samples_per_bin=SAMPLES_PER_BIN,
        bin_width=BIN_WIDTH
    )
    
    if df_uniform is not None:
        # Save to CSV
        df_uniform.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved dataset to {OUTPUT_FILE}")
        
        # Optional: Verify Distribution
        print("\nSample counts per bin:")
        print(df_uniform['elo_bin'].value_counts().sort_index())
        
        # Check recency of the head
        print("\nMost recent selected problems (Head):")
        print(df_uniform[['title', 'rating', 'contest_start_year']].head())