"""
Export regenerated stories comparison to CSV format for easy analysis.

Creates a CSV with columns:
- sample_number
- victim
- weapon
- crime_scene
- suspects (comma-separated)
- murderer
- question
- choices (pipe-separated)
- answer_index
- answer_choice
- num_regenerations
- original_story
- regenerated_story_1, regenerated_story_2, ... (dynamic based on num_regenerations)
"""

import json
import csv
from pathlib import Path

def main():
    from src.utils.paths import OUTPUT_FOLDER
    
    input_file = OUTPUT_FOLDER / 'murder_mystery_regenerated_first5_v3.json'
    output_file = OUTPUT_FOLDER / 'murder_mystery_comparison.csv'
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        print("Run regenerate_from_trees.py first.")
        return
    
    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine maximum number of regenerations across all samples
    max_regenerations = max(entry.get('num_regenerations', 1) for entry in data)
    print(f"Exporting {len(data)} samples with up to {max_regenerations} regenerations each to CSV...")
    
    # Prepare CSV data
    rows = []
    for entry in data:
        # Extract question data
        question_data = entry['questions'][0]
        question_text = question_data['question']
        choices = question_data['choices']
        answer_index = question_data['answer']
        answer_choice = choices[answer_index]
        
        # Get regenerated stories (handle both old single-story and new multi-story format)
        if 'regenerated_stories' in entry:
            # New format: list of stories
            regenerated_stories = entry['regenerated_stories']
            num_regenerations = entry.get('num_regenerations', len(regenerated_stories))
        else:
            # Old format: single story (backwards compatibility)
            regenerated_stories = [entry.get('regenerated_story', '')]
            num_regenerations = 1
        
        # Create row with base fields
        row = {
            'sample_number': entry['sample_number'],
            'victim': entry['victim'],
            'weapon': entry['weapon'],
            'crime_scene': entry['crime_scene'],
            'suspects': ', '.join(entry['suspects']),
            'murderer': entry['murderer'],
            'question': question_text,
            'choices': str(choices),  # Format as Python list: ['Name1', 'Name2']
            'answer_index': answer_index,
            'answer_choice': answer_choice,
            'num_regenerations': num_regenerations,
            'original_story': entry['original_story']
        }
        
        # Add regenerated stories dynamically
        for i, story in enumerate(regenerated_stories, start=1):
            row[f'regenerated_story_{i}'] = story
        
        # Fill in empty columns for samples with fewer regenerations
        for i in range(len(regenerated_stories) + 1, max_regenerations + 1):
            row[f'regenerated_story_{i}'] = ''
        
        rows.append(row)
    
    # Build fieldnames dynamically
    base_fieldnames = [
        'sample_number',
        'victim',
        'weapon',
        'crime_scene',
        'suspects',
        'murderer',
        'question',
        'choices',
        'answer_index',
        'answer_choice',
        'num_regenerations',
        'original_story'
    ]
    
    # Add regenerated story columns
    regenerated_fieldnames = [f'regenerated_story_{i}' for i in range(1, max_regenerations + 1)]
    fieldnames = base_fieldnames + regenerated_fieldnames
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n{'='*80}")
    print(f"✓ Export complete!")
    print(f"  Output: {output_file}")
    print(f"  Rows: {len(rows)}")
    print(f"  Columns: {len(fieldnames)}")
    print(f"  Base columns: {len(base_fieldnames)}")
    print(f"  Regenerated story columns: {max_regenerations}")
    print(f"{'='*80}\n")
    
    # Show preview
    print("Preview of first row:")
    print(f"{'-'*80}")
    for key, value in rows[0].items():
        # Truncate long story fields
        if 'story' in key:
            preview = value[:100] + '...' if len(value) > 100 else value
            print(f"  {key}: {preview}")
        else:
            print(f"  {key}: {value}")
    print(f"{'-'*80}\n")

if __name__ == "__main__":
    main()


