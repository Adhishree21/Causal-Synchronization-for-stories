import zipfile
import os

team_name = "Decoders"
zip_name = f"{team_name}_KDSH_2026.zip"

files_to_include = [
    "results.csv",
    "Technical_Report.md",
    "requirements.txt",
    "README.md",
    "core/bdh_model.py",
    "core/tokenizer.py",
    "scripts/pathway_pipeline.py"
]

with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file in files_to_include:
        if os.path.exists(file):
            zipf.write(file)
            print(f"Added: {file}")
        else:
            print(f"Warning: File not found: {file}")

print(f"\nFinal submission package created: {zip_name}")
