import os
import re

file_path = "podcast-pipeline/main_original_ASR_MoE.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace class name
content = content.replace("SepReformerSeparator", "SRCorrNetSeparator")

# Replace strings
content = content.replace("SepReformer", "SR-CorrNet")
content = content.replace("sepreformer", "srcorrnet")

# Replace the specific model path inside SRCorrNetSeparator
# We assume the directory for the large model is SR_CorrNet_L_WSJ0 based on naming conventions
content = content.replace("SR-CorrNet_Base_WSJ0", "SR_CorrNet_L_WSJ0")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Replacement complete.")
