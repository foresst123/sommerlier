import json

path = '/home/lamkd2/Documents/sommerlier/podcast-pipeline/main.py'
with open(path, 'r') as f:
    content = f.read()

# I want to add config mapping.
# Wait, I can use replace_file_content or multi_replace_file_content directly.
