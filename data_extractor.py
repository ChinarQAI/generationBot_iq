import os        


def extract_text(path):
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            file_path = os.path.join(str(path+"/"+filename))
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                return text