import os

# 配置
SEARCH_TEXT = 'StablePoly4'
REPLACE_TEXT = 'StablePoly7'
EXTENSIONS = {'.py', '.md', '.yaml', '.json'} # 只修改这些后缀的文件

def update_source_code(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in EXTENSIONS):
                file_path = os.path.join(root, file)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if SEARCH_TEXT in content:
                    new_content = content.replace(SEARCH_TEXT, REPLACE_TEXT)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"已更新源码: {file_path}")

if __name__ == "__main__":
    update_source_code('.')
