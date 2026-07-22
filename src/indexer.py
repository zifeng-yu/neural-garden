"""
Neural Garden - 知识单元索引器
Lesson 01: 骨架版本（支持单文件导入）

完整功能将在 Lesson 02 实现：
- 批量导入 pilot/ 目录下所有 .md 文件
- 知识单元提取（标题、摘要、关键词）
- 调用 DashScope Embedding API
"""

import chromadb
from chromadb.config import Settings
import os
import hashlib
import yaml

def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_embedding(text: str, api_key: str) -> list:
    """
    调用 DashScope API 获取文本向量
    
    Args:
        text: 输入文本
        api_key: DashScope API Key
        
    Returns:
        向量列表（float）
    """
    try:
        import dashscope
        dashscope.api_key = api_key
        
        response = dashscope.TextEmbedding.generate(
            model='text-embedding-v3',
            input=text
        )
        
        if response.status_code == 200:
            return response.output['embeddings'][0]['embedding']
        else:
            print(f"⚠️  Embedding API 调用失败：{response.code} - {response.message}")
            return None
    except Exception as e:
        print(f"⚠️  Embedding 调用异常：{e}")
        return None

def load_document(file_path: str) -> str:
    """
    读取 Markdown 文件内容
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件内容字符串
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_id(content: str) -> str:
    """
    生成内容哈希 ID（避免重复）
    
    Args:
        content: 文件内容
        
    Returns:
        16 位哈希字符串
    """
    return hashlib.md5(content.encode()).hexdigest()[:16]

def index_file(file_path: str, persist_dir: str, api_key: str):
    """
    将单个文件导入 Chroma 向量库
    
    Args:
        file_path: Markdown 文件路径
        persist_dir: Chroma 持久化目录
        api_key: DashScope API Key
    """
    # 1. 读取文件
    content = load_document(file_path)
    doc_id = generate_id(content)
    source = os.path.basename(file_path)
    
    print(f"📄 正在索引：{source}")
    
    # 2. 初始化 Chroma 客户端
    client = chromadb.Client(Settings(persist_directory=persist_dir))
    collection = client.get_or_create_collection("knowledge")
    
    # 3. 检查是否已存在
    existing = collection.get(ids=[doc_id])
    if existing['ids']:
        print(f"⏭️  已存在，跳过：{source}")
        return
    
    # 4. 调用 Embedding API
    print(f"🔢 正在生成向量...")
    embedding = get_embedding(content, api_key)
    
    if embedding is None:
        print(f"⚠️  跳过 {source}（Embedding 失败）")
        return
    
    # 5. 添加到向量库
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{'source': source}]
    )
    
    print(f"✅ 索引完成：{source}")

def index_directory(dir_path: str, persist_dir: str, api_key: str):
    """
    批量索引目录下所有 .md 文件
    
    Args:
        dir_path: 目录路径
        persist_dir: Chroma 持久化目录
        api_key: DashScope API Key
    """
    md_files = [f for f in os.listdir(dir_path) if f.endswith('.md')]
    
    if not md_files:
        print(f"⚠️  未找到 .md 文件：{dir_path}")
        return
    
    print(f"📂 发现 {len(md_files)} 个 Markdown 文件\n")
    
    for filename in md_files:
        file_path = os.path.join(dir_path, filename)
        index_file(file_path, persist_dir, api_key)
    
    print(f"\n✅ 批量索引完成，共 {len(md_files)} 个文件")

if __name__ == "__main__":
    import sys
    
    # 加载配置
    config = load_config()
    api_key = config.get('dashscope', {}).get('api_key', '')
    
    if not api_key or api_key == 'your-api-key-here':
        print("❌ 错误：请先配置 config.yaml 中的 DashScope API Key")
        print(f"📁 配置文件路径：{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')}")
        sys.exit(1)
    
    # 配置路径
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(base_dir, "data/chroma")
    pilot_dir = os.path.join(base_dir, "data/pilot")
    
    if len(sys.argv) > 1:
        # 指定文件路径
        file_path = sys.argv[1]
        if os.path.isfile(file_path):
            index_file(file_path, persist_dir, api_key)
        elif os.path.isdir(file_path):
            index_directory(file_path, persist_dir, api_key)
        else:
            print(f"❌ 路径不存在：{file_path}")
            sys.exit(1)
    else:
        # 默认索引 pilot/ 目录
        print("🌱 Neural Garden 索引器（骨架版本）")
        print(f"📂 索引目录：{pilot_dir}")
        print(f"💾 存储目录：{persist_dir}\n")
        index_directory(pilot_dir, persist_dir, api_key)
        print("\n" + "=" * 40)
        print("💡 提示：运行以下命令搜索知识：")
        print(f"   python -m src.search <查询内容>")
