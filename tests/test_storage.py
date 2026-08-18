# tests/test_storage.py
"""Storage 模块单元测试"""

import os
import sys

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from storage import Storage


def test_storage_initialization():
    """测试 1：数据库初始化"""
    storage = Storage(':memory:')
    
    # 验证表已创建
    conn = storage._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert 'concepts' in tables, "concepts 表未创建"
    assert 'insights' in tables, "insights 表未创建"
    assert 'search_logs' in tables, "search_logs 表未创建"
    print("✅ 测试 1 通过：数据库初始化成功")


def test_concept_crud():
    """测试 2：概念 CRUD"""
    storage = Storage(':memory:')
    
    # Create
    success = storage.create_concept(
        id="c001",
        name="负利率",
        description="央行对存款收取利息",
        category="货币政策"
    )
    assert success, "创建概念失败"
    
    # Read
    concept = storage.get_concept("c001")
    assert concept is not None, "查询概念返回 None"
    assert concept['name'] == "负利率", "概念名称不匹配"
    assert concept['category'] == "货币政策", "概念类别不匹配"
    
    # Update
    storage.update_concept("c001", description="新的描述")
    concept = storage.get_concept("c001")
    assert concept['description'] == "新的描述", "更新失败"
    
    # List
    concepts = storage.list_concepts()
    assert len(concepts) == 1, "列表数量不对"
    
    # Delete
    storage.delete_concept("c001")
    concept = storage.get_concept("c001")
    assert concept is None, "删除后仍能查询到"
    
    print("✅ 测试 2 通过：概念 CRUD 正常")


def test_insight_crud():
    """测试 3：Insight CRUD"""
    storage = Storage(':memory:')
    
    # Create
    storage.create_insight(
        id="i001",
        title="负利率是对储蓄的惩罚",
        content="负利率不是奖励借钱，而是惩罚存钱",
        trigger_content="日本负利率政策",
        source="金融深潜 W1",
        related_concepts=["负利率", "货币政策"],
        action_items=["重新审视资产配置"]
    )
    
    # Read
    insight = storage.get_insight("i001")
    assert insight is not None, "查询 Insight 返回 None"
    assert insight['title'] == "负利率是对储蓄的惩罚", "标题不匹配"
    assert "负利率" in insight['related_concepts'], "关联概念不匹配"
    assert "重新审视资产配置" in insight['action_items'], "行动项不匹配"
    
    print("✅ 测试 3 通过：Insight CRUD 正常")


def test_search_logs():
    """测试 4：搜索日志"""
    storage = Storage(':memory:')
    
    # Log searches
    storage.log_search("负利率", 5, 0.123)
    storage.log_search("量化宽松", 3, 0.089)
    storage.log_search("通货膨胀", 10, 0.234)
    
    # Stats
    stats = storage.get_search_stats(days=7)
    assert stats['total_searches'] == 3, "搜索次数不对"
    assert stats['avg_results'] == 6.0, "平均结果数不对"  # (5+3+10)/3
    
    print("✅ 测试 4 通过：搜索日志正常")


if __name__ == "__main__":
    test_storage_initialization()
    test_concept_crud()
    test_insight_crud()
    test_search_logs()
    print("\n🎉 所有测试通过！")
