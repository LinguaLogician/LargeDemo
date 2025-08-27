# -*- coding: utf-8 -*-
# @project: LargeDemo
# @filename: process.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/8/26 14:23

import json
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_notebook_cells(notebook_data):
    """
    清理notebook数据中的cells，移除outputs、metadata和execution_count属性

    Args:
        notebook_data (dict): 从.ipynb文件读取的JSON数据

    Returns:
        dict: 清理后的notebook数据
    """
    if 'cells' not in notebook_data:
        logger.warning("No 'cells' found in notebook data")
        return notebook_data
    if 'metadata' in notebook_data:
        del notebook_data['metadata']
    for cell in notebook_data['cells']:
        # 移除outputs属性
        if 'outputs' in cell:
            del cell['outputs']

        # 移除metadata属性
        if 'metadata' in cell:
            del cell['metadata']

        # 移除execution_count属性（通常在code cells中）
        if 'execution_count' in cell:
            del cell['execution_count']

    return notebook_data


def process_notebook_file(ipynb_path, output_dir=None):
    """
    处理单个.ipynb文件

    Args:
        ipynb_path (str/Path): .ipynb文件路径
        output_dir (str/Path, optional): 输出目录，默认为None（同目录）

    Returns:
        bool: 处理是否成功
    """
    try:
        ipynb_path = Path(ipynb_path)

        # 读取.ipynb文件
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)

        # 清理cells
        cleaned_data = clean_notebook_cells(notebook_data)

        # 确定输出路径
        if output_dir:
            output_path = Path(output_dir) / f"{ipynb_path.stem}.json"
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_path = ipynb_path.parent / f"{ipynb_path.stem}.json"

        # 写入清理后的JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        logger.error(f"Error processing {ipynb_path}: {str(e)}")
        return False


def find_ipynb_files(base_dir):
    """
    查找base_dir目录下所有的.ipynb文件

    Args:
        base_dir (str/Path): 基础目录路径

    Returns:
        list: 所有.ipynb文件的路径列表
    """
    base_dir = Path(base_dir)
    ipynb_files = []

    # 递归查找所有.ipynb文件
    for pattern in ["**/*.json", "*.ipynb"]:
        ipynb_files.extend(base_dir.glob(pattern))

    # 去除隐藏文件（以.开头的文件）
    ipynb_files = [f for f in ipynb_files if not any(part.startswith('.') for part in f.parts)]

    return ipynb_files


def process_all_notebooks(base_dir):
    """
    处理base_dir目录下所有的.ipynb文件

    Args:
        base_dir (str/Path): 基础目录路径
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        logger.error(f"Base directory {base_dir} does not exist")
        return

    # 查找所有.ipynb文件
    ipynb_files = find_ipynb_files(base_dir)

    if not ipynb_files:
        logger.warning(f"No .ipynb files found in {base_dir}")
        return

    logger.info(f"Found {len(ipynb_files)} .ipynb files to process")

    # 使用进度条处理文件
    success_count = 0
    for ipynb_file in tqdm(ipynb_files, desc="Processing notebooks"):
        if process_notebook_file(ipynb_file):
            success_count += 1
        else:
            logger.warning(f"Failed to process: {ipynb_file}")

    logger.info(f"Processing completed. Success: {success_count}/{len(ipynb_files)}")


def main():
    """主函数"""

    # args = parser.parse_args()

    logger.info(f"Starting processing of notebooks in: jupyter")
    process_all_notebooks('json/')
    logger.info("Processing finished")


if __name__ == "__main__":
    main()
