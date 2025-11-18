# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
实时结果收集器

每完成一个任务就立即写入结果到文件，确保任务中断时数据不丢失
"""

import os
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class RealtimeResultCollector:
    """实时结果收集器"""
    
    def __init__(self, output_dir: str):
        """
        初始化收集器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.txt_file = self.output_dir / f'realtime_results_{timestamp}.txt'
        self.csv_file = self.output_dir / f'realtime_results_{timestamp}.csv'
        
        # 初始化文件
        self._initialize_files()
    
    def _initialize_files(self):
        """初始化文件，写入表头"""
        # 初始化txt文件
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("AIKG 实时结果收集器\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        # 初始化csv文件
        with open(self.csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['任务名', 'Task文件夹', '个体路径', 'Torch时间(us)', 'AIKG时间(us)', '加速比'])
    
    def collect_task_result(self, 
                           op_name: str,
                           output_log_content: str) -> Dict[str, Any]:
        """
        收集单个任务的结果并立即写入文件
        
        Args:
            op_name: 算子名称
            output_log_content: 输出日志内容
            
        Returns:
            收集到的结果字典
        """
        print(f"\n{'='*80}")
        print(f"正在收集任务结果: {op_name}")
        print(f"{'='*80}")
        
        # 1. 从日志中提取Task文件夹名和log_dir路径
        task_folder_name, log_dir_path = self._extract_task_info_from_log(output_log_content)
        print(f"  Task文件夹: {task_folder_name}")
        
        # 2. 查找并解析 speed_up_record.txt（使用log_dir路径）
        speedup_records = []
        if log_dir_path:
            speedup_file = self._find_speedup_record_from_log_dir(log_dir_path, op_name)
            if speedup_file:
                speedup_records = self._parse_speedup_record(speedup_file)
                print(f"  找到 {len(speedup_records)} 条 speedup 记录")
        
        # 3. 从输出日志中提取Top 5信息（包括task_id）
        top5_results = self._parse_top5_from_log(output_log_content)
        print(f"  提取到 {len(top5_results)} 条 Top 结果")
        
        # 4. 提取最佳结果
        best_result = self._extract_best_result(speedup_records, top5_results)
        
        # 5. 立即写入到txt文件
        self._append_to_txt(op_name, task_folder_name, speedup_records, top5_results, best_result)
        
        # 6. 立即写入到csv文件
        self._append_to_csv(op_name, task_folder_name, best_result)
        
        print(f"  结果已写入文件")
        print(f"{'='*80}\n")
        
        return {
            'op_name': op_name,
            'task_folder': task_folder_name,
            'speedup_records': speedup_records,
            'top5_results': top5_results,
            'best_result': best_result
        }
    
    def _extract_task_info_from_log(self, log_content: str) -> tuple:
        """从日志内容中提取Task文件夹名和log_dir路径
        
        Returns:
            tuple: (task_folder_name, log_dir_path)
        """
        task_folder = 'Unknown_Task'
        log_dir = None
        
        try:
            lines = log_content.split('\n')
            for line in lines:
                line_stripped = line.strip()
                
                # 匹配格式: "Task文件夹: Task_xxx"
                if line_stripped.startswith('Task文件夹:'):
                    task_folder = line_stripped.split('Task文件夹:')[1].strip()
                
                # 匹配格式: "Log目录: /path/to/log_dir"
                if line_stripped.startswith('Log目录:'):
                    log_dir = line_stripped.split('Log目录:')[1].strip()
            
            if task_folder == 'Unknown_Task':
                print(f"  未在日志中找到Task文件夹信息")
            
            return task_folder, log_dir
        except Exception as e:
            print(f"  提取Task信息失败: {e}")
            return 'Unknown_Task', None
    
    def _find_speedup_record_from_log_dir(self, log_dir: str, op_name: str) -> Optional[Path]:
        """从log_dir查找 speed_up_record.txt 文件
        
        Args:
            log_dir: 完整的log目录路径（如 ~/aikg_logs/Task_xxx）
            op_name: 算子名称
            
        Returns:
            speed_up_record.txt 文件的Path对象，如果未找到则返回None
        """
        log_path = Path(os.path.expanduser(log_dir))
        
        # 构建可能的路径
        possible_paths = [
            log_path / op_name / 'profiling' / 'speed_up_record.txt',
            log_path / op_name / 'speed_up_record.txt',
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        print(f"  未找到 speed_up_record.txt")
        return None
    
    def _parse_speedup_record(self, file_path: Path) -> List[Dict]:
        """解析 speed_up_record.txt 文件"""
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            pattern = r'op_name:\s*([^,]+),\s*task_id:\s*([^,]+),\s*unique_dir:\s*([^,]+),\s*base_time:\s*([\d.]+)\s*us,\s*generation_time:\s*([\d.]+)\s*us,\s*speedup:\s*([\d.]+)x'
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                match = re.search(pattern, line)
                if match:
                    records.append({
                        'op_name': match.group(1).strip(),
                        'task_id': match.group(2).strip(),
                        'unique_dir': match.group(3).strip(),
                        'base_time': float(match.group(4)),
                        'generation_time': float(match.group(5)),
                        'speedup': float(match.group(6))
                    })
        except Exception as e:
            print(f"  解析 speed_up_record.txt 失败: {e}")
        
        return records
    
    def _parse_top5_from_log(self, log_content: str) -> List[Dict]:
        """从输出日志中提取Top 5结果，包括task_id"""
        top_results = []
        
        try:
            lines = log_content.split('\n')
            in_best_section = False
            rank = 0
            
            for line in lines:
                line = line.strip()
                
                # 检测最佳实现区域
                if '最佳实现' in line or 'Top' in line:
                    in_best_section = True
                    continue
                
                if in_best_section:
                    # 匹配格式: "1. aikg_1_xxx (轮次 4, 来源岛屿 1, 个体路径: I1_4_0_S02_verify, 生成代码: 323.6864us, ..."
                    # 或者: "1. aikg_1_xxx (轮次 4, 个体路径: I0_4_0_S02_verify, 生成代码: 323.6864us, ..."
                    pattern = r'(\d+)\.\s+([^\(]+)\s+\(轮次\s+(\d+)(?:,\s+来源岛屿\s+(\d+))?,\s+个体路径:\s+([^,]+),\s+生成代码:\s+([\d.]+)us,\s+基准代码:\s+([\d.]+)us,\s+加速比:\s+([\d.]+)x'
                    match = re.search(pattern, line)
                    
                    if match:
                        rank += 1
                        island = match.group(4)
                        round_num = match.group(3)
                        unique_dir = match.group(5).strip()  # 直接从日志中提取unique_dir
                        
                        top_results.append({
                            'rank': rank,
                            'op_name': match.group(2).strip(),
                            'round': int(round_num),
                            'island': int(island) if island else None,
                            'unique_dir': unique_dir,
                            'generation_time': float(match.group(6)),
                            'base_time': float(match.group(7)),
                            'speedup': float(match.group(8))
                        })
                        
                        if rank >= 5:
                            break
                    elif line and not line.startswith(('=', '-', '轮次', '每轮')):
                        # 如果遇到其他内容，可能离开了最佳实现区域
                        if rank > 0:
                            break
        except Exception as e:
            print(f"  解析Top结果失败: {e}")
        
        return top_results
    
    def _extract_best_result(self, speedup_records: List[Dict], top5_results: List[Dict]) -> Optional[Dict]:
        """提取最佳结果（最小生成时间）"""
        # 优先从speedup_records中找最小生成时间
        if speedup_records:
            best = min(speedup_records, key=lambda x: x['generation_time'])
            return {
                'unique_dir': best.get('unique_dir', 'N/A'),
                'torch_time': best['base_time'],
                'aikg_time': best['generation_time'],
                'speedup': best['speedup']
            }
        
        # 否则从top5中找最小生成时间
        if top5_results:
            best = min(top5_results, key=lambda x: x['generation_time'])
            return {
                'unique_dir': best.get('unique_dir', 'N/A'),
                'torch_time': best['base_time'],
                'aikg_time': best['generation_time'],
                'speedup': best['speedup']
            }
        
        return None
    
    def _append_to_txt(self, op_name: str, task_folder: str, 
                       speedup_records: List[Dict], top5_results: List[Dict],
                       best_result: Optional[Dict]):
        """追加写入到txt文件"""
        with open(self.txt_file, 'a', encoding='utf-8') as f:
            # 1. 算子名称
            f.write(f"{op_name}\n")
            
            # 2. Task文件夹名
            f.write(f"Task文件夹: {task_folder}\n")
            
            # 3. speed_up_record 文件的内容
            if speedup_records:
                for record in speedup_records:
                    f.write(f"op_name: {record['op_name']}, "
                           f"task_id: {record['task_id']}, "
                           f"unique_dir: {record['unique_dir']}, "
                           f"base_time: {record['base_time']:.6f} us, "
                           f"generation_time: {record['generation_time']:.6f} us, "
                           f"speedup: {record['speedup']:.6f}x\n")
            else:
                f.write("(无 speedup 记录)\n")
            
            # 4. 最佳实现 (前5个)
            if top5_results:
                f.write(f"最佳实现 (前{len(top5_results)}个):\n")
                for item in top5_results:
                    island_str = f", 来源岛屿 {item['island']}" if item['island'] is not None else ""
                    f.write(f"  {item['rank']}. {item['op_name']} "
                           f"(轮次 {item['round']}{island_str}, "
                           f"个体路径: {item['unique_dir']}, "
                           f"生成代码: {item['generation_time']:.4f}us, "
                           f"基准代码: {item['base_time']:.4f}us, "
                           f"加速比: {item['speedup']:.2f}x)\n")
            else:
                f.write("(无Top结果)\n")
            
            # 5. 最佳结果（最小生成时间）
            if best_result:
                f.write(f"最佳结果: 个体路径: {best_result.get('unique_dir', 'N/A')}, "
                       f"Torch时间: {best_result['torch_time']:.6f}us, "
                       f"AIKG时间: {best_result['aikg_time']:.6f}us, "
                       f"加速比: {best_result['speedup']:.6f}x\n")
            else:
                f.write("(无最佳结果)\n")
            
            f.write("\n")
            f.flush()  # 立即刷新到磁盘
    
    def _append_to_csv(self, op_name: str, task_folder: str, best_result: Optional[Dict]):
        """追加写入到csv文件（只写入最优实现）"""
        with open(self.csv_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if best_result:
                writer.writerow([
                    op_name,
                    task_folder,
                    best_result.get('unique_dir', 'N/A'),
                    f"{best_result['torch_time']:.6f}",
                    f"{best_result['aikg_time']:.6f}",
                    f"{best_result['speedup']:.6f}"
                ])
            else:
                writer.writerow([op_name, task_folder, 'N/A', 'N/A', 'N/A', 'N/A'])
            f.flush()  # 立即刷新到磁盘
    
    def get_output_files(self) -> Dict[str, Path]:
        """获取输出文件路径"""
        return {
            'txt': self.txt_file,
            'csv': self.csv_file
        }

