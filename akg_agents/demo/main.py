"""
算子任务构建 Demo - 入口文件

使用方式:
  # 交互模式
  python main.py

  # 直接传入文件
  python main.py --input path/to/code.py --desc "RMS归一化算子"

  # 传入目录
  python main.py --input path/to/repo/ --desc "提取其中的softmax算子"
"""
import argparse
import json
import sys
import logging
from pathlib import Path

# 确保 demo 目录在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.agent import ReactAgent
from demo.config import OUTPUT_DIR


def setup_logging(debug: bool = False, log_file: str = None):
    level = logging.DEBUG if debug else logging.WARNING
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )


def interactive_mode(agent: ReactAgent):
    """交互模式: 引导用户输入"""
    print("=" * 60)
    print("  算子任务构建工具 (ReAct Agent)")
    print("=" * 60)
    print()
    print("请输入代码或文件/目录路径。")
    print("  - 直接粘贴代码（输入 END 结束）")
    print("  - 输入文件绝对/相对路径")
    print("  - 输入目录路径")
    print("  - 可在路径后空格追加描述")
    print()

    user_input_lines = []
    first_line = input("输入 > ").strip()

    # 判断是否是多行代码输入
    is_code = (
        first_line.startswith("import ")
        or first_line.startswith("from ")
        or first_line.startswith("def ")
        or first_line.startswith("class ")
        or first_line.startswith("@")
    )

    if is_code:
        # 多行代码
        user_input_lines.append(first_line)
        print("  (继续输入代码，输入 END 结束)")
        while True:
            line = input()
            if line.strip() == "END":
                break
            user_input_lines.append(line)
        user_input = "\n".join(user_input_lines)
        description = input("\n描述（可选，直接回车跳过）> ").strip()
    else:
        # 单行：路径或路径+描述
        user_input = first_line
        # 如果输入中已经包含描述（InputParser 会自动拆分），不再单独问
        description = input("\n描述（可选，直接回车跳过）> ").strip()

    print("\n开始处理...\n")
    print(f"日志目录: {agent.session_log.log_dir}\n")
    result = agent.run(user_input, description)
    return result


def main():
    parser = argparse.ArgumentParser(description="算子任务构建 Demo")
    parser.add_argument("--input", "-i", type=str, help="代码文件/目录路径")
    parser.add_argument("--desc", "-d", type=str, default="", help="代码描述")
    parser.add_argument("--stdin", action="store_true", help="从 stdin 读取代码")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="LLM 模型级别 (complex/standard/fast)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--debug", action="store_true", help="开启调试日志")
    parser.add_argument("--quiet", "-q", action="store_true", help="安静模式")
    args = parser.parse_args()

    setup_logging(args.debug)
    agent = ReactAgent(model_level=args.model, verbose=not args.quiet)

    if args.stdin:
        print("从 stdin 读取代码（Ctrl+D 结束）:")
        code = sys.stdin.read()
        result = agent.run(code, args.desc)
    elif args.input:
        result = agent.run(args.input, args.desc)
    else:
        result = interactive_mode(agent)

    # 输出结果
    print("\n" + "=" * 60)
    print(f"  结果: {result['status']} ({result['steps']} 步)")
    print("=" * 60)

    if result["status"] == "success" and result.get("task_code"):
        task_code = result["task_code"]
        source_path = None  # agent 已保存的文件路径

        # 如果 task_code 是文件路径，读取文件内容
        if task_code.strip().endswith(".py") and len(task_code.strip()) < 300:
            code_path = Path(task_code.strip())
            if not code_path.is_absolute():
                code_path = OUTPUT_DIR / code_path
            if code_path.exists():
                source_path = code_path.resolve()
                task_code = code_path.read_text(encoding="utf-8")

        print("\n--- 生成的任务代码 ---\n")
        print(task_code[:5000])
        if len(task_code) > 5000:
            print(f"\n... (代码共{len(task_code)}字符，已截断显示)")

        # 确定最终保存路径
        if args.output:
            out_path = args.output
        elif source_path:
            # agent 已保存到具体文件，直接使用
            out_path = str(source_path)
        else:
            out_path = str(OUTPUT_DIR / "task_output.py")
        
        # 只在需要时写入（避免重复写入 agent 已保存的文件）
        final_path = Path(out_path).resolve()
        if source_path and final_path == source_path:
            pass  # agent 已保存，不需要重复写入
        else:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text(task_code, encoding="utf-8")
        print(f"\n任务代码已保存到: {final_path}")

        # 保存完整历史
        history_path = Path(out_path).with_suffix(".history.json")
        history_data = {
            "status": result["status"],
            "summary": result.get("summary", ""),
            "steps": result["steps"],
            "history": result["history"],
        }
        history_path.write_text(
            json.dumps(history_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"执行历史: {history_path}")

    else:
        print(f"\n错误: {result.get('error', '未知错误')}")
        if result.get("summary"):
            print(f"摘要: {result['summary']}")

    # 打印日志目录
    if result.get("log_dir"):
        print(f"\n完整日志: {result['log_dir']}")

    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
