import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from akg_agents.op.verifier.kernel_verifier import KernelVerifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_batch(args):
    base_dir = Path(__file__).parent.parent.parent / "thirdparty" / "sol-execbench" / "data" / "benchmark"
    level_dir = base_dir / args.level
    
    if not level_dir.exists():
        logger.error(f"SOL Bench data directory not found: {level_dir}")
        logger.error("Please run 'bash download.sh --with_sol_execbench' first.")
        return
        
    code_dir = Path(args.code_dir)
    if not code_dir.exists():
        logger.error(f"Code directory not found: {code_dir}")
        return
        
    cases = sorted([d for d in level_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(cases)} cases in {args.level}")
    
    passed_count = 0
    failed_count = 0
    missing_count = 0
    
    for case_dir in cases:
        op_name = case_dir.name
        # 假设生成的代码文件名为 {op_name}_{dsl}.py
        code_file = code_dir / f"{op_name}_{args.dsl}.py"
        
        if not code_file.exists():
            # 尝试其他命名约定
            code_file = code_dir / f"{op_name}.py"
            if not code_file.exists():
                logger.warning(f"[{op_name}] Missing code file, skipping.")
                missing_count += 1
                continue
                
        with open(code_file, "r", encoding="utf-8") as f:
            impl_code = f.read()
            
        config = {
            "log_dir": args.log_dir,
            "sol_problem_dir": str(case_dir),
            "verify_timeout": args.timeout
        }
        
        verifier = KernelVerifier(
            op_name=op_name,
            framework_code="", # SOL 模式下不需要
            framework="torch",
            dsl=args.dsl,
            backend=args.backend,
            arch=args.arch,
            config=config,
            bench_type="sol"
        )
        
        task_info = {"coder_code": impl_code}
        
        logger.info(f"[{op_name}] Starting verification...")
        try:
            passed, log = await verifier.run(task_info, device_id=args.device_id)
            if passed:
                logger.info(f"[{op_name}] ✅ PASSED")
                passed_count += 1
            else:
                logger.error(f"[{op_name}] ❌ FAILED\n{log[:500]}...") # 只打印前500个字符的日志
                failed_count += 1
        except Exception as e:
            logger.error(f"[{op_name}] ❌ ERROR: {e}")
            failed_count += 1
            
    logger.info("="*50)
    logger.info(f"Batch Verification Summary for {args.level}")
    logger.info(f"Total: {len(cases)} | Passed: {passed_count} | Failed: {failed_count} | Missing Code: {missing_count}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run SOL-ExecBench cases")
    parser.add_argument("--level", type=str, default="L1", choices=["L1", "L2", "Quant", "FlashInfer-Bench"], help="SOL Bench level")
    parser.add_argument("--code-dir", type=str, required=True, help="Directory containing the generated kernel codes")
    parser.add_argument("--dsl", type=str, default="triton_cuda", help="DSL type (e.g., triton_cuda, cpp, ascendc)")
    parser.add_argument("--backend", type=str, default="cuda", choices=["cuda", "ascend", "cpu"], help="Backend type")
    parser.add_argument("--arch", type=str, default="a100", help="Architecture type")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID to use")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per case in seconds")
    parser.add_argument("--log-dir", type=str, default="~/akg_agents_logs/sol_batch", help="Directory to save logs")
    
    args = parser.parse_args()
    
    args.log_dir = os.path.expanduser(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    asyncio.run(run_batch(args))
