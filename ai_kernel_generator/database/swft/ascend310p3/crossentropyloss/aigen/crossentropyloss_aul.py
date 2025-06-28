import aul as U

def crossentropyloss_op_impl_npu(input_np: U.TensorPtr, target_np: U.TensorPtr, output: U.TensorPtr, tiling: list):
    # 硬编码参数（来自Tiling函数）
    BATCH_SIZE = 4096
    NUM_CLASSES = 10
    TILE_BATCH_SIZE = 64
    LOOP_COUNT = BATCH_SIZE // TILE_BATCH_SIZE
    
    # 获取当前核ID
    core_idx = U.get_core_idx()
    
    # 仅核0执行计算（单核模式）
    if core_idx != 0:
        return
    
    # 创建Tile用于输入和目标
    input_tile = U.Tile(shape=(TILE_BATCH_SIZE, NUM_CLASSES), dtype=U.float16, pos=U.VecBuf)
    target_tile = U.Tile(shape=(TILE_BATCH_SIZE,), dtype=U.int64, pos=U.VecBuf)
    
    # 创建中间结果Tile
    max_vals_tile = U.Tile(shape=(TILE_BATCH_SIZE, 1), dtype=U.float16, pos=U.VecBuf)
    shifted_tile = U.Tile(shape=(TILE_BATCH_SIZE, NUM_CLASSES), dtype=U.float16, pos=U.VecBuf)
    exp_shifted_tile = U.Tile(shape=(TILE_BATCH_SIZE, NUM_CLASSES), dtype=U.float16, pos=U.VecBuf)
    sum_exp_tile = U.Tile(shape=(TILE_BATCH_SIZE, 1), dtype=U.float16, pos=U.VecBuf)
    log_softmax_tile = U.Tile(shape=(TILE_BATCH_SIZE, NUM_CLASSES), dtype=U.float16, pos=U.VecBuf)
    selected_tile = U.Tile(shape=(TILE_BATCH_SIZE,), dtype=U.float16, pos=U.VecBuf)
    
    # 创建log_sum_exp_tile（移到循环外）
    log_sum_exp_tile = U.Tile(shape=(TILE_BATCH_SIZE, 1), dtype=U.float16, pos=U.VecBuf)
    
    # 创建索引和值Tile（移到循环外）
    index_tile = U.Tile(shape=(1,), dtype=U.int64, pos=U.VecBuf)
    value_tile = U.Tile(shape=(1,), dtype=U.float16, pos=U.VecBuf)
    
    # 创建累加器Tile
    accumulator = U.Tile(shape=(1,), dtype=U.float32, pos=U.VecBuf)
    U.vunary_op(op="vector_dup", dst=accumulator, fill_shape=[1], fill_value=0.0)
    
    # 流水线处理每个tile
    for i in U.Pipelined(iterations=LOOP_COUNT):
        start_idx = i * TILE_BATCH_SIZE
        end_idx = start_idx + TILE_BATCH_SIZE
        
        # 加载数据
        U.data_copy(dst=input_tile, src=input_np[start_idx:end_idx, 0:NUM_CLASSES], 
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        U.data_copy(dst=target_tile, src=target_np[start_idx:end_idx], 
                    src_pos=U.GlobalMem, dst_pos=U.VecBuf)
        
        # 计算每行最大值
        U.vreduce_op(op="max", dst=max_vals_tile, src=input_tile, axis=-1)
        
        # 减去最大值
        U.vbinary_op(op="sub", dst=shifted_tile, src1=input_tile, src2=max_vals_tile)
        
        # 计算指数
        U.vunary_op(op="exp", dst=exp_shifted_tile, src=shifted_tile)
        
        # 计算指数和
        U.vreduce_op(op="sum", dst=sum_exp_tile, src=exp_shifted_tile, axis=-1)
        
        # 计算log_softmax
        U.vunary_op(op="ln", dst=log_sum_exp_tile, src=sum_exp_tile)
        U.vbinary_op(op="sub", dst=log_softmax_tile, src1=shifted_tile, src2=log_sum_exp_tile)
        
        # 选择目标值
        for j in range(TILE_BATCH_SIZE):
            U.data_copy(dst=index_tile, src=target_tile[j:j+1], src_pos=U.VecBuf, dst_pos=U.VecBuf)
            U.data_copy(dst=value_tile, src=log_softmax_tile[j:j+1, index_tile[0]:index_tile[0]+1], 
                        src_pos=U.VecBuf, dst_pos=U.VecBuf)
            U.data_copy(dst=selected_tile[j:j+1], src=value_tile, src_pos=U.VecBuf, dst_pos=U.VecBuf)
        
        # 转换为float32并累加
        selected_fp32_tile = U.Tile(shape=(TILE_BATCH_SIZE,), dtype=U.float32, pos=U.VecBuf)
        U.vunary_op(op="cast_fp16_to_fp32", dst=selected_fp32_tile, src=selected_tile)
        
        sum_selected_tile = U.Tile(shape=(1,), dtype=U.float32, pos=U.VecBuf)
        U.vreduce_op(op="sum", dst=sum_selected_tile, src=selected_fp32_tile, axis=-1)
        
        U.vbinary_op(op="add", dst=accumulator, src1=accumulator, src2=sum_selected_tile)
    
    # 计算最终损失
    total_batch_tile = U.Tile(shape=(1,), dtype=U.float32, pos=U.VecBuf)
    U.vunary_op(op="vector_dup", dst=total_batch_tile, fill_shape=[1], fill_value=float(BATCH_SIZE))
    
    loss_tile = U.Tile(shape=(1,), dtype=U.float32, pos=U.VecBuf)
    U.vbinary_op(op="div", dst=loss_tile, src1=accumulator, src2=total_batch_tile)
    U.vectorscalar_op(op="muls", dst=loss_tile, src=loss_tile, factor=-1.0)
    
    loss_fp16_tile = U.Tile(shape=(1,), dtype=U.float16, pos=U.VecBuf)
    U.vunary_op(op="cast_fp32_to_fp16", dst=loss_fp16_tile, src=loss_tile)
    
    # 写回结果
    U.data_copy(dst=output, src=loss_fp16_tile, src_pos=U.VecBuf, dst_pos=U.GlobalMem)

def crossentropyloss_op_impl_tiling(input_np, target_np, output_np, tiling):
    BLOCK_DIM = 1
    WORKSPACE_SIZE = 0




class AgentTaskPool:
    input:
        kernel_code
        host_code
        deviced_id
        op_task???? ==> data

        # export DEVICE_ID=2; python task.py

    output:
        async run: pass/fialed ---->log(error)-->***.log

        async profiling: task ms




    def __init__(self, max_concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.tasks: Dict[str, asyncio.Task] = {}
        self.results: Dict[str, Any] = {}
    ...


# 使用示例
async def main():
    # 创建任务池
    task_pool = AgentTaskPool(max_concurrency=3)  # 最多3个并发任务
    
    # 准备多个任务
    task_ids = []
    for i in range(5):  # 创建5个任务
        
        # 创建任务
        task = Task(
            agents=agents,
            initial_input={"requirements": f"实现排序算法 {i}"},

        )
        
        # 提交任务到池中
        task_id = await task_pool.submit_task(task)
        task_ids.append(task_id)
        print(f"Submitted task {task_id}")
    
    # 方式1：等待所有任务完成
    all_results = await task_pool.wait_all()
    
    # 方式2：逐个获取任务结果
    for task_id in task_ids:
        try:
            result = await task_pool.get_result(task_id, timeout=30)
            print(f"Task {task_id} completed with result: {result}")
        except TimeoutError:
            print(f"Task {task_id} timed out")
        except Exception as e:
            print(f"Task {task_id} failed: {str(e)}")
