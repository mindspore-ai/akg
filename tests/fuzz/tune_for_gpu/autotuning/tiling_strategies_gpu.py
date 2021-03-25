from akg.utils import custom_tiling as ct_util

def reduce_gpu_tiling_strategy(in_shape, reduce_axis):
    """Custom tiling strategy for reduce op in gpu"""
    strategy = list()

    if reduce_axis == None or len(reduce_axis) == len(in_shape):
        """all-reduce"""
        strategy.append(
            ct_util.create_constraint_on_axis(
                values=32, constraints=ct_util.TileConstraint.MOD, band=0, axis=0
            )[0]
        )
        strategy.append(
            ct_util.modify_common_constraints(
                value=[32, 1, 1], constraint=ct_util.TileConstraint.THREAD_MOD
            )
        )
        strategy.append(
            ct_util.modify_common_constraints(
                value=[1024, 1, 1], constraint=ct_util.TileConstraint.THREAD_MAX
            )
        )
        strategy.append(
            ct_util.modify_common_constraints(
                value=[32, 1, 1], constraint=ct_util.TileConstraint.THREAD_MIN
            )
        )
    elif (len(in_shape) - 1) in reduce_axis:
        """Reduce-X: dummy strategy for hand-write space"""
        strategy.append(
            ct_util.create_constraint_on_axis(
                values=1, constraints=ct_util.TileConstraint.MAX, band=0, axis=0
            )[0]
        )
        strategy.append(
            ct_util.create_constraint_on_axis(
                values=1, constraints=ct_util.TileConstraint.MAX, band=0, axis=1
            )[0]
        )
        strategy.append(
            ct_util.modify_common_constraints(
                value=[1, 1, 1], constraint=ct_util.TileConstraint.THREAD_MAX
            )
        )
        strategy.append(
            ct_util.modify_common_constraints(
                value=[1, 1, 1], constraint=ct_util.TileConstraint.BLOCK_MAX
            )
        )        

    else:
        """Reduce-Y: dummy strategy for hand-write space"""
        strategy.append(
            ct_util.create_constraint_on_axis(
                values=1, constraints=ct_util.TileConstraint.MAX, band=0, axis=0
            )[0]
        )
        strategy.append(
            ct_util.create_constraint_on_axis(
                values=1, constraints=ct_util.TileConstraint.MAX, band=0, axis=1
            )[0]
        )
        strategy.append(
            ct_util.modify_common_constraints(
                value=[1, 1, 1], constraint=ct_util.TileConstraint.THREAD_MAX
            )
        )
        strategy.append(
            ct_util.modify_common_constraints(
                value=[1, 1, 1], constraint=ct_util.TileConstraint.BLOCK_MAX
            )
        )

    return strategy


def conv_dummy_strategy():
    """Conv strategy: dummy strategy"""
    return 

def batch_matmul_gpu_tiling_strategy(desc):
    """Custom tiling strategy for batch matmul in gpu with or without tensor core"""
    return 