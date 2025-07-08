class AIKG_reduce_sum_op {
public:
    AIKG_reduce_sum_op() = default;
    ~AIKG_reduce_sum_op() = default;

    AikgStatus Init() {
        return AikgStatus::Success;
    }

    std::vector<int> InferShape(const OpArgs& args) {
        // Input shape is (64, 128), output shape is (64,) after reduce_sum along last axis
        if (args.tensors.empty()) {
            return {};
        }
        const auto& input_shape = args.tensors[0].shape;
        if (input_shape.size() != 2) {
            return {};
        }
        return {input_shape[0]};
    }

    int GetWorkspaceSize() {
        return workspace_size_;
    }

    int GetBlockDim() {
        return block_dim_;
    }

    AikgStatus Tiling(const OpArgs& args) {
        tiling_param_.clear();
        
        // From AUL code: BLOCK_DIM = 8, total_rows = 64, cols = 128
        const int BLOCK_DIM = 8;
        const int total_rows = 64;
        const int cols = 128;
        
        // Calculate rows per core
        const int rows_per_core = total_rows / BLOCK_DIM;
        
        // Set tiling parameters (order must match kernel expectation)
        tiling_param_.push_back(total_rows);   // Total rows
        tiling_param_.push_back(cols);         // Columns
        tiling_param_.push_back(rows_per_core); // Rows per core
        
        // Set block dim and workspace size (from AUL: WORKSPACE_SIZE = 0)
        block_dim_ = BLOCK_DIM;
        workspace_size_ = 0;
        
        return AikgStatus::Success;
    }

    AikgStatus SetTiling(uint8_t* host_ptr) {
        if (tiling_param_.empty()) {
            return AikgStatus::Success;
        }
        auto tiling_size_ = tiling_param_.size() * sizeof(int64_t);
        memcpy(host_ptr, tiling_param_.data(), tiling_size_);
        return AikgStatus::Success;
    }

    int GetTilingSize() {
        return tiling_param_.size() * sizeof(int64_t);
    }

private:
    int block_dim_ = 0;
    int workspace_size_ = 0;
    std::vector<int64_t> tiling_param_;
};