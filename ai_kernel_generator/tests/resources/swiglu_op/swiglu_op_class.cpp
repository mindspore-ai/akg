class AIKG_swiglu_op {
public:
    AIKG_swiglu_op() = default;
    ~AIKG_swiglu_op() = default;

    AikgStatus Init() {
        return AikgStatus::Success;
    }

    std::vector<int> InferShape(const OpArgs& args) {
        // Input shape is (40, 256), output shape is (40, 128)
        if (args.tensors.empty()) {
            return {};
        }
        auto input_shape = args.tensors[0].shape;
        if (input_shape.size() != 2) {
            return {};
        }
        return {input_shape[0], input_shape[1] / 2};
    }

    int GetWorkspaceSize() {
        return workspace_size_;
    }

    int GetBlockDim() {
        return block_dim_;
    }

    AikgStatus Tiling(const OpArgs& args) {
        tiling_param_.clear();
        
        // Extract tiling parameters from AUL implementation
        const int dim0 = 40;       // Total rows
        const int dim1 = 256;      // Total columns
        const int dim0_split = 5;  // Rows per core
        const int dim1_split = 128; // Columns split point
        
        // Calculate block dimension (number of cores needed)
        block_dim_ = dim0 / dim0_split;
        
        // Calculate workspace size (no workspace needed in this case)
        workspace_size_ = 0;
        
        // Pack tiling parameters into vector
        tiling_param_.push_back(dim0);
        tiling_param_.push_back(dim1);
        tiling_param_.push_back(dim0_split);
        tiling_param_.push_back(dim1_split);
        
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