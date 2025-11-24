class AIKG_swiglu_quant_op {
public:
    AIKG_swiglu_quant_op() = default;
    ~AIKG_swiglu_quant_op() = default;

    AikgStatus Init() {
        return AikgStatus::Success;
    }

    std::vector<int> InferShape(const OpArgs& args) {
        // Input shape is (M, 2*N)
        // Output shapes:
        //   output: (M, N) int8
        //   scale: (M,) float32
        //   swiglu_out: (M, N) float16
        
        std::vector<int> output_shapes;
        if (args.tensors.size() < 2) {
            return output_shapes;
        }
        
        const auto& input_shape = args.tensors[0].shape;
        if (input_shape.size() != 2) {
            return output_shapes;
        }
        
        int M = input_shape[0];
        int N = input_shape[1] / 2;
        
        // output shape
        output_shapes.push_back(M);
        output_shapes.push_back(N);
        
        // scale shape
        output_shapes.push_back(M);
        
        // swiglu_out shape
        output_shapes.push_back(M);
        output_shapes.push_back(N);
        
        return output_shapes;
    }

    int GetWorkspaceSize() {
        return workspace_size_;
    }

    int GetBlockDim() {
        return block_dim_;
    }

    AikgStatus Tiling(const OpArgs& args) {
        tiling_param_.clear();
        
        // From AUL code, we can see:
        // BLOCK_DIM = 8
        // M = 1024, N = 3584 (these are likely example values)
        // per_core_rows = M // BLOCK_DIM
        
        // Get actual input shape
        if (args.tensors.empty()) {
            return AikgStatus::UnsupportedDataType;
        }
        
        const auto& input_shape = args.tensors[0].shape;
        if (input_shape.size() != 2) {
            return AikgStatus::UnsupportedDataType;
        }
        
        int M = input_shape[0];
        int N = input_shape[1] / 2;  // Split into x0 and x1
        
        // Set block_dim to 8 as in AUL code
        block_dim_ = 8;
        
        // Calculate per_core_rows
        int per_core_rows = M / block_dim_;
        if (M % block_dim_ != 0) {
            per_core_rows += 1;
        }
        
        // Add tiling parameters
        tiling_param_.push_back(M);
        tiling_param_.push_back(N);
        tiling_param_.push_back(block_dim_);
        tiling_param_.push_back(per_core_rows);
        
        // No workspace needed as per AUL code
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
    std::vector<int64_t> tiling_param_; // Contains [M, N, block_dim, per_core_rows]
};