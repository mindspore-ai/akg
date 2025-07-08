class AIKG_exp_adds_op {
public:
  AIKG_exp_adds_op() = default;
  ~AIKG_exp_adds_op() = default;

  AikgStatus Init() {
    // No initialization needed for this op
    return AikgStatus::Success;
  }

  std::vector<int> InferShape(const OpArgs &args) {
    // Output shape is same as input shape
    if (args.tensors.empty()) {
      return {};
    }
    return args.tensors[0].shape; // Assuming first tensor is input
  }

  int GetWorkspaceSize() { return workspace_size_; }

  int GetBlockDim() { return block_dim_; }

  AikgStatus Tiling(const OpArgs &args) {
    if (args.tensors.empty()) {
      return AikgStatus::UnsupportedDataType;
    }

    // Based on numpy implementation, we parallelize across rows (first
    // dimension)
    const auto &input_shape = args.tensors[0].shape;
    if (input_shape.size() != 2) {
      return AikgStatus::UnsupportedDataType; // Only support 2D inputs
    }

    // Each vector core processes one row (as in numpy implementation)
    block_dim_ = 8;
    return AikgStatus::Success;
  }

  AikgStatus SetTiling(uint8_t *host_ptr) {
    if (tiling_param_.empty()) {
      return AikgStatus::Success;
    }
    auto tiling_size_ = tiling_param_.size() * sizeof(int64_t);
    memcpy(host_ptr, tiling_param_.data(), tiling_size_);
    return AikgStatus::Success;
  }

  int GetTilingSize() { return tiling_param_.size() * sizeof(int64_t); }

private:
    int block_dim_ = 0;
    int workspace_size_ = 0;
    std::vector<int64_t> tiling_param_;
};