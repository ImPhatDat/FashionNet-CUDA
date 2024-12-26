nvcc src_parallel/infer.cu src_parallel/utils/fashion_mnist.cu \
    src_parallel/layer/layer.cu src_parallel/layer/softmax.cu src_parallel/layer/relu.cu src_parallel/layer/dense.cu \
    src_parallel/loss/loss.cu src_parallel/loss/categorical_crossentropy.cu \
    src_parallel/Model/Model.cu \
    src_parallel/metrics/accuracy.cu \
    -o infer_parallel.out