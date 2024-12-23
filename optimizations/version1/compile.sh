nvcc main.cu ../../src_parallel/utils/fashion_mnist.cu \
    ../../src_parallel/layer/layer.cu ../../src_parallel/layer/softmax.cu ../../src_parallel/layer/relu.cu layer/dense.cu \
    ../../src_parallel/loss/loss.cu ../../src_parallel/loss/categorical_crossentropy.cu \
    ../../src_parallel/Model/Model.cu \
    ../../src_parallel/metrics/accuracy.cu \
    -o a_parallel.out