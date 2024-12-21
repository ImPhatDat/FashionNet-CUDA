nvcc src/main.cu src/fashion_mnist.cu \
    src/layer/layer.cu src/layer/softmax.cu src/layer/relu.cu src/layer/dense.cu \
    src/loss/loss.cu src/loss/categorical_crossentropy.cu \
    src/Model/Model.cu \
    src/metrics/accuracy.cu \
    -o a.out \
    && ./a.out