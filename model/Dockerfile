FROM pacpaint-exec-v2-pt-2110

RUN pip uninstall -y torch
ENV TORCH_CUDA_ARCH_LIST="8.6"
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
