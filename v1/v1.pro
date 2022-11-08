QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
    deduction.cpp \
    detection.cpp \
    main.cpp \
    mainwin.cpp \
    nms.cpp \
    nms_rotated_cpu.cpp \
    optdeduction.cpp \
    probability.cpp \
    roi_align_rotated_cpu.cpp \
    sardeduction.cpp \
    util.cpp

HEADERS += \
    box_iou_rotated_utils.h \
    deduction.h \
    detection.h \
    mainwin.h \
    nms.h \
    nms_rotated.h \
    optdeduction.h \
    probability.h \
    roi_align_rotated_cpu.h \
    sardeduction.h \
    util.h

FORMS += \
    deduction.ui \
    detection.ui \
    mainwin.ui \
    optdeduction.ui \
    probability.ui \
    sardeduction.ui


#
MY_Libtorch_PATH = /home/ckq/software/libtorch1.7.1/
# 添加OpenCV 和 libtorch的路径
INCLUDEPATH +=  /usr/local/include \
                /usr/local/include/opencv4 \
                /usr/local/include/opencv4/opencv2 \
                $${MY_Libtorch_PATH}include \
                $${MY_Libtorch_PATH}include/torch/csrc/api/include
#                /home/ckq/software/libtorch/include \
#                /home/ckq/software/libtorch/include/torch/csrc/api/include
#/home/ckq/Downloads/pre_libtorch/include/ \
#/home/ckq/Downloads/pre_libtorch/include/torch/csrc/api/include/
#/home/ckq/software/libtorch/include/torch/csrc/api/include/ \
#/home/ckq/software/libtorch/include/

#QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
#CONFIG += no_keywords


#QMAKE_LFLAGS += -Wl,--no-as-needed
#QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX14_ABI=0
QMAKE_LFLAGS += -INCLUDE:?warp_size@cuda@at@@YAHXZ
#QMAKE_LFLAGS += -INCLUDE:?searchsorted_cuda@native@at@@YA?AVTensor@2@AEBV32@0_N1@Z
#QMAKE_LFLAGS += -INCLUDE:"?ignore_this_library_placeholder@@YAHXZ"


#QMAKE_LIBDIR += /home/ckq/software/libtorch/lib
QMAKE_LIBDIR += $${MY_Libtorch_PATH}/lib
QMAKE_LFLAGS += -Wl,-rpath=$${MY_Libtorch_PATH}/lib/


LIBS += /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_highgui.so.4.5 \
        /usr/local/lib/libopencv_core.so    \
        /usr/local/lib/libopencv_imgproc.so \
        /usr/local/lib/libopencv_imgcodecs.so



INCLUDEPATH +=  /usr/local/cuda/include
#LIBS += /home/ckq/software/libtorch/lib/libtorch.so \
#        /home/ckq/software/libtorch/lib/libtorch_cuda.so \
#        /usr/local/cuda/lib64/libcudnn.so
LIBS += $${MY_Libtorch_PATH}lib/libtorch.so \
        $${MY_Libtorch_PATH}lib/libtorch_cuda.so \
        /usr/local/cuda/lib64/libcudnn.so

#LIBS += /home/ckq/software/libtorch/lib/libc10.so \
#        /home/ckq/software/libtorch/lib/libkineto.a \
#        /usr/local/cuda/lib64/stubs/libcuda.so \
#        /usr/local/cuda/lib64/libnvrtc.so \
#        /usr/local/cuda/lib64/libnvToolsExt.so \
#        /usr/local/cuda/lib64/libcudart.so \
#        /home/ckq/software/libtorch/lib/libc10_cuda.so
LIBS += $${MY_Libtorch_PATH}lib/libc10.so \
        /usr/local/cuda/lib64/stubs/libcuda.so \
        /usr/local/cuda/lib64/libnvrtc.so \
        /usr/local/cuda/lib64/libnvToolsExt.so \
        /usr/local/cuda/lib64/libcudart.so \
        $${MY_Libtorch_PATH}/lib/libc10_cuda.so

#LIBS += -L"/home/ckq/Downloads/pre_libtorch/lib" \
#LIBS += -L"/home/ckq/software/libtorch/lib" \
##        -lbackend_with_compiler \
#        -lc10_cuda \
#        -lc10d_cuda_test \
#        -lc10 \
#        -lcaffe2_nvrtc \
#        -ljitbackend_test \
##        -lnnapi_backend \
#        -lnvrtc-builtins \
#        -lshm \
#        -ltorchbind_test \
#        -ltorch_cpu \
##        -ltorch_cuda_linalg \
#        -ltorch_cuda \
#        -ltorch_global_deps \
#        -ltorch_python \
#        -ltorch
LIBS += -L"$${MY_Libtorch_PATH}lib" \
#        -lbackend_with_compiler \
        -lc10_cuda \
        -lc10d_cuda_test \
        -lc10 \
        -lcaffe2_nvrtc \
        -ljitbackend_test \
#        -lnnapi_backend \
        -lnvrtc-builtins \
        -lshm \
        -ltorchbind_test \
        -ltorch_cpu \
#        -ltorch_cuda_linalg \
        -ltorch_cuda \
        -ltorch_global_deps \
        -ltorch_python \
        -ltorch


#LIBS += /home/ckq/software/libtorch/lib/libcublas-1e3c0411.so.10 \
#        /home/ckq/software/libtorch/lib/libcublasLt-a1cbff2e.so.10 \
#        /home/ckq/software/libtorch/lib/libcudart-80664282.so.10.2 \
#        /home/ckq/software/libtorch/lib/libnvToolsExt-3965bdd0.so.1


#LIBS += -L"/home/ckq/software/libtorch/lib" \
#        -lasmjit \
##        -lkineto \
#        -lnnpack \
#        -lnnpack_reference_layers \
#        -lbenchmark \
#        -lbenchmark_main \
#        -lcaffe2_protos \
#        -lonnx \
#        -lonnx_proto \
#        -lclog \
#        -lprotobuf \
#        -lcpuinfo   \
#        -lprotobuf-lite \
#        -lcpuinfo_internals \
#        -lprotoc \
#        -lpthreadpool \
#        -lpytorch_qnnpack \
#        -lqnnpack \
#        -ldnnl \
##        -ldnnl_graph \
#        -lfbgemm \
#        -lfmt \
#        -lfoxi_loader \
#        -lgloo \
#        -lgloo_cuda \
#        -lgmock \
#        -lgmock_main \
#        -lgtest \
#        -lgtest_main \
#        -ltensorpipe \
##        -ltensorpipe_cuda \
#        -ltensorpipe_uv
LIBS += -L"$${MY_Libtorch_PATH}lib" \
        -lasmjit \
#        -lkineto \
        -lnnpack \
        -lnnpack_reference_layers \
        -lbenchmark \
        -lbenchmark_main \
        -lcaffe2_protos \
        -lonnx \
        -lonnx_proto \
        -lclog \
        -lprotobuf \
        -lcpuinfo   \
        -lprotobuf-lite \
        -lcpuinfo_internals \
        -lprotoc \
        -lpthreadpool \
        -lpytorch_qnnpack \
        -lqnnpack \
        -ldnnl \
#        -ldnnl_graph \
        -lfbgemm \
        -lfmt \
        -lfoxi_loader \
        -lgloo \
        -lgloo_cuda \
        -lgmock \
        -lgmock_main \
        -lgtest \
        -lgtest_main \
        -ltensorpipe \
#        -ltensorpipe_cuda \
        -ltensorpipe_uv





# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

#DISTFILES +=
