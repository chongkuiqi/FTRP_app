QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    deduction.cpp \
    detect.cpp \
    detection.cpp \
    main.cpp \
    mainwin.cpp \
    probability.cpp

HEADERS += \
    deduction.h \
    detect.h \
    detection.h \
    mainwin.h \
    probability.h

FORMS += \
    deduction.ui \
    detection.ui \
    mainwin.ui \
    probability.ui

# 添加OpenCV 和 libtorch的路径
INCLUDEPATH +=  /usr/local/include\
                /usr/local/include/opencv4 \
                /usr/local/include/opencv4/opencv2 \
                /home/ckq/software/libtorch/include/ \
                /home/ckq/software/libtorch/include/torch/csrc/api/include/
#                /home/ckq/miniconda3/envs/ckq/lib/python3.8/site-packages/torch/include/ \
#                /home/ckq/miniconda3/envs/ckq/lib/python3.8/site-packages/torch/include/torch/csrc/api/include/ \


#LIBS += /usr/local/lib/libopencv_highgui.so \
#        /usr/local/lib/libopencv_highgui.so.4.5 \
#        /usr/local/lib/libopencv_core.so    \
#        /usr/local/lib/libopencv_imgproc.so \
#        /usr/local/lib/libopencv_imgcodecs.so \
#        -L"/home/ckq/miniconda3/envs/ckq/lib/python3.8/site-packages/torch/lib/" \
#        -lc10 \
#        -lc10_cuda \
#        -ltorch \
#        -ltorch_cpu \
#        -ltorch_cuda \
#        -lcaffe2_nvrtc \
#        -lshm \
#        -lcaffe2_detectron_ops_gpu \
#        -lcaffe2_module_test_dynamic \
#        -ltorch_global_deps \
#        -lcaffe2_observers \
#        -lnvrtc-builtins

#QMAKE_LFLAGS += -INCLUDE:?warp_size@cuda@at@@YAHXZ
#QMAKE_LFLAGS += -INCLUDE:?searchsorted_cuda@native@at@@YA?AVTensor@2@AEBV32@0_N1@Z
#QMAKE_LFLAGS += -Wl,--no-as-needed

#QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX14_ABI=0
QMAKE_LFLAGS += -INCLUDE:?warp_size@cuda@at@@YAHXZ
#QMAKE_LFLAGS += -INCLUDE:?searchsorted_cuda@native@at@@YA?AVTensor@2@AEBV32@0_N1@Z
#QMAKE_LFLAGS += -INCLUDE:"?ignore_this_library_placeholder@@YAHXZ"

#-INCLUDE:?warp_size@cuda@at@@YAHXZ \
#-INCLUDE:?searchsorted_cuda@native@at@@YA?AVTensor@2@AEBV32@0_N1@Z \
#-INCLUDE:"?ignore_this_library_placeholder@@YAHXZ" \

QMAKE_LIBDIR += /home/ckq/software/libtorch/lib

LIBS += /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_highgui.so.4.5 \
        /usr/local/lib/libopencv_core.so    \
        /usr/local/lib/libopencv_imgproc.so \
        /usr/local/lib/libopencv_imgcodecs.so \
        -L"/home/ckq/software/libtorch/lib" \
        -lc10 \
        -lc10_cuda \
        -lc10d_cuda_test \
        -lfbgemm \
        -ltorch \
        -ltorch_cpu \
        -ltorch_cuda \
        -lasmjit \
        -lclog \
        -lpthreadpool \
        -lgloo_cuda \
        -ldnnl \
        -lcpuinfo \
        -lXNNPACK \
        -ljitbackend_test \
        -lnnapi_backend \
        -ltorch_cuda_linalg \
        -lbackend_with_compiler \
        -lcaffe2_nvrtc \
        -lcaffe2_protos \
        -lgloo_cuda \
        -lgloo







#-INCLUDE:?warp_size@cuda@at@@YAHXZ
#-INCLUDE:?searchsorted_cuda@native@at@@YA?AVTensor@2@AEBV32@0_N1@Z \
#-lcublas-1e3c0411 \
#-libcublasLt-a1cbff2e.so.10 \
#-libcudart-80664282.so.10.2 \
#-libnvrtc-08c4863f.so.10.2 \
#-libnvrtc-builtins.so \
#-libnvToolsExt-3965bdd0.so.1 \
#-libshm.so \
#-libtorchbind_test.so \
#-libtorch_global_deps.so \

#        -lmkldnn \
#        -llibprotoc \
#        -llibprotobuf-lite \
#        -lcaffe2_detection_ops_gpu \
#        -lcaffe2_module_test_dynamic



# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
