LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
include ../native/jni/OpenCV.mk

LOCAL_SRC_FILES  := DetectionBasedTracker_jni.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_LDLIBS     += -llog -ldl
LOCAL_MODULE     := facedetector

#LOCAL_SRC_FILES :=  libopencv_java.so
#LOCAL_MODULE := opencv_java



include $(BUILD_SHARED_LIBRARY)
