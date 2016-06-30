LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

#$(call import-add-path,$(LOCAL_PATH)/../../modules)
#$(call import-add-path,$(LOCAL_PATH)/../../3rdparty)

include ../native/jni/OpenCV.mk

LOCAL_SRC_FILES  := DetectionBasedTracker_jni.cpp
LOCAL_C_INCLUDES += $(LOCAL_PATH)
LOCAL_LDLIBS     += -llog -ldl
LOCAL_MODULE     := facedetector

#LOCAL_SRC_FILES :=  libopencv_java.so
#LOCAL_MODULE := opencv_java

LOCAL_STATIC_LIBRARIES := cv

include $(BUILD_SHARED_LIBRARY)

#$(call import-module, .)

include $(CLEAR_VARS)
LOCAL_MODULE :=opencv_java3
LOCAL_SRC_FILES := libopencv_java3.so
include $(PREBUILT_SHARED_LIBRARY)

