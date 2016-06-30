LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := zlib
LOCAL_MODULE_FILENAME := zlib
LOCAL_SRC_FILES := \
adler32.c


LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/../../include/android \
                         $(LOCAL_PATH)/../../include/android/freetype2
include $(BUILD_STATIC_LIBRARY)
