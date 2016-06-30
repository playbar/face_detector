LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

$(call import-add-path,$(LOCAL_PATH)/core/src )

LOCAL_MODULE := cv

LOCAL_MODULE_FILENAME := libcv

LOCAL_SRC_FILES := \
core/src/algorithm.cpp \
core/src/alloc.cpp \


LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH) \
                    $(LOCAL_PATH)/. \
                    $(LOCAL_PATH)/core/include \
                    $(LOCAL_PATH)/objdetect/include \
                    $(LOCAL_PATH)/../external/clipper

LOCAL_C_INCLUDES := $(LOCAL_PATH) \
					$(LOCAL_PATH)/core/include \
					$(LOCAL_PATH)/core/src \
					$(LOCAL_PATH)/../include \
                    $(LOCAL_PATH)/../external/tinyxml2 \
                    $(LOCAL_PATH)/../external/clipper

LOCAL_EXPORT_LDLIBS := -lGLESv2 \
                       -llog \
                       -landroid

LOCAL_STATIC_LIBRARIES := cocos_freetype2_static
LOCAL_STATIC_LIBRARIES += cocos_png_static

LOCAL_WHOLE_STATIC_LIBRARIES := cocos2dxandroid_static

# define the macro to compile through support/zip_support/ioapi.c
LOCAL_CFLAGS := -DUSE_FILE32API
LOCAL_CFLAGS += -fexceptions
LOCAL_CFLAGS += -D__OPENCV_BUILD

LOCAL_CPPFLAGS := -Wno-deprecated-declarations -Wno-extern-c-compat
LOCAL_EXPORT_CFLAGS   := -DUSE_FILE32API
LOCAL_EXPORT_CPPFLAGS := -Wno-deprecated-declarations -Wno-extern-c-compat

include $(BUILD_STATIC_LIBRARY)

#==============================================================

include $(CLEAR_VARS)

LOCAL_MODULE := novo2d_static
LOCAL_MODULE_FILENAME := libnovo2d

LOCAL_STATIC_LIBRARIES += novo3d_static
LOCAL_STATIC_LIBRARIES += spine_static
LOCAL_STATIC_LIBRARIES += audioengine_static

include $(BUILD_STATIC_LIBRARY)

#==============================================================
$(call import-module,zlib)
#$(call import-module,platform/android)

