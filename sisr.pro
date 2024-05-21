#-------------------------------------------------
#
# Project created by QtCreator 2023-05-12T19:27:23
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

QMAKE_CXXFLAGS+= -openmp
QMAKE_LFLAGS += -openmp

TARGET = sisr
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        main.cpp \
        mainwindow.cpp \
        sisr.cpp

HEADERS += \
        mainwindow.h \
        sisr.h

FORMS += \
        mainwindow.ui

INCLUDEPATH  +=  C:\libs\opencv\build\install\include

release {
    LIBS  +=  C:\libs\opencv\build\install\x64\vc16\lib\opencv_core470.lib
    LIBS  +=  C:\libs\opencv\build\install\x64\vc16\lib\opencv_highgui470.lib
    LIBS  +=  C:\libs\opencv\build\install\x64\vc16\lib\opencv_imgcodecs470.lib
    LIBS  +=  C:\libs\opencv\build\install\x64\vc16\lib\opencv_imgproc470.lib
}

debug {
    LIBS  +=  C:\libs\opencv\build\install\x64\vc16\lib\opencv_core470d.lib
    LIBS  +=  C:\libs\opencv\build\install\x64\vc16\lib\opencv_highgui470d.lib
    LIBS  +=  C:\libs\opencv\build\install\x64\vc16\lib\opencv_imgcodecs470d.lib
    LIBS  +=  C:\libs\opencv\build\install\x64\vc16\lib\opencv_imgproc470d.lib
}


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
