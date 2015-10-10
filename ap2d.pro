QT += core
QT -= gui

TARGET = ap2d
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    ap2d.cpp

LIBS += -lfftw3

HEADERS += \
    ap2d.h

INCLUDEPATH += mdaio
DEPENDPATH += mdaio
VPATH += mdaio
HEADERS += mda.h mdaio.h usagetracking.h
SOURCES += mda.cpp mdaio.cpp usagetracking.cpp

HEADERS += parse_command_line_params.h
SOURCES += parse_command_line_params.cpp

QMAKE_LFLAGS += -fopenmp
QMAKE_CXXFLAGS += -fopenmp -std=c++11
LIBS += -fopenmp
