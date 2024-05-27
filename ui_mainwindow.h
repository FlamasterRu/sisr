/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.13.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout;
    QFrame *frameSettings;
    QVBoxLayout *verticalLayout_2;
    QSpacerItem *horizontalSpacer;
    QComboBox *comboBoxImageType;
    QComboBox *comboBoxScalling;
    QFrame *frame;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *pushButtonSwipeLeft;
    QLabel *labelChoosedImageNum;
    QPushButton *pushButtonSwipeRight;
    QPushButton *pushButtonCount;
    QTabWidget *tabWidget;
    QWidget *tabLoad;
    QVBoxLayout *verticalLayout;
    QFrame *frameFilePath;
    QHBoxLayout *horizontalLayout_3;
    QLabel *labelFilePath;
    QLineEdit *lineEditDirPath;
    QPushButton *pushButtonFilePath;
    QSpacerItem *horizontalSpacer_2;
    QFrame *frameImages;
    QHBoxLayout *horizontalLayout_4;
    QLabel *startImage;
    QLabel *LRImage;
    QLabel *label;
    QWidget *tabSplitColor;
    QGridLayout *gridLayout;
    QLabel *Image3;
    QLabel *Image1;
    QLabel *Image4;
    QLabel *Image2;
    QLabel *Image5;
    QWidget *tabLRHRPairs;
    QGridLayout *gridLayout_2;
    QLabel *BLRPart;
    QLabel *RHRPart;
    QLabel *label3PartCur;
    QLabel *GreyHRPart;
    QSlider *horizontalSlider3Part;
    QLabel *label2PartMax;
    QLabel *label4PartCur;
    QSlider *horizontalSlider4Part;
    QLabel *GHRPart;
    QLabel *label4PartMax;
    QLabel *GLRPart;
    QLabel *GreyLRPart;
    QLabel *BHRPart;
    QLabel *label1PartMax;
    QSlider *horizontalSlider1Part;
    QSlider *horizontalSlider2Part;
    QLabel *label1PartCur;
    QLabel *label2PartCur;
    QLabel *RLRPart;
    QLabel *label3PartMax;
    QWidget *tabHRAssembling;
    QGridLayout *gridLayout_3;
    QLabel *AssembPartStatistic2;
    QLabel *AssembPartLR;
    QLabel *AssembPartHR;
    QLabel *AssembPartHR7;
    QLabel *AssembPartHR2;
    QFrame *frame_2;
    QHBoxLayout *horizontalLayout_5;
    QLabel *labelCurPatch;
    QSlider *horizontalSliderHRAssemb;
    QLabel *labelPatchCount;
    QLabel *AssembPartHR5;
    QLabel *AssembPartStatistic5;
    QLabel *AssembPartLR8;
    QLabel *AssembPartStatistic6;
    QLabel *AssembPartStatistic3;
    QLabel *AssembPartLR5;
    QLabel *AssembPartHR8;
    QLabel *AssembPartLR1;
    QLabel *AssembPartHR1;
    QLabel *AssembPartHR4;
    QLabel *AssembStatistic;
    QLabel *AssembPartHR6;
    QLabel *LRImageAssemb;
    QLabel *AssembPartLR3;
    QLabel *AssembPartLR2;
    QLabel *AssembPartStatistic4;
    QLabel *AssembPartLR6;
    QLabel *AssembPartHR3;
    QLabel *AssembPartStatistic8;
    QLabel *AssembPartStatistic7;
    QLabel *AssembPartStatistic1;
    QLabel *AssembPartLR7;
    QLabel *AssembPartLR4;
    QWidget *tabResult;
    QGridLayout *gridLayout_4;
    QLabel *Result1;
    QLabel *Result2;
    QLabel *Result3;
    QLabel *labelResultStatistic;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1113, 581);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        horizontalLayout = new QHBoxLayout(centralWidget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        frameSettings = new QFrame(centralWidget);
        frameSettings->setObjectName(QString::fromUtf8("frameSettings"));
        frameSettings->setFrameShape(QFrame::StyledPanel);
        frameSettings->setFrameShadow(QFrame::Raised);
        verticalLayout_2 = new QVBoxLayout(frameSettings);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        horizontalSpacer = new QSpacerItem(100, 20, QSizePolicy::Fixed, QSizePolicy::Minimum);

        verticalLayout_2->addItem(horizontalSpacer);

        comboBoxImageType = new QComboBox(frameSettings);
        comboBoxImageType->setObjectName(QString::fromUtf8("comboBoxImageType"));
        comboBoxImageType->setSizeAdjustPolicy(QComboBox::AdjustToContents);

        verticalLayout_2->addWidget(comboBoxImageType);

        comboBoxScalling = new QComboBox(frameSettings);
        comboBoxScalling->setObjectName(QString::fromUtf8("comboBoxScalling"));
        comboBoxScalling->setSizeAdjustPolicy(QComboBox::AdjustToContents);

        verticalLayout_2->addWidget(comboBoxScalling);

        frame = new QFrame(frameSettings);
        frame->setObjectName(QString::fromUtf8("frame"));
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        horizontalLayout_2 = new QHBoxLayout(frame);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(6, 6, 6, 6);
        pushButtonSwipeLeft = new QPushButton(frame);
        pushButtonSwipeLeft->setObjectName(QString::fromUtf8("pushButtonSwipeLeft"));
        pushButtonSwipeLeft->setMaximumSize(QSize(32, 16777215));
        pushButtonSwipeLeft->setFocusPolicy(Qt::ClickFocus);

        horizontalLayout_2->addWidget(pushButtonSwipeLeft);

        labelChoosedImageNum = new QLabel(frame);
        labelChoosedImageNum->setObjectName(QString::fromUtf8("labelChoosedImageNum"));

        horizontalLayout_2->addWidget(labelChoosedImageNum);

        pushButtonSwipeRight = new QPushButton(frame);
        pushButtonSwipeRight->setObjectName(QString::fromUtf8("pushButtonSwipeRight"));
        pushButtonSwipeRight->setMaximumSize(QSize(32, 16777215));
        pushButtonSwipeRight->setFocusPolicy(Qt::ClickFocus);

        horizontalLayout_2->addWidget(pushButtonSwipeRight);


        verticalLayout_2->addWidget(frame);

        pushButtonCount = new QPushButton(frameSettings);
        pushButtonCount->setObjectName(QString::fromUtf8("pushButtonCount"));

        verticalLayout_2->addWidget(pushButtonCount);

        verticalLayout_2->setStretch(0, 1);

        horizontalLayout->addWidget(frameSettings);

        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setFocusPolicy(Qt::ClickFocus);
        tabWidget->setUsesScrollButtons(true);
        tabLoad = new QWidget();
        tabLoad->setObjectName(QString::fromUtf8("tabLoad"));
        verticalLayout = new QVBoxLayout(tabLoad);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        frameFilePath = new QFrame(tabLoad);
        frameFilePath->setObjectName(QString::fromUtf8("frameFilePath"));
        frameFilePath->setFrameShape(QFrame::StyledPanel);
        frameFilePath->setFrameShadow(QFrame::Raised);
        horizontalLayout_3 = new QHBoxLayout(frameFilePath);
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(6, 6, 6, 6);
        labelFilePath = new QLabel(frameFilePath);
        labelFilePath->setObjectName(QString::fromUtf8("labelFilePath"));

        horizontalLayout_3->addWidget(labelFilePath);

        lineEditDirPath = new QLineEdit(frameFilePath);
        lineEditDirPath->setObjectName(QString::fromUtf8("lineEditDirPath"));
        lineEditDirPath->setFocusPolicy(Qt::ClickFocus);
        lineEditDirPath->setReadOnly(true);

        horizontalLayout_3->addWidget(lineEditDirPath);

        pushButtonFilePath = new QPushButton(frameFilePath);
        pushButtonFilePath->setObjectName(QString::fromUtf8("pushButtonFilePath"));
        pushButtonFilePath->setFocusPolicy(Qt::ClickFocus);

        horizontalLayout_3->addWidget(pushButtonFilePath);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_2);


        verticalLayout->addWidget(frameFilePath);

        frameImages = new QFrame(tabLoad);
        frameImages->setObjectName(QString::fromUtf8("frameImages"));
        frameImages->setFrameShape(QFrame::StyledPanel);
        frameImages->setFrameShadow(QFrame::Raised);
        horizontalLayout_4 = new QHBoxLayout(frameImages);
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        startImage = new QLabel(frameImages);
        startImage->setObjectName(QString::fromUtf8("startImage"));

        horizontalLayout_4->addWidget(startImage);

        LRImage = new QLabel(frameImages);
        LRImage->setObjectName(QString::fromUtf8("LRImage"));

        horizontalLayout_4->addWidget(LRImage);


        verticalLayout->addWidget(frameImages);

        label = new QLabel(tabLoad);
        label->setObjectName(QString::fromUtf8("label"));

        verticalLayout->addWidget(label);

        verticalLayout->setStretch(1, 1);
        tabWidget->addTab(tabLoad, QString());
        tabSplitColor = new QWidget();
        tabSplitColor->setObjectName(QString::fromUtf8("tabSplitColor"));
        gridLayout = new QGridLayout(tabSplitColor);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        Image3 = new QLabel(tabSplitColor);
        Image3->setObjectName(QString::fromUtf8("Image3"));

        gridLayout->addWidget(Image3, 0, 2, 1, 1);

        Image1 = new QLabel(tabSplitColor);
        Image1->setObjectName(QString::fromUtf8("Image1"));

        gridLayout->addWidget(Image1, 0, 0, 1, 1);

        Image4 = new QLabel(tabSplitColor);
        Image4->setObjectName(QString::fromUtf8("Image4"));

        gridLayout->addWidget(Image4, 0, 3, 1, 1);

        Image2 = new QLabel(tabSplitColor);
        Image2->setObjectName(QString::fromUtf8("Image2"));

        gridLayout->addWidget(Image2, 0, 1, 1, 1);

        Image5 = new QLabel(tabSplitColor);
        Image5->setObjectName(QString::fromUtf8("Image5"));

        gridLayout->addWidget(Image5, 0, 4, 1, 1);

        tabWidget->addTab(tabSplitColor, QString());
        tabLRHRPairs = new QWidget();
        tabLRHRPairs->setObjectName(QString::fromUtf8("tabLRHRPairs"));
        gridLayout_2 = new QGridLayout(tabLRHRPairs);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        BLRPart = new QLabel(tabLRHRPairs);
        BLRPart->setObjectName(QString::fromUtf8("BLRPart"));

        gridLayout_2->addWidget(BLRPart, 1, 7, 1, 1);

        RHRPart = new QLabel(tabLRHRPairs);
        RHRPart->setObjectName(QString::fromUtf8("RHRPart"));

        gridLayout_2->addWidget(RHRPart, 0, 1, 1, 1);

        label3PartCur = new QLabel(tabLRHRPairs);
        label3PartCur->setObjectName(QString::fromUtf8("label3PartCur"));

        gridLayout_2->addWidget(label3PartCur, 4, 6, 1, 1);

        GreyHRPart = new QLabel(tabLRHRPairs);
        GreyHRPart->setObjectName(QString::fromUtf8("GreyHRPart"));

        gridLayout_2->addWidget(GreyHRPart, 0, 10, 1, 1);

        horizontalSlider3Part = new QSlider(tabLRHRPairs);
        horizontalSlider3Part->setObjectName(QString::fromUtf8("horizontalSlider3Part"));
        horizontalSlider3Part->setOrientation(Qt::Horizontal);

        gridLayout_2->addWidget(horizontalSlider3Part, 4, 7, 1, 1);

        label2PartMax = new QLabel(tabLRHRPairs);
        label2PartMax->setObjectName(QString::fromUtf8("label2PartMax"));

        gridLayout_2->addWidget(label2PartMax, 4, 5, 1, 1);

        label4PartCur = new QLabel(tabLRHRPairs);
        label4PartCur->setObjectName(QString::fromUtf8("label4PartCur"));

        gridLayout_2->addWidget(label4PartCur, 4, 9, 1, 1);

        horizontalSlider4Part = new QSlider(tabLRHRPairs);
        horizontalSlider4Part->setObjectName(QString::fromUtf8("horizontalSlider4Part"));
        horizontalSlider4Part->setOrientation(Qt::Horizontal);

        gridLayout_2->addWidget(horizontalSlider4Part, 4, 10, 1, 1);

        GHRPart = new QLabel(tabLRHRPairs);
        GHRPart->setObjectName(QString::fromUtf8("GHRPart"));

        gridLayout_2->addWidget(GHRPart, 0, 4, 1, 1);

        label4PartMax = new QLabel(tabLRHRPairs);
        label4PartMax->setObjectName(QString::fromUtf8("label4PartMax"));

        gridLayout_2->addWidget(label4PartMax, 4, 11, 1, 1);

        GLRPart = new QLabel(tabLRHRPairs);
        GLRPart->setObjectName(QString::fromUtf8("GLRPart"));

        gridLayout_2->addWidget(GLRPart, 1, 4, 1, 1);

        GreyLRPart = new QLabel(tabLRHRPairs);
        GreyLRPart->setObjectName(QString::fromUtf8("GreyLRPart"));

        gridLayout_2->addWidget(GreyLRPart, 1, 10, 1, 1);

        BHRPart = new QLabel(tabLRHRPairs);
        BHRPart->setObjectName(QString::fromUtf8("BHRPart"));

        gridLayout_2->addWidget(BHRPart, 0, 7, 1, 1);

        label1PartMax = new QLabel(tabLRHRPairs);
        label1PartMax->setObjectName(QString::fromUtf8("label1PartMax"));

        gridLayout_2->addWidget(label1PartMax, 4, 2, 1, 1);

        horizontalSlider1Part = new QSlider(tabLRHRPairs);
        horizontalSlider1Part->setObjectName(QString::fromUtf8("horizontalSlider1Part"));
        horizontalSlider1Part->setOrientation(Qt::Horizontal);

        gridLayout_2->addWidget(horizontalSlider1Part, 4, 1, 1, 1);

        horizontalSlider2Part = new QSlider(tabLRHRPairs);
        horizontalSlider2Part->setObjectName(QString::fromUtf8("horizontalSlider2Part"));
        horizontalSlider2Part->setOrientation(Qt::Horizontal);

        gridLayout_2->addWidget(horizontalSlider2Part, 4, 4, 1, 1);

        label1PartCur = new QLabel(tabLRHRPairs);
        label1PartCur->setObjectName(QString::fromUtf8("label1PartCur"));

        gridLayout_2->addWidget(label1PartCur, 4, 0, 1, 1);

        label2PartCur = new QLabel(tabLRHRPairs);
        label2PartCur->setObjectName(QString::fromUtf8("label2PartCur"));

        gridLayout_2->addWidget(label2PartCur, 4, 3, 1, 1);

        RLRPart = new QLabel(tabLRHRPairs);
        RLRPart->setObjectName(QString::fromUtf8("RLRPart"));

        gridLayout_2->addWidget(RLRPart, 1, 1, 1, 1);

        label3PartMax = new QLabel(tabLRHRPairs);
        label3PartMax->setObjectName(QString::fromUtf8("label3PartMax"));

        gridLayout_2->addWidget(label3PartMax, 4, 8, 1, 1);

        tabWidget->addTab(tabLRHRPairs, QString());
        tabHRAssembling = new QWidget();
        tabHRAssembling->setObjectName(QString::fromUtf8("tabHRAssembling"));
        gridLayout_3 = new QGridLayout(tabHRAssembling);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        AssembPartStatistic2 = new QLabel(tabHRAssembling);
        AssembPartStatistic2->setObjectName(QString::fromUtf8("AssembPartStatistic2"));

        gridLayout_3->addWidget(AssembPartStatistic2, 2, 3, 1, 1);

        AssembPartLR = new QLabel(tabHRAssembling);
        AssembPartLR->setObjectName(QString::fromUtf8("AssembPartLR"));

        gridLayout_3->addWidget(AssembPartLR, 0, 1, 1, 1);

        AssembPartHR = new QLabel(tabHRAssembling);
        AssembPartHR->setObjectName(QString::fromUtf8("AssembPartHR"));

        gridLayout_3->addWidget(AssembPartHR, 1, 1, 2, 1);

        AssembPartHR7 = new QLabel(tabHRAssembling);
        AssembPartHR7->setObjectName(QString::fromUtf8("AssembPartHR7"));

        gridLayout_3->addWidget(AssembPartHR7, 5, 4, 1, 1);

        AssembPartHR2 = new QLabel(tabHRAssembling);
        AssembPartHR2->setObjectName(QString::fromUtf8("AssembPartHR2"));

        gridLayout_3->addWidget(AssembPartHR2, 1, 3, 1, 1);

        frame_2 = new QFrame(tabHRAssembling);
        frame_2->setObjectName(QString::fromUtf8("frame_2"));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        horizontalLayout_5 = new QHBoxLayout(frame_2);
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        labelCurPatch = new QLabel(frame_2);
        labelCurPatch->setObjectName(QString::fromUtf8("labelCurPatch"));

        horizontalLayout_5->addWidget(labelCurPatch);

        horizontalSliderHRAssemb = new QSlider(frame_2);
        horizontalSliderHRAssemb->setObjectName(QString::fromUtf8("horizontalSliderHRAssemb"));
        horizontalSliderHRAssemb->setOrientation(Qt::Horizontal);

        horizontalLayout_5->addWidget(horizontalSliderHRAssemb);

        labelPatchCount = new QLabel(frame_2);
        labelPatchCount->setObjectName(QString::fromUtf8("labelPatchCount"));

        horizontalLayout_5->addWidget(labelPatchCount);


        gridLayout_3->addWidget(frame_2, 7, 0, 1, 6);

        AssembPartHR5 = new QLabel(tabHRAssembling);
        AssembPartHR5->setObjectName(QString::fromUtf8("AssembPartHR5"));

        gridLayout_3->addWidget(AssembPartHR5, 5, 2, 1, 1);

        AssembPartStatistic5 = new QLabel(tabHRAssembling);
        AssembPartStatistic5->setObjectName(QString::fromUtf8("AssembPartStatistic5"));

        gridLayout_3->addWidget(AssembPartStatistic5, 6, 2, 1, 1);

        AssembPartLR8 = new QLabel(tabHRAssembling);
        AssembPartLR8->setObjectName(QString::fromUtf8("AssembPartLR8"));

        gridLayout_3->addWidget(AssembPartLR8, 4, 5, 1, 1);

        AssembPartStatistic6 = new QLabel(tabHRAssembling);
        AssembPartStatistic6->setObjectName(QString::fromUtf8("AssembPartStatistic6"));

        gridLayout_3->addWidget(AssembPartStatistic6, 6, 3, 1, 1);

        AssembPartStatistic3 = new QLabel(tabHRAssembling);
        AssembPartStatistic3->setObjectName(QString::fromUtf8("AssembPartStatistic3"));

        gridLayout_3->addWidget(AssembPartStatistic3, 2, 4, 1, 1);

        AssembPartLR5 = new QLabel(tabHRAssembling);
        AssembPartLR5->setObjectName(QString::fromUtf8("AssembPartLR5"));

        gridLayout_3->addWidget(AssembPartLR5, 4, 2, 1, 1);

        AssembPartHR8 = new QLabel(tabHRAssembling);
        AssembPartHR8->setObjectName(QString::fromUtf8("AssembPartHR8"));

        gridLayout_3->addWidget(AssembPartHR8, 5, 5, 1, 1);

        AssembPartLR1 = new QLabel(tabHRAssembling);
        AssembPartLR1->setObjectName(QString::fromUtf8("AssembPartLR1"));

        gridLayout_3->addWidget(AssembPartLR1, 0, 2, 1, 1);

        AssembPartHR1 = new QLabel(tabHRAssembling);
        AssembPartHR1->setObjectName(QString::fromUtf8("AssembPartHR1"));

        gridLayout_3->addWidget(AssembPartHR1, 1, 2, 1, 1);

        AssembPartHR4 = new QLabel(tabHRAssembling);
        AssembPartHR4->setObjectName(QString::fromUtf8("AssembPartHR4"));

        gridLayout_3->addWidget(AssembPartHR4, 1, 5, 1, 1);

        AssembStatistic = new QLabel(tabHRAssembling);
        AssembStatistic->setObjectName(QString::fromUtf8("AssembStatistic"));

        gridLayout_3->addWidget(AssembStatistic, 4, 0, 3, 2);

        AssembPartHR6 = new QLabel(tabHRAssembling);
        AssembPartHR6->setObjectName(QString::fromUtf8("AssembPartHR6"));

        gridLayout_3->addWidget(AssembPartHR6, 5, 3, 1, 1);

        LRImageAssemb = new QLabel(tabHRAssembling);
        LRImageAssemb->setObjectName(QString::fromUtf8("LRImageAssemb"));

        gridLayout_3->addWidget(LRImageAssemb, 0, 0, 3, 1);

        AssembPartLR3 = new QLabel(tabHRAssembling);
        AssembPartLR3->setObjectName(QString::fromUtf8("AssembPartLR3"));

        gridLayout_3->addWidget(AssembPartLR3, 0, 4, 1, 1);

        AssembPartLR2 = new QLabel(tabHRAssembling);
        AssembPartLR2->setObjectName(QString::fromUtf8("AssembPartLR2"));

        gridLayout_3->addWidget(AssembPartLR2, 0, 3, 1, 1);

        AssembPartStatistic4 = new QLabel(tabHRAssembling);
        AssembPartStatistic4->setObjectName(QString::fromUtf8("AssembPartStatistic4"));

        gridLayout_3->addWidget(AssembPartStatistic4, 2, 5, 1, 1);

        AssembPartLR6 = new QLabel(tabHRAssembling);
        AssembPartLR6->setObjectName(QString::fromUtf8("AssembPartLR6"));

        gridLayout_3->addWidget(AssembPartLR6, 4, 3, 1, 1);

        AssembPartHR3 = new QLabel(tabHRAssembling);
        AssembPartHR3->setObjectName(QString::fromUtf8("AssembPartHR3"));

        gridLayout_3->addWidget(AssembPartHR3, 1, 4, 1, 1);

        AssembPartStatistic8 = new QLabel(tabHRAssembling);
        AssembPartStatistic8->setObjectName(QString::fromUtf8("AssembPartStatistic8"));

        gridLayout_3->addWidget(AssembPartStatistic8, 6, 5, 1, 1);

        AssembPartStatistic7 = new QLabel(tabHRAssembling);
        AssembPartStatistic7->setObjectName(QString::fromUtf8("AssembPartStatistic7"));

        gridLayout_3->addWidget(AssembPartStatistic7, 6, 4, 1, 1);

        AssembPartStatistic1 = new QLabel(tabHRAssembling);
        AssembPartStatistic1->setObjectName(QString::fromUtf8("AssembPartStatistic1"));

        gridLayout_3->addWidget(AssembPartStatistic1, 2, 2, 1, 1);

        AssembPartLR7 = new QLabel(tabHRAssembling);
        AssembPartLR7->setObjectName(QString::fromUtf8("AssembPartLR7"));

        gridLayout_3->addWidget(AssembPartLR7, 4, 4, 1, 1);

        AssembPartLR4 = new QLabel(tabHRAssembling);
        AssembPartLR4->setObjectName(QString::fromUtf8("AssembPartLR4"));

        gridLayout_3->addWidget(AssembPartLR4, 0, 5, 1, 1);

        tabWidget->addTab(tabHRAssembling, QString());
        tabResult = new QWidget();
        tabResult->setObjectName(QString::fromUtf8("tabResult"));
        gridLayout_4 = new QGridLayout(tabResult);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        Result1 = new QLabel(tabResult);
        Result1->setObjectName(QString::fromUtf8("Result1"));

        gridLayout_4->addWidget(Result1, 0, 0, 1, 1);

        Result2 = new QLabel(tabResult);
        Result2->setObjectName(QString::fromUtf8("Result2"));

        gridLayout_4->addWidget(Result2, 0, 1, 1, 1);

        Result3 = new QLabel(tabResult);
        Result3->setObjectName(QString::fromUtf8("Result3"));

        gridLayout_4->addWidget(Result3, 0, 2, 1, 1);

        labelResultStatistic = new QLabel(tabResult);
        labelResultStatistic->setObjectName(QString::fromUtf8("labelResultStatistic"));

        gridLayout_4->addWidget(labelResultStatistic, 1, 0, 1, 3);

        tabWidget->addTab(tabResult, QString());

        horizontalLayout->addWidget(tabWidget);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1113, 21));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "MainWindow", nullptr));
        pushButtonSwipeLeft->setText(QCoreApplication::translate("MainWindow", "<", nullptr));
        labelChoosedImageNum->setText(QString());
        pushButtonSwipeRight->setText(QCoreApplication::translate("MainWindow", ">", nullptr));
        pushButtonCount->setText(QCoreApplication::translate("MainWindow", "\320\227\320\260\320\277\321\203\321\201\321\202\320\270\321\202\321\214", nullptr));
        labelFilePath->setText(QCoreApplication::translate("MainWindow", "\320\237\321\203\321\202\321\214 \320\272 \321\204\320\260\320\271\320\273\320\260\320\274", nullptr));
        pushButtonFilePath->setText(QCoreApplication::translate("MainWindow", "\320\236\321\202\320\272\321\200\321\213\321\202\321\214", nullptr));
        startImage->setText(QString());
        LRImage->setText(QString());
        label->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tabLoad), QCoreApplication::translate("MainWindow", "\320\227\320\260\320\263\321\200\321\203\320\267\320\272\320\260", nullptr));
        Image3->setText(QString());
        Image1->setText(QString());
        Image4->setText(QString());
        Image2->setText(QString());
        Image5->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tabSplitColor), QCoreApplication::translate("MainWindow", "\320\230\321\201\321\205\320\276\320\264\320\275\320\276\320\265 \320\270\320\267\320\276\320\261\321\200\320\260\320\266\320\265\320\275\320\270\320\265", nullptr));
        BLRPart->setText(QString());
        RHRPart->setText(QString());
        label3PartCur->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        GreyHRPart->setText(QString());
        label2PartMax->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label4PartCur->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        GHRPart->setText(QString());
        label4PartMax->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        GLRPart->setText(QString());
        GreyLRPart->setText(QString());
        BHRPart->setText(QString());
        label1PartMax->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label1PartCur->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        label2PartCur->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        RLRPart->setText(QString());
        label3PartMax->setText(QCoreApplication::translate("MainWindow", "0", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tabLRHRPairs), QCoreApplication::translate("MainWindow", "LR-HR \320\277\320\260\321\200\321\213", nullptr));
        AssembPartStatistic2->setText(QString());
        AssembPartLR->setText(QString());
        AssembPartHR->setText(QString());
        AssembPartHR7->setText(QString());
        AssembPartHR2->setText(QString());
        labelCurPatch->setText(QString());
        labelPatchCount->setText(QString());
        AssembPartHR5->setText(QString());
        AssembPartStatistic5->setText(QString());
        AssembPartLR8->setText(QString());
        AssembPartStatistic6->setText(QString());
        AssembPartStatistic3->setText(QString());
        AssembPartLR5->setText(QString());
        AssembPartHR8->setText(QString());
        AssembPartLR1->setText(QString());
        AssembPartHR1->setText(QString());
        AssembPartHR4->setText(QString());
        AssembStatistic->setText(QString());
        AssembPartHR6->setText(QString());
        LRImageAssemb->setText(QString());
        AssembPartLR3->setText(QString());
        AssembPartLR2->setText(QString());
        AssembPartStatistic4->setText(QString());
        AssembPartLR6->setText(QString());
        AssembPartHR3->setText(QString());
        AssembPartStatistic8->setText(QString());
        AssembPartStatistic7->setText(QString());
        AssembPartStatistic1->setText(QString());
        AssembPartLR7->setText(QString());
        AssembPartLR4->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tabHRAssembling), QCoreApplication::translate("MainWindow", "\320\241\320\261\320\276\321\200\320\272\320\260 HR \320\270\320\267\320\276\320\261\321\200\320\260\320\266\320\265\320\275\320\270\321\217", nullptr));
        Result1->setText(QString());
        Result2->setText(QString());
        Result3->setText(QString());
        labelResultStatistic->setText(QString());
        tabWidget->setTabText(tabWidget->indexOf(tabResult), QCoreApplication::translate("MainWindow", "HR \320\270\320\267\320\276\320\261\321\200\320\260\320\266\320\265\320\275\320\270\320\265", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
