#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ui_mainwindow.h"
#include <sisr.h>

#include <iostream>
#include <QFileDialog>
#include <QDir>
#include <QKeyEvent>
#include <QPainter>



namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();


private slots:
    void on_pushButtonFilePath_clicked();   // открыть папку с картинками
    void on_pushButtonSwipeLeft_clicked();  // листать картинки влево
    void on_pushButtonSwipeRight_clicked(); // листать картинки вправо

    void on_pushButtonCount_clicked();  // здесь выполняются все операции увеличения разрешения

    void on_horizontalSlider1Part_valueChanged(int value);  // листать кусочки изображений

    void on_comboBoxScalling_currentTextChanged(const QString &arg1);

    void on_horizontalSliderHRAssemb_valueChanged(int value);

private:
    void DefaultTab();
    void InitStartImage(const int imageNum);    // рисует начальное изображение
    QPixmap PixmapFromCVMat(const cv::Mat& image, QImage::Format format);
    cv::Mat UpscalePartImage(const cv::Mat& image, int scale);
    cv::Mat UpscalePartImageGrey(const cv::Mat& image, int scale);

private:
    Ui::MainWindow *ui;

    QStringList mStartImageFileNames;
    SISR s1, s2, s3, s4;
    cv::Mat mLRImage1, mHRImage1;
    cv::Mat mStartImage, mLRImage, mHRImage;
};

#endif // MAINWINDOW_H
