#ifndef SISR_H
#define SISR_H

#include <QObject>
#include <QPixmap>
#include <QImage>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

class SISR
{
public:
    SISR();

    bool InitImage(const QString& fullFileName);    // инициализация начального изображения

    QPixmap GetStartImage();    // для вывода на qt
    QPixmap GetLRImage();   // для вывода на qt
    QPixmap GetHRImage();   // для вывода на qt

private:
    QPixmap PixmapFromCVMat(const cv::Mat& image);  // преобразует картинку из opencv в qt вид

private:
    QString mFullFileName;

    cv::Mat mStartImage, mLRImage, mHRImage;
};

#endif // SISR_H
