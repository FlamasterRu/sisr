#ifndef SISR_H
#define SISR_H

#include <QObject>
#include <QPixmap>
#include <QImage>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QList>
#include <QMap>
#include <QPair>

#define MPair QPair<cv::Mat, cv::Mat>

class SISR
{
public:
    SISR();

    bool InitImage(const QString& fullFileName);    // инициализация начального изображения
    bool SplitColors(); // разобъёт LR изображение по цветам
    bool CreateLRHRPairs(); // для всех изображений строит пары мелких, крупных фрагментов

    QPixmap GetStartImage();    // для вывода на qt
    QPixmap GetLRImage();   // для вывода на qt
    QPixmap GetHRImage();   // для вывода на qt
    QPixmap GetLRRed();   // для вывода на qt
    QPixmap GetLRGreen();   // для вывода на qt
    QPixmap GetLRBlue();   // для вывода на qt
    QPixmap GetLRGrey();   // для вывода на qt
    int GetPairsMax();
    QPair<QPixmap, QPixmap> GetRPair(int i);
    QPair<QPixmap, QPixmap> GetGPair(int i);
    QPair<QPixmap, QPixmap> GetBPair(int i);
    QPair<QPixmap, QPixmap> GetGreyPair(int i);

private:
    QPixmap PixmapFromCVMat(const cv::Mat& image);  // преобразует картинку из opencv в qt вид
    QPixmap PixmapFromCVMatGrey(const cv::Mat& image);  // преобразует картинку из opencv в qt вид
    QList<MPair> GetLRHRPairs(const cv::Mat& image);   // для изображения строит пары мелких, крупных фрагментов
    cv::Mat UpscalePartImage(const cv::Mat& image, int scale);  // попиксельно повышает разрешение для улучшения визуализации фрагментов
    cv::Mat UpscalePartImageGrey(const cv::Mat& image, int scale);  // попиксельно повышает разрешение для улучшения визуализации фрагментов



private:
    QString mFullFileName;

    cv::Mat mStartImage, mLRImage, mHRImage;
    cv::Mat mLRRed, mLRGreen, mLRBlue, mLRGrey;
    QList<MPair> mRPairs, mGPairs, mBPairs, mGreyPairs;
};

#endif // SISR_H
