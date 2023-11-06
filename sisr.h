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
#include <omp.h>
#include <QTime>

typedef QPair<cv::Mat, cv::Mat> PairImag;

// LR - фрагмент исходного изображения низкого разрешения
// HR - фрагмент исходного изображения высокого разрешения
// AHR - построенный фрагмент высокого разрешения

/// \brief Хранит пару фрагментов низкого и высокого разрешения. И данные для отображения в gui.
struct LRAHRInfo
{
    cv::Mat LR, HR;
    cv::Rect rect;
    QList<int> patchNums;
    QList<double> distToPatches;
};

/// \brief Класс содержащий вариации алгоритма повышения разрешения одиночного изображения.
class SISR
{
public:
    SISR();

    /// \brief Задаёт начальное изображение низкого разрешения.
    bool InitImage(const cv::Mat& image);

    /// \brief Строит список пар патчей и хэш таблицу для быстрого поиска.
    bool CreateLRHRPairs();

    /// \brief Строит изображение высокого разрешения.
    bool AssemblyHRImage();

    /// \brief Возвращает размер списока пар фрагментов низкого и высокого разрешения из исходного изображения.
    int GetPairsCount();
    /// \brief Возвращает пару фрагментов низкого и высокого разрешения из исходного изображения.
    PairImag GetPair(int i);

    /// \brief Возвращает размер списка патчей.
    int GetPatchesCount();
    /// \brief Возвращает объект из LR, AHR и дополнительной информации по сборке AHR.
    LRAHRInfo GetPatch(int i);

    /// \brief Возвращает изображение высокого разрешения.
    cv::Mat GetHRImage();

    static double RMSE(const cv::Mat& image1, const cv::Mat& image2);
    static double MaxDeviation(const cv::Mat& image1, const cv::Mat& image2);
    static double PSNR(const cv::Mat& image1, const cv::Mat& image2);
    static double SSIM(const cv::Mat& image1, const cv::Mat& image2);
    static double Max(const cv::Mat& image);


private:
    // Поиск подходящих пар LR-HR для LR
    /// \brief Простой поиск по списку за О(n).
    void GetNearestPairsIDS(const cv::Mat& part, QList<int>& nearest, QList<double>& dist);
    ///\ brief Поиск в хэш таблице за О(n).
    void GetNearestPairsIDS2(const cv::Mat& part, QList<int>& nearest, QList<double>& dist);

    double EuclidDist(const cv::Mat& i1, const cv::Mat& i2);
    double StandartDerivation(const cv::Mat& i1, const cv::Mat& i2);
    cv::Mat AssemblyHRPatch(QList<int> nearest, QList<double> weight);

private:
    /// \brief Изображения низкого и увеличенного разрешения
    cv::Mat mLRImage, mHRImage;

    /// \brief Список пар фрагментов низкого и высокого разрешения.
    QList<PairImag> mPairs;

    /// \brief Таблица пар для быстрого поиска похожих
    QHash<uint, QList<int>> mHash;

    /// \brief Список фрагментов низкого разрешения и полученных фрагментов высокого разрешения.
    QList<LRAHRInfo> mPatches;

    double sumT1, sumT2;
};

#endif // SISR_H
