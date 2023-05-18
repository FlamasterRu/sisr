#include "sisr.h"

SISR::SISR()
{

}

bool SISR::InitImage(const QString& fullFileName)
{
    mFullFileName = fullFileName;
    mStartImage = cv::imread(fullFileName.toStdString());
    cv::cvtColor(mStartImage, mStartImage, cv::COLOR_BGR2RGB);
    cv::resize(mStartImage, mLRImage, cv::Size(mStartImage.cols/2, mStartImage.rows/2), cv::INTER_CUBIC);




//    cv::cvtColor(imagecv, imagecv, cv::COLOR_BGR2RGB);
//    QImage image( (uchar*)imagecv.data, imagecv.cols, imagecv.rows, imagecv.step, QImage::Format_RGB888 );
//    QPixmap pixel;
//    pixel.convertFromImage(image);
//    ui->startImage->setPixmap(pixel);

//    cv::Mat LRImage;
//    cv::resize(imagecv, LRImage, cv::Size(imagecv.cols/2, imagecv.rows/2), cv::INTER_CUBIC);
//    QImage image2( (uchar*)LRImage.data, LRImage.cols, LRImage.rows, LRImage.step, QImage::Format_RGB888 );
//    QPixmap pixel2;
//    pixel2.convertFromImage(image2);
//    ui->LRImage->setPixmap(pixel2);

//    ui->labelStartInfo->setText(QString("Размер исходного изображения: ") + QString::number(imagecv.cols) + QString("x") + QString::number(imagecv.rows) );
//    ui->labelLRInfo->setText(QString("Размер LR изображения: ") + QString::number(LRImage.cols) + QString("x") + QString::number(LRImage.rows) );
}

QPixmap SISR::GetStartImage()    // для вывода на qt
{
    return PixmapFromCVMat(mStartImage);
}

QPixmap SISR::GetLRImage()   // для вывода на qt
{
    return PixmapFromCVMat(mLRImage);
}

QPixmap SISR::GetHRImage()   // для вывода на qt
{
    return PixmapFromCVMat(mHRImage);
}

QPixmap SISR::PixmapFromCVMat(const cv::Mat& image)
{
    QImage qIm( (uchar*)image.data, image.cols, image.rows, image.step, QImage::Format_RGB888 );
    QPixmap pixel;
    pixel.convertFromImage(qIm);
    return pixel;
}





