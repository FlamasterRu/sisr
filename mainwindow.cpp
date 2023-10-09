#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // настройки
    ui->pushButtonSwipeLeft->setVisible(false);
    ui->pushButtonSwipeRight->setVisible(false);
    ui->labelChoosedImageNum->setVisible(false);
    ui->pushButtonCount->setVisible(false);

    ui->comboBoxScalling->setVisible(false);
    ui->comboBoxScalling->clear();
    ui->comboBoxScalling->addItems(QStringList() << QString("2.0") << QStringList("4.0"));

    ui->comboBoxImageType->setVisible(false);
    ui->comboBoxImageType->clear();
    ui->comboBoxImageType->addItems(QStringList() << QString("Grey"));

    DefaultTab();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButtonFilePath_clicked()
{
    // путь к папке с файлами
//    QString dirPath = QFileDialog::getExistingDirectory(this, QString("Укажите директорию"));
//    if (dirPath.size() == 0)
//    {
//        return; // ничего не указали
//    }
    QString dirPath("D:/nikita_files/nngu/diplom/sisr_images");
    //QString dirPath("D:/projects/other/HR");

    // получение имён всех изображений
    QDir dir(dirPath);
    mStartImageFileNames = dir.entryList(QStringList() << "*.jpg" << "*.png");  // фильтры по расширениям файлов
    if (mStartImageFileNames.size() == 0)
    {
        return; // в папке нет картинок
    }
    mStartImageFileNames.sort();    // сортировка по алфавиту
    for (QString& fileName : mStartImageFileNames)
    {
        fileName = dir.absoluteFilePath(fileName);
    }
    ui->lineEditDirPath->setText(dirPath);

    // включение кнопок переключения изображения
    ui->pushButtonSwipeLeft->setVisible(true);
    ui->pushButtonSwipeRight->setVisible(true);
    ui->labelChoosedImageNum->setVisible(true);
    ui->labelChoosedImageNum->setText(QString("0/") + QString::number(mStartImageFileNames.size()));
    ui->comboBoxScalling->setVisible(true);
    ui->comboBoxImageType->setVisible(true);
    ui->pushButtonCount->setVisible(true);

    InitStartImage(0);
}

void MainWindow::on_pushButtonCount_clicked()
{
    DefaultTab();
    if (ui->comboBoxImageType->currentText() == QString("Grey"))
    {
        // вторая вкладка (исходное изображение)
        cv::cvtColor(mStartImage, mHRImage1, cv::COLOR_RGB2GRAY);
        ui->Image1->setPixmap(PixmapFromCVMat(mHRImage1, QImage::Format_Grayscale8));
        cv::cvtColor(mLRImage, mLRImage1, cv::COLOR_RGB2GRAY);
        ui->Image2->setPixmap(PixmapFromCVMat(mLRImage1, QImage::Format_Grayscale8));

        // третья вкладка (пары патчей)
        s1.InitImage(mLRImage1);
        QTime t1, t2;
        t1.restart();
        t2.restart();
        s1.CreateLRHRPairs();
        std::cout << "CreateLRHRPairs " << t1.restart()/1000. << std::endl;

        ui->label1PartCur->setVisible(true);
        ui->label1PartMax->setVisible(true);
        ui->horizontalSlider1Part->setVisible(true);
        ui->label1PartCur->setText(0);
        ui->label1PartMax->setText(QString::number(s1.GetPairsCount()-1));
        ui->horizontalSlider1Part->setMaximum(s1.GetPairsCount()-1);
        ui->horizontalSlider1Part->setValue(1); // костыль, чтобы картинки обновились
        ui->horizontalSlider1Part->setValue(0); // костыль, чтобы картинки обновились

        // четвёртая вкладка, список подходящих патчей (сборка HR изображения)
        t1.restart();
        s1.AssemblyHRImage();
        std::cout << "AssemblyHRImage " << t1.restart()/1000. << std::endl;
        std::cout << "All time " << t2.restart()/1000. << std::endl;

        ui->horizontalSliderHRAssemb->setVisible(true);
        ui->labelCurPatch->setVisible(true);
        ui->labelPatchCount->setVisible(true);
        ui->labelCurPatch->setText(0);
        ui->labelPatchCount->setText(QString::number(s1.GetPatchesCount()-1));
        ui->horizontalSliderHRAssemb->setMaximum(s1.GetPatchesCount()-1);
        ui->horizontalSliderHRAssemb->setValue(1); // костыль, чтобы картинки обновились
        ui->horizontalSliderHRAssemb->setValue(0); // костыль, чтобы картинки обновились

        // пятая вкладка, результаты сборки изображения
        double rmse = SISR::RMSE(mHRImage1, s1.GetHRImage());
        double maxDev = SISR::MaxDeviation(mHRImage1, s1.GetHRImage());
        double psnr = SISR::PSNR(mHRImage1, s1.GetHRImage());
        double ssim = SISR::SSIM(mHRImage1, s1.GetHRImage());
        ui->Result1->setPixmap(PixmapFromCVMat(mHRImage1, QImage::Format_Grayscale8));
        ui->Result2->setPixmap(PixmapFromCVMat(s1.GetHRImage(), QImage::Format_Grayscale8));
        ui->Result3->setPixmap(PixmapFromCVMat(mLRImage1, QImage::Format_Grayscale8));
        QString st = QString("Оценка результата:\n") +
                QString("Среднеквадратичное отклонение (RMSE) = ") + QString::number(rmse) + QString("\n") +
                QString("Максимальное отклонение = ") + QString::number(maxDev) + QString("\n") +
                QString("PSNR = ") + QString::number(psnr) + QString("\n") +
                QString("SSIM = ") + QString::number(ssim);
        ui->labelResultStatistic->setText(st);

    }
}

void MainWindow::on_pushButtonSwipeLeft_clicked()
{
    QStringList tmp = ui->labelChoosedImageNum->text().split(QString("/"));
    int currentImageNum = tmp.first().toInt(); // номер текущей картинки
    if (currentImageNum <= 0)
    {
        return; // влево нет цифр
    }

    ui->labelChoosedImageNum->setText(QString::number(currentImageNum-1) + QString("/") + QString::number(mStartImageFileNames.size()));
    InitStartImage(currentImageNum-1);
}

void MainWindow::on_pushButtonSwipeRight_clicked()
{
    QStringList tmp = ui->labelChoosedImageNum->text().split(QString("/"));
    int currentImageNum = tmp.first().toInt(); // номер текущей картинки
    if (currentImageNum >= mStartImageFileNames.size()-1)
    {
        return; // влево нет цифр
    }

    ui->labelChoosedImageNum->setText(QString::number(currentImageNum+1) + QString("/") + QString::number(mStartImageFileNames.size()));
    InitStartImage(currentImageNum+1);
}

void MainWindow::on_horizontalSlider1Part_valueChanged(int value)
{
    if (mStartImageFileNames.isEmpty())
    {
        return;
    }
    MPair p = s1.GetPair(value);
    ui->RLRPart->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(p.first, 50), QImage::Format_Grayscale8));
    ui->RHRPart->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(p.second, 20), QImage::Format_Grayscale8));
    ui->label1PartCur->setText(QString::number(value));
}

void MainWindow::on_comboBoxScalling_currentTextChanged(const QString &arg1)
{
    if (mStartImageFileNames.isEmpty())
    {
        return;
    }

    // создаёт LR изображение, в зависимости от выбранной настройки
    int scalling = 2;
    if (ui->comboBoxScalling->currentText() == QString("2.0"))
    {
        scalling = 2;
    }
    else if (ui->comboBoxScalling->currentText() == QString("4.0"))
    {
        scalling = 4;
    }
    cv::resize(mStartImage, mLRImage, cv::Size(mStartImage.cols/scalling, mStartImage.rows/scalling), cv::INTER_CUBIC);
    ui->LRImage->setPixmap(PixmapFromCVMat(mLRImage, QImage::Format_RGB888));
}

void MainWindow::on_horizontalSliderHRAssemb_valueChanged(int value)
{
    ui->labelCurPatch->setText(QString::number(value));

    ui->AssembPartLR1->setPixmap(QPixmap());
    ui->AssembPartLR2->setPixmap(QPixmap());
    ui->AssembPartLR3->setPixmap(QPixmap());
    ui->AssembPartLR4->setPixmap(QPixmap());
    ui->AssembPartLR5->setPixmap(QPixmap());
    ui->AssembPartLR6->setPixmap(QPixmap());
    ui->AssembPartLR7->setPixmap(QPixmap());
    ui->AssembPartLR8->setPixmap(QPixmap());
    ui->AssembPartHR1->setPixmap(QPixmap());
    ui->AssembPartHR2->setPixmap(QPixmap());
    ui->AssembPartHR3->setPixmap(QPixmap());
    ui->AssembPartHR4->setPixmap(QPixmap());
    ui->AssembPartHR5->setPixmap(QPixmap());
    ui->AssembPartHR6->setPixmap(QPixmap());
    ui->AssembPartHR7->setPixmap(QPixmap());
    ui->AssembPartHR8->setPixmap(QPixmap());
    ui->AssembPartStatistic1->setText(QString());
    ui->AssembPartStatistic2->setText(QString());
    ui->AssembPartStatistic3->setText(QString());
    ui->AssembPartStatistic4->setText(QString());
    ui->AssembPartStatistic5->setText(QString());
    ui->AssembPartStatistic6->setText(QString());
    ui->AssembPartStatistic7->setText(QString());
    ui->AssembPartStatistic8->setText(QString());


    Patch p = s1.GetPatch(value);

    cv::Mat tmp;
    mLRImage1.copyTo(tmp);
    QPixmap map = PixmapFromCVMat(tmp, QImage::Format_Grayscale8);
    QPainter painter(&map);
    painter.setPen(QColor("red"));
    painter.drawRect(p.rect.x, p.rect.y, p.rect.width, p.rect.height);
    ui->LRImageAssemb->setPixmap(map);

    ui->AssembPartLR->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(p.LR, 15), QImage::Format_Grayscale8));
    ui->AssembPartHR->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(p.HR, 10), QImage::Format_Grayscale8));

    int pairCount = p.patchNums.size();
    double maxDist = 0;
    double aveDist = 0;
    double minDist = 1e40;
    for (int i = 0; i < p.patchNums.size(); ++i)
    {
        if (p.distToPatches[i] > maxDist)
        {
            maxDist = p.distToPatches[i];
        }
        if (p.distToPatches[i] < minDist)
        {
            minDist = p.distToPatches[i];
        }
        aveDist += p.distToPatches[i];
    }
    aveDist /= static_cast<double>(pairCount);
    ui->AssembStatistic->setText(QString("Количество похожих пар: ") + QString::number(pairCount) + QString("\n") +
                                 QString("Максимальное отклонение: ") + QString::number(maxDist) + QString("\n") +
                                 QString("Среднее отклонение: ") + QString::number(aveDist) + QString("\n") +
                                 QString("Минимальное отклонение: ") + QString::number(minDist) + QString("\n"));

    if (pairCount >= 1)
    {
        MPair pr = s1.GetPair(p.patchNums[0]);
        ui->AssembPartLR1->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.first, 15), QImage::Format_Grayscale8));
        ui->AssembPartHR1->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.second, 10), QImage::Format_Grayscale8));
        ui->AssembPartStatistic1->setText(QString("Отклонение: ") + QString::number(p.distToPatches[0]));
    }
    if (pairCount >= 2)
    {
        MPair pr = s1.GetPair(p.patchNums[1]);
        ui->AssembPartLR2->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.first, 15), QImage::Format_Grayscale8));
        ui->AssembPartHR2->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.second, 10), QImage::Format_Grayscale8));
        ui->AssembPartStatistic2->setText(QString("Отклонение: ") + QString::number(p.distToPatches[1]));
    }
    if (pairCount >= 3)
    {
        MPair pr = s1.GetPair(p.patchNums[2]);
        ui->AssembPartLR3->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.first, 15), QImage::Format_Grayscale8));
        ui->AssembPartHR3->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.second, 10), QImage::Format_Grayscale8));
        ui->AssembPartStatistic3->setText(QString("Отклонение: ") + QString::number(p.distToPatches[2]));
    }
    if (pairCount >= 4)
    {
        MPair pr = s1.GetPair(p.patchNums[3]);
        ui->AssembPartLR4->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.first, 15), QImage::Format_Grayscale8));
        ui->AssembPartHR4->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.second, 10), QImage::Format_Grayscale8));
        ui->AssembPartStatistic4->setText(QString("Отклонение: ") + QString::number(p.distToPatches[3]));
    }
    if (pairCount >= 5)
    {
        MPair pr = s1.GetPair(p.patchNums[4]);
        ui->AssembPartLR5->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.first, 15), QImage::Format_Grayscale8));
        ui->AssembPartHR5->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.second, 10), QImage::Format_Grayscale8));
        ui->AssembPartStatistic5->setText(QString("Отклонение: ") + QString::number(p.distToPatches[4]));
    }
    if (pairCount >= 6)
    {
        MPair pr = s1.GetPair(p.patchNums[5]);
        ui->AssembPartLR6->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.first, 15), QImage::Format_Grayscale8));
        ui->AssembPartHR6->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.second, 10), QImage::Format_Grayscale8));
        ui->AssembPartStatistic6->setText(QString("Отклонение: ") + QString::number(p.distToPatches[5]));
    }
    if (pairCount >= 7)
    {
        MPair pr = s1.GetPair(p.patchNums[6]);
        ui->AssembPartLR7->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.first, 15), QImage::Format_Grayscale8));
        ui->AssembPartHR7->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.second, 10), QImage::Format_Grayscale8));
        ui->AssembPartStatistic7->setText(QString("Отклонение: ") + QString::number(p.distToPatches[6]));
    }
    if (pairCount >= 8)
    {
        MPair pr = s1.GetPair(p.patchNums[7]);
        ui->AssembPartLR8->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.first, 15), QImage::Format_Grayscale8));
        ui->AssembPartHR8->setPixmap(PixmapFromCVMat(UpscalePartImageGrey(pr.second, 10), QImage::Format_Grayscale8));
        ui->AssembPartStatistic8->setText(QString("Отклонение: ") + QString::number(p.distToPatches[7]));
    }

}


void MainWindow::DefaultTab()
{
    // вкладка LR-HR пары
    ui->label1PartCur->setVisible(false);
    ui->label1PartMax->setVisible(false);
    ui->horizontalSlider1Part->setVisible(false);
    ui->label2PartCur->setVisible(false);
    ui->label2PartMax->setVisible(false);
    ui->horizontalSlider2Part->setVisible(false);
    ui->label3PartCur->setVisible(false);
    ui->label3PartMax->setVisible(false);
    ui->horizontalSlider3Part->setVisible(false);
    ui->label4PartCur->setVisible(false);
    ui->label4PartMax->setVisible(false);
    ui->horizontalSlider4Part->setVisible(false);

    // четвёртая вкладка, список подходящих патчей (сборка HR изображения)
    ui->labelCurPatch->setVisible(false);
    ui->labelPatchCount->setVisible(false);
    ui->horizontalSliderHRAssemb->setVisible(false);
}

void MainWindow::InitStartImage(const int imageNum)
{
    mStartImage = cv::imread(mStartImageFileNames[imageNum].toStdString()); // чтение изображения
    cv::cvtColor(mStartImage, mStartImage, cv::COLOR_BGR2RGB);  // преобразует к нормальной rgb палитре (читает в bgr)
    ui->startImage->setPixmap(PixmapFromCVMat(mStartImage, QImage::Format_RGB888));
    on_comboBoxScalling_currentTextChanged(ui->comboBoxScalling->currentText());
}

QPixmap MainWindow::PixmapFromCVMat(const cv::Mat& image, QImage::Format format)
{
    QImage qIm( (uchar*)image.data, image.cols, image.rows, image.step, format );
    QPixmap pixel;
    pixel.convertFromImage(qIm);
    return pixel;
}

cv::Mat MainWindow::UpscalePartImage(const cv::Mat& image, int scale)
{
    cv::Mat res;
    cv::resize(image, res, cv::Size(image.rows*scale, image.cols*scale), cv::INTER_NEAREST);
    for (int i = 0; i < res.rows; ++i)
    {
        for (int j = 0; j < res.cols; ++j)
        {
            res.at<cv::Vec3b>(i,j)[0] = image.at<cv::Vec3b>(i/scale, j/scale)[0];
            res.at<cv::Vec3b>(i,j)[1] = image.at<cv::Vec3b>(i/scale, j/scale)[1];
            res.at<cv::Vec3b>(i,j)[2] = image.at<cv::Vec3b>(i/scale, j/scale)[2];
        }
    }
    return res;
}

cv::Mat MainWindow::UpscalePartImageGrey(const cv::Mat& image, int scale)
{
    cv::Mat res;
    cv::resize(image, res, cv::Size(image.rows*scale, image.cols*scale), cv::INTER_NEAREST);
    for (int i = 0; i < res.rows; ++i)
    {
        for (int j = 0; j < res.cols; ++j)
        {
            res.at<uchar>(i,j) = image.at<uchar>(i/scale, j/scale); // серый цвет обходится только так
        }
    }
    return res;
}








