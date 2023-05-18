#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    ui->pushButtonSwipeLeft->setVisible(false);
    ui->pushButtonSwipeRight->setVisible(false);
    ui->labelChoosedImageNum->setVisible(false);
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

    InitStartImage(0);
}

//void MainWindow::keyPressEvent(QKeyEvent *e)
//{
//    std::cout << e->key() << std::endl;
//    if (e->key() == Qt::Key_Left)
//    {
//        on_pushButtonSwipeLeft_clicked();
//    }
//    else if (e->key() == Qt::Key_Right)
//    {
//        on_pushButtonSwipeRight_clicked();
//    }
//}

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

void MainWindow::InitStartImage(const int imageNum)
{
    mSISR.InitImage(mStartImageFileNames[imageNum]);    // инициализация расчётов

    ui->startImage->setPixmap(mSISR.GetStartImage());
    ui->LRImage->setPixmap(mSISR.GetLRImage());
}




