#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "ui_mainwindow.h"
#include <sisr.h>

#include <iostream>
#include <QFileDialog>
#include <QDir>
#include <QKeyEvent>



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

    void on_horizontalSliderRPart_valueChanged(int value);  // листать кусочки ихображений
    void on_horizontalSliderGPart_valueChanged(int value);
    void on_horizontalSliderBPart_valueChanged(int value);
    void on_horizontalSliderGreyPart_valueChanged(int value);

private:
    void InitStartImage(const int imageNum);    // рисует начальное изображение

private:
    Ui::MainWindow *ui;

    QStringList mStartImageFileNames;
    SISR mSISR;
};

#endif // MAINWINDOW_H
