#ifndef MAINWIN_H
#define MAINWIN_H

#include <QMainWindow>
#include<deduction.h>
#include<detection.h>
#include<probability.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWin; }
QT_END_NAMESPACE

class MainWin : public QMainWindow
{
    Q_OBJECT

public:
    MainWin(QWidget *parent = nullptr);
    ~MainWin();

    void show_deduc();
    void show_detec();
    void show_proba();

private slots:
    void dealdeduc();

private:
    Ui::MainWin *ui;

    Deduction deduc;
    Detection detec;
    Probability proba;

};
#endif // MAINWIN_H
