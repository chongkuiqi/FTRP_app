#ifndef OPTDEDUCTION_H
#define OPTDEDUCTION_H

#include <QMainWindow>
#include <QFileDialog>
namespace Ui {
class optdeduction;
}

class optdeduction : public QMainWindow
{
    Q_OBJECT

public:
    explicit optdeduction(QWidget *parent = nullptr);
    ~optdeduction();
signals:
    void optexit();

private slots:
    void on_exit_clicked();

    void on_browse_clicked();


    void on_classify_clicked();

    void on_save_clicked();

private:
    Ui::optdeduction *ui;
};

#endif // OPTDEDUCTION_H
