#ifndef DEDUCTION_H
#define DEDUCTION_H

#include <QMainWindow>

namespace Ui {
class Deduction;
}

class Deduction : public QMainWindow
{
    Q_OBJECT

public:
    explicit Deduction(QWidget *parent = nullptr);
    ~Deduction();

private:
    Ui::Deduction *ui;
};

#endif // DEDUCTION_H
