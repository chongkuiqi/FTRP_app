#ifndef PROBABILITY_H
#define PROBABILITY_H

#include <QMainWindow>



namespace Ui {
class Probability;
}

class Probability : public QMainWindow
{
    Q_OBJECT

public:
    explicit Probability(QWidget *parent = nullptr);
    ~Probability();

    void browse_img();
    void browse_gt();

    void extract_fe();

private:
    Ui::Probability *ui;
};

#endif // PROBABILITY_H
