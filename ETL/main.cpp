/* This program will parse a CSV into a matrix format, used for machine learning. For this script to work, the
target variable should be the rightmost column of the CSV.*/

#include "ETL.h"

using namespace std;

// select parameters
const double TRAIN_SIZE = 0.7;


// select target directory
const string EXPORT_PATH_LOSS = "";
const string EXPORT_PATH_PARAM = "";
const string EXPORT_PATH_YHAT = "";



int main(int argc, char* argv[]) {

    if (argc < 4) // print error if wrong usage
        cout << "Error: Insufficient arguments" << endl;

    ETL myETL(argv[1], argv[2], argv[3]);

    vector<vector<string>> dataset = myETL.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    Eigen::MatrixXd dataM = myETL.toMat(dataset, rows, cols);

    Eigen::MatrixXd dataNormed = myETL.norm(dataM);

    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> dataSplit = myETL.TrainTestSplit(dataNormed, TRAIN_SIZE);
    tie(X_train, y_train, X_test, y_test) = dataSplit;

    cout << dataNormed << endl;

    return 0;
}