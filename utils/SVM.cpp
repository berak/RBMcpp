/* 
 * File:   SVM.cpp
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 */

#include "SVM.h"

namespace artelab
{

    SVM::SVM(cv::SVMParams params)
    {
        _model = cv::ml::SVM::create();
        set_params(params);
    }

    bool SVM::is_trained()
    {
        return _trained;
    }

    void SVM::load(std::string file)
    {
        _model = cv::Algorithm::load<cv::ml::SVM>(file.c_str());
        _trained = true;
    }

    void SVM::save(std::string file)
    {
        _model->save(file.c_str());
    }

    bool SVM::train(const cv::Mat& trainingData, const cv::Mat& targets, const int kfold)
    {
        if(kfold == NO_KFOLD)
        {
            _trained = _model->train(trainingData, cv::ml::ROW_SAMPLE, targets);
            _params = get_params();
        }
        else if(kfold > 1)
        {
            cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(trainingData, cv::ml::ROW_SAMPLE, targets);
            _trained = _model->trainAuto(trainData, kfold);
            _params = get_params();
        }
        else
        {
            _trained = false;
        }
        return _trained;
    }

    void SVM::predict(const cv::Mat& samples, cv::Mat& outPredictions)
    {
        outPredictions.create(samples.rows, 1, CV_32FC1);
        for(unsigned int i = 0; i < samples.rows; i++)
        {
            outPredictions.at<float>(i, 0) = _model->predict(samples.row(i));
        }
    }

    cv::SVMParams SVM::get_params()
    {
        _params.kernel_type = _model->getKernelType();
        _params.svm_type = _model->getType();
        _params.C = _model->getC();
        _params.nu = _model->getNu();
        _params.gamma = _model->getGamma();
        _params.degree = _model->getDegree();
        return _params;
    }

    SVM& SVM::set_params(cv::SVMParams params)
    {
        _params = params;
        _model->setKernel(_params.kernel_type); 
        _model->setType(_params.svm_type); 
        _model->setNu(_params.nu); 
        _model->setC(_params.C); 
        _model->setGamma(_params.gamma); 
        _model->setDegree(_params.degree); 
        return *this;
    }

    std::string SVM::description()
    {
        std::ostringstream ss;

        ss << "SVM Type: ";
        switch(_params.svm_type)
        {
            case cv::ml::SVM::C_SVC:
                ss << "C_SVC\n" << "C: " << _params.C;
                break;
            case cv::ml::SVM::NU_SVC:
                ss << "NU_SVC\n" << "Nu: " << _params.nu;
                break;
            case cv::ml::SVM::ONE_CLASS:
                ss << "ONE_CLASS\n" << "Nu: " << _params.nu;
                break;
            case cv::ml::SVM::EPS_SVR:
                ss << "EPS_SVR\n" << "C: " << _params.C << " p: " << _params.p;
                break;
            case cv::ml::SVM::NU_SVR:
                ss << "NU_SVR\n" << "C: " << _params.C << " Nu: " << _params.nu;
                break;
            default:
                ss << "-unknown type-";
        }
        ss << "\n";

        ss << "Kernel: ";
        switch(_params.kernel_type)
        {
            case cv::ml::SVM::LINEAR:
                ss << "LINEAR";
                break;
            case cv::ml::SVM::POLY:
                ss << "POLY degree" << _params.degree 
                   << " gamma: " << _params.gamma
                   << " coef0: " << _params.coef0;
                break;
            case cv::ml::SVM::RBF:
                ss << "RBF gamma: " << _params.gamma;
                break;
            case cv::ml::SVM::SIGMOID:
                ss << "SIGMOID gamma: " << _params.gamma 
                   << " coef0: " << _params.coef0;
                break;
            default:
                ss << "-unknown kernel-";
        }
        ss << "\n";

        ss << "TERM type: ";
        switch(_params.term_crit.type)
        {
            case cv::TermCriteria::MAX_ITER:
                ss << "iter " << _params.term_crit.maxCount;
                break;
            case cv::TermCriteria::EPS:
                ss << "epsilon " << _params.term_crit.epsilon;
                break;
            case cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER:
                ss << "iter&epsilon iter: " << _params.term_crit.maxCount
                   << " epsilon: " << _params.term_crit.epsilon;
                break;
            default:
                ss << "-no term criteria-";
        }

        return ss.str();
    }

}