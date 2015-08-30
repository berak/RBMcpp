/* 
 * File:   SVM.h
 * Author: Simone Albertini
 * 
 * albertini.simone@gmail.com
 * http://www.dicom.uninsubria.it/~simone.albertini/
 * 
 */

#ifndef SVM_H
#define	SVM_H

#include <opencv2/ml.hpp>

namespace cv 
{
    struct SVMParams // removed in 3.0, so we have to re-invent it.
    {
        SVMParams() 
            : svm_type(cv::ml::SVM::C_SVC)
            , kernel_type(cv::ml::SVM::LINEAR)
            , degree(0.5f)
            , gamma(0.5f)
            , coef0(1)
            , C(10)
            , nu(0.1f)
            , p(0.5f)
        {}
        SVMParams( int svm_type, int kernel_type,
                     double degree, double gamma, double coef0,
                     double Cvalue, double nu, double p,
                     //CvMat* class_weights, 
                     TermCriteria term_crit )
            : svm_type(svm_type)
            , kernel_type(kernel_type)
            , degree(degree)
            , gamma(gamma)
            , coef0(coef0)
            , C(Cvalue)
            , nu(nu)
            , p(p)
            , term_crit(term_crit)
        {}

        int         svm_type;
        int         kernel_type;
        double      degree; // for poly
        double      gamma;  // for poly/rbf/sigmoid
        double      coef0;  // for poly/sigmoid

        double      C;  // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        double      nu; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
        double      p; // for CV_SVM_EPS_SVR
        //CvMat*      class_weights; // for CV_SVM_C_SVC
        TermCriteria term_crit; // termination criteria
    };
}

namespace artelab
{


    class SVM 
    {
    public:

        enum { NO_KFOLD = 0 };

        SVM(cv::SVMParams params=cv::SVMParams());

        void load(std::string file);
        void save(std::string file);

        bool train(const cv::Mat& trainingData, const cv::Mat& targets, const int kfold=NO_KFOLD);
        void predict(const cv::Mat& samples, cv::Mat& outPredictions);

        bool is_trained();

        cv::SVMParams get_params();
        SVM& set_params(cv::SVMParams params);
        std::string description();


    private:
        cv::SVMParams  _params;
        cv::Ptr<cv::ml::SVM> _model;
        bool _trained;
    };
}
    
#endif	/* SVM_H */

