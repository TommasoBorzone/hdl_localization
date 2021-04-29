/**
 * UnscentedKalmanFilterX.hpp
 * @author koide
 * 16/02/01
 **/
#ifndef KKL_UNSCENTED_KALMAN_FILTER_X_HPP
#define KKL_UNSCENTED_KALMAN_FILTER_X_HPP

#include <random>
#include <Eigen/Dense>

namespace kkl {
  namespace alg {

/**
 * @brief Unscented Kalman Filter class
 * @param T        scaler type
 * @param System   system class to be estimated
 */
template<typename T, class System>
class UnscentedKalmanFilterX {
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> VectorXt;
  typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> MatrixXt;
public:
  /**
   * @brief constructor
   * @param system               system to be estimated
   * @param state_dim            state vector dimension
   * @param input_dim            input vector dimension
   * @param measurement_dim      measurement vector dimension
   * @param process_noise        process noise covariance (state_dim x state_dim)
   * @param measurement_noise    measurement noise covariance (measurement_dim x measuremend_dim)
   * @param mean                 initial mean
   * @param residual             difference between true measurement and expected measurement
   * @param cov                  initial covariance
   * @param use_cov estim        To use or not the measuremets online covariance estimation
   */
  UnscentedKalmanFilterX(const System& system, int state_dim, int input_dim, int measurement_dim, const MatrixXt& process_noise, const MatrixXt& measurement_noise, const MatrixXt& residual, const VectorXt& mean, const MatrixXt& cov,  bool use_measurements_cov_estim)
    : state_dim(state_dim),
    input_dim(input_dim),
    measurement_dim(measurement_dim),
    N(state_dim),
    M(input_dim),
    K(measurement_dim),
    S(2 * state_dim + 1),
    mean(mean),
    cov(cov),
    system(system),
    residual(residual),
    residual_mean(residual),
    residual_mean2(residual),
    residual_variance(residual),
    process_noise(process_noise),
    measurement_noise(measurement_noise),
    lambda(1),
    normal_dist(0.0, 1.0),
    use_measurements_cov_estim(use_measurements_cov_estim)
  {
    weights.resize(S, 1);
    sigma_points.resize(S, N);
    ext_weights.resize(2 * (N + K) + 1, 1);
    ext_sigma_points.resize(2 * (N + K) + 1, N + K);
    expected_measurements.resize(2 * (N + K) + 1, K);

    for (int i = 0; i < measurement_dim; i++) {
      residual_mean2(i) = measurement_noise(i,i);
      residual_variance(i)= measurement_noise(i,i);
    }

    // initialize weights for unscented filter
    weights[0] = lambda / (N + lambda);
    for (int i = 1; i < 2 * N + 1; i++) {
      weights[i] = 1 / (2 * (N + lambda));
    }

    // weights for extended state space which includes error variances
    ext_weights[0] = lambda / (N + K + lambda);
    for (int i = 1; i < 2 * (N + K) + 1; i++) {
      ext_weights[i] = 1 / (2 * (N + K + lambda));
    }
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict() {
    // calculate sigma points
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i));
    }

    const auto& R = process_noise;

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += R;

    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief predict
   * @param control  input vector
   */
  void predict(const VectorXt& control) {
    // calculate sigma points
    ensurePositiveFinite(cov);
    computeSigmaPoints(mean, cov, sigma_points);
    for (int i = 0; i < S; i++) {
      sigma_points.row(i) = system.f(sigma_points.row(i), control);
    }

    const auto& R = process_noise;

    // unscented transform
    VectorXt mean_pred(mean.size());
    MatrixXt cov_pred(cov.rows(), cov.cols());

    mean_pred.setZero();
    cov_pred.setZero();
    for (int i = 0; i < S; i++) {
      mean_pred += weights[i] * sigma_points.row(i);
    }
    for (int i = 0; i < S; i++) {
      VectorXt diff = sigma_points.row(i).transpose() - mean_pred;
      cov_pred += weights[i] * diff * diff.transpose();
    }
    cov_pred += R;

    mean = mean_pred;
    cov = cov_pred;
  }

  /**
   * @brief correct
   * @param measurement  measurement vector
   */
  void correct(const VectorXt& measurement) {
    // create extended state space which includes error variances
	
	
    VectorXt ext_mean_pred = VectorXt::Zero(N + K, 1);
    MatrixXt ext_cov_pred = MatrixXt::Zero(N + K, N + K);
    ext_mean_pred.topLeftCorner(N, 1) = VectorXt(mean);
    ext_cov_pred.topLeftCorner(N, N) = MatrixXt(cov);
    ext_cov_pred.bottomRightCorner(K, K) = measurement_noise;

    ensurePositiveFinite(ext_cov_pred);
    computeSigmaPoints(ext_mean_pred, ext_cov_pred, ext_sigma_points);

    // unscented transform
    expected_measurements.setZero();
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurements.row(i) = system.h(ext_sigma_points.row(i).transpose().topLeftCorner(N, 1));
      expected_measurements.row(i) += VectorXt(ext_sigma_points.row(i).transpose().bottomRightCorner(K, 1));
    }

    VectorXt expected_measurement_mean = VectorXt::Zero(K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      expected_measurement_mean += ext_weights[i] * expected_measurements.row(i);
    }
    MatrixXt expected_measurement_cov = MatrixXt::Zero(K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      VectorXt diff = expected_measurements.row(i).transpose() - expected_measurement_mean;
      expected_measurement_cov += ext_weights[i] * diff * diff.transpose();
    }

    // calculated transformed covariance
    MatrixXt sigma = MatrixXt::Zero(N + K, K);
    for (int i = 0; i < ext_sigma_points.rows(); i++) {
      auto diffA = (ext_sigma_points.row(i).transpose() - ext_mean_pred);
      auto diffB = (expected_measurements.row(i).transpose() - expected_measurement_mean);
      sigma += ext_weights[i] * (diffA * diffB.transpose());
    }

    kalman_gain = sigma * expected_measurement_cov.inverse();
    const auto& K = kalman_gain;

    residual = measurement - expected_measurement_mean;
    VectorXt ext_mean = ext_mean_pred + K * (residual);
    MatrixXt ext_cov = ext_cov_pred - K * expected_measurement_cov * K.transpose();
    
    if (use_measurements_cov_estim) {
      updateMeasurementNoise(residual);
    }
    mean = ext_mean.topLeftCorner(N, 1);
    cov = ext_cov.topLeftCorner(N, N);
  }

  /*			getter			*/
  const VectorXt& getMean() const { return mean; }
  const MatrixXt& getCov() const { return cov; }
  const MatrixXt& getSigmaPoints() const { return sigma_points; }

  System& getSystem() { return system; }
  const System& getSystem() const { return system; }
  const MatrixXt& getProcessNoiseCov() const { return process_noise; }
  const MatrixXt& getMeasurementNoiseCov() const { return measurement_noise; }

  const MatrixXt& getKalmanGain() const { return kalman_gain; }

  /*			setter			*/
  UnscentedKalmanFilterX& setMean(const VectorXt& m) { mean = m;			return *this; }
  UnscentedKalmanFilterX& setCov(const MatrixXt& s) { cov = s;			return *this; }

  UnscentedKalmanFilterX& setProcessNoiseCov(const MatrixXt& p) { process_noise = p;			return *this; }
  UnscentedKalmanFilterX& setMeasurementNoiseCov(const MatrixXt& m) { measurement_noise = m;	return *this; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  const int state_dim;
  const int input_dim;
  const int measurement_dim;

  const int N;
  const int M;
  const int K;
  const int S;

  bool use_measurements_cov_estim;

public:
  VectorXt mean;

  VectorXt residual;
  VectorXt residual_mean;
  VectorXt residual_mean2;
  VectorXt residual_variance;

  MatrixXt cov;

  System system;
  MatrixXt process_noise;		//
  MatrixXt measurement_noise;	//

  T lambda;
  VectorXt weights;

  MatrixXt sigma_points;

  VectorXt ext_weights;
  MatrixXt ext_sigma_points;
  MatrixXt expected_measurements;

private:
  /**
   * @brief compute sigma points
   * @param mean          mean
   * @param cov           covariance
   * @param sigma_points  calculated sigma points
   */
  void computeSigmaPoints(const VectorXt& mean, const MatrixXt& cov, MatrixXt& sigma_points) {
    const int n = mean.size();
    assert(cov.rows() == n && cov.cols() == n);

    Eigen::LLT<MatrixXt> llt;
    llt.compute((n + lambda) * cov);
    MatrixXt l = llt.matrixL();

    sigma_points.row(0) = mean;
    for (int i = 0; i < n; i++) {
      sigma_points.row(1 + i * 2) = mean + l.col(i);
      sigma_points.row(1 + i * 2 + 1) = mean - l.col(i);
    }
  }

  /**
   * @brief make covariance matrix positive finite
   * @param cov  covariance matrix
   */
  void ensurePositiveFinite(MatrixXt& cov) {
    return;
    const double eps = 1e-9;

    Eigen::EigenSolver<MatrixXt> solver(cov);
    MatrixXt D = solver.pseudoEigenvalueMatrix();
    MatrixXt V = solver.pseudoEigenvectors();
    for (int i = 0; i < D.rows(); i++) {
      if (D(i, i) < eps) {
        D(i, i) = eps;
      }
    }

    cov = V * D * V.inverse();
  }

  /**
   * @brief This function estimates mean and variance of a m dimensional input_value
            signal.
   * @param cov  covariance matrix
   */
  void updateStatistics(const VectorXt& input, const int& window_leght) {

      VectorXt ext_mean = updateMean(residual_mean, input, window_leght);
      VectorXt pow = VectorXt::Zero(measurement_dim);
      for (int i = 0; i < measurement_dim; i++) {
          pow[i] = input[i]*input[i];
      }
      VectorXt ext_mean2    = updateMean(residual_mean2, pow, window_leght);
      VectorXt pow2 = VectorXt::Zero(measurement_dim);
      for (int i = 0; i < measurement_dim; i++) {
          pow2[i] = ext_mean[i]*ext_mean[i];
      }
      VectorXt ext_variance = (float(window_leght)/float(window_leght-1)) * (ext_mean2 - pow2);

      residual_mean = ext_mean.col(0);
      residual_mean2 = ext_mean2.col(0);
      residual_variance = ext_variance.col(0);

      return;
  }

  VectorXt updateMean(const VectorXt& mean,const VectorXt& input, const int& window_leght) {
      float a = float(window_leght-1)/float(window_leght);
      float b = float(1)/float(window_leght);

      return a * mean + b * input;
  }
  
  void updateMeasurementNoise(const VectorXt& input) {
    updateStatistics(input, 10);
    for (int i = 0; i < measurement_dim; i++) {
       measurement_noise(i,i) = residual_variance[i];
    }
	  return;
  }

public:
  MatrixXt kalman_gain;

  std::mt19937 mt;
  std::normal_distribution<T> normal_dist;
};

  }
}


#endif
