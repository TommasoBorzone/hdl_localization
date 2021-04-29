#include <mutex>
#include <memory>
#include <iostream>
#include <boost/circular_buffer.hpp>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>

#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistWithCovarianceStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>

#define H (7)
#define W (1)

namespace hdl_localization {

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer) {
  }
  virtual ~HdlLocalizationNodelet() {
  }

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    processing_time.resize(16);
    initialize_params();

    robot_odom_frame_id = private_nh.param<std::string>("robot_odom_frame_id", "robot_odom");
    odom_child_frame_id = private_nh.param<std::string>("odom_child_frame_id", "base_link");
    map_frame_id  = private_nh.param<std::string>("map_frame_id", "map");
    fcu_body_frame_id = private_nh.param<std::string>("fcu_body_frame_id", "base_link_frd");
    fcu_map_frame_id = private_nh.param<std::string>("fcu_map_frame_id", "map_nav");

    use_imu = private_nh.param<bool>("use_imu", true);
    invert_acc = private_nh.param<bool>("invert_acc", false);
    invert_gyro = private_nh.param<bool>("invert_gyro", false);
    if (use_imu) {
      NODELET_INFO("enable imu-based prediction");
      imu_sub = mt_nh.subscribe("/gpsimu_driver/imu_data", 256, &HdlLocalizationNodelet::imu_callback, this);
    }
    points_sub = mt_nh.subscribe("/velodyne_points", 1, &HdlLocalizationNodelet::points_callback, this);
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmap_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialpose_callback, this);

    pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 5, false);
    fcu_pose_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/mavros/vision_pose/pose_cov", 5, false);
    fcu_speed_pub = nh.advertise<geometry_msgs::TwistWithCovarianceStamped>("/mavros/vision_speed/speed_twist", 5, false);
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5, false);
    status_pub = nh.advertise<ScanMatchingStatus>("/status", 5, false);

    // global localization
    use_global_localization = private_nh.param<bool>("use_global_localization", true);
    if(use_global_localization) {
      NODELET_INFO_STREAM("wait for global localization services");
      ros::service::waitForService("/hdl_global_localization/set_global_map");
      ros::service::waitForService("/hdl_global_localization/query");

      set_global_map_service = nh.serviceClient<hdl_global_localization::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = nh.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/hdl_global_localization/query");

      relocalize_server = nh.advertiseService("/relocalize", &HdlLocalizationNodelet::relocalize, this);
    }
  }

private:
  pcl::Registration<PointT, PointT>::Ptr create_registration() const {
    std::string reg_method = private_nh.param<std::string>("reg_method", "NDT_OMP");
    std::string ndt_neighbor_search_method = private_nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
    double ndt_neighbor_search_radius = private_nh.param<double>("ndt_neighbor_search_radius", 2.0);
    double ndt_resolution = private_nh.param<double>("ndt_resolution", 1.0);

    if(reg_method == "NDT_OMP") {
      NODELET_INFO("NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          NODELET_INFO("search_method KDTREE is selected");
        } else {
          NODELET_WARN("invalid search method was given");
          NODELET_WARN("default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if(reg_method.find("NDT_CUDA") != std::string::npos) {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if(reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        NODELET_INFO_STREAM("search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      } else {
        NODELET_WARN("invalid search method was given");
      }
      return ndt;
    }

    NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    NODELET_INFO("create registration method for localization");
    registration = create_registration();

    // global localization
    NODELET_INFO("create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // distance filter
    use_distance_filter = private_nh.param<bool>("use_distance_filter", true);
    distance_near_thresh = private_nh.param<double>("distance_near_thresh", 1.0);
    distance_far_thresh = private_nh.param<double>("distance_far_thresh", 100.0);
    boost::shared_ptr<pcl::PassThrough<PointT>> filter_x(new pcl::PassThrough<PointT>(true));
    distance_filter_x = filter_x;
    distance_filter_x->setFilterFieldName("x");
    distance_filter_x->setFilterLimits(-distance_far_thresh, distance_far_thresh);
    boost::shared_ptr<pcl::PassThrough<PointT>> filter_y(new pcl::PassThrough<PointT>(true));
    distance_filter_y = filter_y;
    distance_filter_y->setFilterFieldName("y");
    distance_filter_y->setFilterLimits(-distance_far_thresh, distance_far_thresh);
    boost::shared_ptr<pcl::PassThrough<PointT>> filter_z(new pcl::PassThrough<PointT>(true));
    distance_filter_z = filter_z;
    distance_filter_z->setFilterFieldName("z");
    distance_filter_z->setFilterLimits(-distance_far_thresh, distance_far_thresh);

	
    // initialize pose estimator
    if(private_nh.param<bool>("specify_init_pose", true)) {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
        ros::Time::now(),
        Eigen::Vector3f(private_nh.param<double>("init_pos_x", 0.0), private_nh.param<double>("init_pos_y", 0.0), private_nh.param<double>("init_pos_z", 0.0)),
        Eigen::Quaternionf(private_nh.param<double>("init_ori_w", 1.0), private_nh.param<double>("init_ori_x", 0.0), private_nh.param<double>("init_ori_y", 0.0), private_nh.param<double>("init_ori_z", 0.0)),
        private_nh.param<double>("cool_time_duration", 0.5),
        private_nh.param<bool>("use_measurements_covariance_online_estimation", true)
      ));
    }
  }

private:
  /**
   * @brief callback for imu data
   * @param imu_msg
   */
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }

  /**
   * @brief callback for point cloud data
   * @param points_msg
   */
  void points_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    if(!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
    }

    if(!globalmap) {
      NODELET_ERROR("globalmap has not been received!!");
      return;
    }
    
    auto t1 = ros::WallTime::now();
	
    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    if(pcl_cloud->empty()) {
      NODELET_ERROR("cloud is empty!!");
      return;
    }

    // transform pointcloud into odom_child_frame_id
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    if(!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, this->tf_buffer)) {
        NODELET_ERROR("point cloud cannot be transformed into target frame!!");
        return;
    }
    
    auto filtered = downsample(cloud);
    filtered = distance_filter(filtered);
    last_scan = filtered;

    if(relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    Eigen::Matrix4f before = pose_estimator->matrix();

    // predict
    if(!use_imu) {
      pose_estimator->predict(stamp, Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
    } else {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      for(imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if(stamp < (*imu_iter)->header.stamp) {
          break;
        }
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;
        pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    // odometry-based prediction
    ros::Time last_correction_time = pose_estimator->last_correction_time();
    if(private_nh.param<bool>("enable_robot_odometry_prediction", false) && !last_correction_time.isZero()) {
      geometry_msgs::TransformStamped odom_delta;
      if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0.1))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0));
      } else if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0));
      }

      if(odom_delta.header.stamp.isZero()) {
        NODELET_WARN_STREAM("failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      }
    }

    // correct
    auto aligned = pose_estimator->correct(stamp, filtered);

    if(aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = map_frame_id;
      aligned->header.stamp = cloud->header.stamp;
      aligned_pub.publish(aligned);
    }
	
	auto t2 = ros::WallTime::now();
	processing_time.push_back((t2 - t1).toSec());
    double avg_processing_time = std::accumulate(processing_time.begin(), processing_time.end(), 0.0) / processing_time.size();
    NODELET_INFO_STREAM("processing_time: " << avg_processing_time * 1000.0 << "[msec]");

    publish_odometry(	points_msg->header.stamp, 
								pose_estimator->matrix_pose(), 
								pose_estimator->matrix_pose_cov(),
								pose_estimator->matrix_velocity(), 
								pose_estimator->matrix_velocity_cov());
  }

  /**
   * @brief callback for globalmap input
   * @param points_msg
   */
  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    registration->setInputTarget(globalmap);

    if(use_global_localization) {
      NODELET_INFO("set globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::toROSMsg(*globalmap, srv.request.global_map);

      if(!set_global_map_service.call(srv)) {
        NODELET_INFO("failed to set global map");
      } else {
        NODELET_INFO("done");
      }
    }
  }

  /**
   * @brief perform global localization to relocalize the sensor position
   * @param
   */
  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
    if(last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    if(!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      relocalizing = false;
      NODELET_INFO_STREAM("global localization failed");
      return false;
    }

    const auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
    pose = pose * delta_estimater->estimated_delta();

    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      ros::Time::now(),
      pose.translation(),
      Eigen::Quaternionf(pose.linear()),
      private_nh.param<double>("cool_time_duration", 0.5)));

    relocalizing = false;

    return true;
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(
          new hdl_localization::PoseEstimator(
            registration,
            ros::Time::now(),
            Eigen::Vector3f(p.x, p.y, p.z),
            Eigen::Quaternionf(q.w, q.x, q.y, q.z),
            private_nh.param<double>("cool_time_duration", 0.5))
    );
  }

  /**
   * @brief downsampling
   * @param cloud   input cloud
   * @return filtered downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  /**
   * @brief distance_filter
   * @param cloud   input cloud
   * @return filtered downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr distance_filter(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!use_distance_filter) {
        return cloud;
    }
    pcl::PointCloud<PointT>::Ptr filtered_x(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr filtered_y(new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr filtered_z(new pcl::PointCloud<PointT>());
    distance_filter_x->setInputCloud(cloud);
    distance_filter_x->filter(*filtered_x);
    distance_filter_y->setInputCloud(filtered_x);
    distance_filter_y->filter(*filtered_y);
    distance_filter_z->setInputCloud(filtered_y);
    distance_filter_z->filter(*filtered_z);

    return filtered_z;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose, const Eigen::Matrix<float, 6, 6>& pose_cov, const Eigen::Matrix4f& velocity, const Eigen::Matrix<float, 6, 6>& velocity_cov) {
    // broadcast the transformation over tf
    if(tf_buffer.canTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0))) {
      geometry_msgs::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
      map_wrt_frame.header.stamp = stamp;
      map_wrt_frame.header.frame_id = odom_child_frame_id;
      map_wrt_frame.child_frame_id = map_frame_id;

      geometry_msgs::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0), ros::Duration(0.1));
      Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

      geometry_msgs::TransformStamped map_wrt_odom;
      tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

      tf2::Transform odom_wrt_map;
      tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
      odom_wrt_map = odom_wrt_map.inverse();

      geometry_msgs::TransformStamped odom_trans;
      odom_trans.transform = tf2::toMsg(odom_wrt_map);
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = map_frame_id;
      odom_trans.child_frame_id = robot_odom_frame_id;

      tf_broadcaster.sendTransform(odom_trans);
    } else {
      geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = map_frame_id;
      odom_trans.child_frame_id = odom_child_frame_id;
      tf_broadcaster.sendTransform(odom_trans);
    }

    // publish the transformation message
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = map_frame_id;

    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.child_frame_id = odom_child_frame_id;
    odom.twist.twist.linear.x = velocity(0,3);
    odom.twist.twist.linear.y = velocity(1,3);
    odom.twist.twist.linear.z = velocity(2,3);
	
    // pubblish the pose for the flight control unit
    geometry_msgs::PoseWithCovarianceStamped fcu_pose_cov;	
    fcu_pose_cov.header.stamp=stamp;
    fcu_pose_cov.header.frame_id = fcu_map_frame_id;
    geometry_msgs::TransformStamped map_wrt_fcu_map = tf_buffer.lookupTransform(fcu_map_frame_id, map_frame_id, ros::Time(0), ros::Duration(0.1));
    Eigen::Matrix4f fcu_map2map = tf2::transformToEigen(map_wrt_fcu_map).cast<float>().matrix();
    geometry_msgs::TransformStamped fcu_body_wrt_odom = tf_buffer.lookupTransform(odom_child_frame_id, fcu_body_frame_id, ros::Time(0), ros::Duration(0.1));
    Eigen::Matrix4f odom2fcu_body = tf2::transformToEigen(fcu_body_wrt_odom).cast<float>().matrix();
    Eigen::Matrix4f fcu_map2fcu_body = fcu_map2map*pose*odom2fcu_body;
    tf::poseEigenToMsg(Eigen::Isometry3d(fcu_map2fcu_body.cast<double>()), fcu_pose_cov.pose.pose);
	
    // pubblish the speed for the flight control unit
    geometry_msgs::TwistWithCovarianceStamped fcu_speed_cov;
    fcu_speed_cov.header.stamp=stamp;
    fcu_speed_cov.header.frame_id = fcu_map_frame_id;
    geometry_msgs::Vector3Stamped fcu_speed_in, fcu_speed_out;
    fcu_speed_in.header.frame_id = map_frame_id;
    fcu_speed_in.vector.x = velocity(0,3);
    fcu_speed_in.vector.y = velocity(1,3);
    fcu_speed_in.vector.z = velocity(2,3);
    fcu_speed_out = tf_buffer.transform(fcu_speed_in, fcu_map_frame_id, ros::Duration(0.1));
    fcu_speed_cov.twist.twist.linear.x = fcu_speed_out.vector.x;
    fcu_speed_cov.twist.twist.linear.y = fcu_speed_out.vector.y;
    fcu_speed_cov.twist.twist.linear.z = fcu_speed_out.vector.z;
	
	
	for (int i = 0; i < pose_cov.size(); i++) {
		odom.pose.covariance[i] = *(pose_cov.data() + i);
		odom.twist.covariance[i] = *(velocity_cov.data() + i);
		fcu_pose_cov.pose.covariance[i] = *(pose_cov.data() + i);
		fcu_speed_cov.twist.covariance[i] = *(velocity_cov.data() + i);
	}
	
  pose_pub.publish(odom);
	fcu_pose_pub.publish(fcu_pose_cov);
	fcu_speed_pub.publish(fcu_speed_cov);
	
  }

  /**
   * @brief convert a Eigen::Matrix to TransformedStamped
   * @param stamp           timestamp
   * @param pose            pose matrix
   * @param frame_id        frame_id
   * @param child_frame_id  child_frame_id
   * @return transform
   */
  geometry_msgs::TransformStamped matrix2transform(const ros::Time& stamp, const Eigen::Matrix4f& pose, const std::string& frame_id, const std::string& child_frame_id) {
    Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
    quat.normalize();
    geometry_msgs::Quaternion odom_quat;
    odom_quat.w = quat.w();
    odom_quat.x = quat.x();
    odom_quat.y = quat.y();
    odom_quat.z = quat.z();

    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header.stamp = stamp;
    odom_trans.header.frame_id = frame_id;
    odom_trans.child_frame_id = child_frame_id;

    odom_trans.transform.translation.x = pose(0, 3);
    odom_trans.transform.translation.y = pose(1, 3);
    odom_trans.transform.translation.z = pose(2, 3);
    odom_trans.transform.rotation = odom_quat;

    return odom_trans;
  }

  /**
   * @brief publish scan matching status information
   */
  void publish_scan_matching_status(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    ScanMatchingStatus status;
    status.header = header;

    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();

    const double max_correspondence_dist = 0.5;

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;

    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);

    std::vector<double> errors(6, 0.0);

    if(pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "without_pred";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }

    status_pub.publish(status);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  std::string map_frame_id;
  std::string odom_child_frame_id;
  std::string robot_odom_frame_id;
  std::string fcu_map_frame_id;
  std::string fcu_body_frame_id;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber globalmap_sub;
  ros::Subscriber initialpose_sub;

  ros::Publisher pose_pub;
  ros::Publisher fcu_pose_pub;
  ros::Publisher fcu_speed_pub;
  ros::Publisher aligned_pub;
  ros::Publisher status_pub;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::PassThrough<PointT>::Ptr distance_filter_x;
  pcl::PassThrough<PointT>::Ptr distance_filter_y;
  pcl::PassThrough<PointT>::Ptr distance_filter_z;
  pcl::Registration<PointT, PointT>::Ptr registration;
  
  bool use_distance_filter;
  double distance_near_thresh;
  double distance_far_thresh;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // processing time buffer
  boost::circular_buffer<double> processing_time;

  // global localization
  bool use_global_localization;
  std::atomic_bool relocalizing;
  std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  ros::ServiceServer relocalize_server;
  ros::ServiceClient set_global_map_service;
  ros::ServiceClient query_global_localization_service;
};
}


PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
