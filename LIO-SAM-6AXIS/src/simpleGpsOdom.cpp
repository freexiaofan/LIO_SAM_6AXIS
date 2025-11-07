#include "ros/ros.h"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <deque>
#include <mutex>

#include "utility.h"

class GNSSOdom : public ParamServer {
public:
    GNSSOdom(ros::NodeHandle &_nh) {
        nh = _nh;
        gpsSub = nh.subscribe(gpsTopic, 1000, &GNSSOdom::GNSSCB, this,
                              ros::TransportHints().tcpNoDelay());
        left_odom_pub = nh.advertise<nav_msgs::Odometry>("/gps_odom", 100, false);
        init_origin_pub = nh.advertise<nav_msgs::Odometry>("/init_odom", 10000, false);
        left_path_pub = nh.advertise<nav_msgs::Path>("/gps_path", 100);
        
        // **初始化所有变量**
        prev_yaw = 0.0;
        valid_yaw_count = 0;
        last_update_time = 0.0;
        prev_pose_left.setZero();
        prev_velocity.setZero();
        valid_pose_count = 0;
        last_valid_time = 0.0;
        position_history.resize(3);
    }

private:
    // 航向角平滑相关变量
    double prev_yaw;
    int valid_yaw_count;
    double last_update_time;
    std::deque<double> yaw_history;
    
    // **运动约束相关变量**
    Eigen::Vector3d prev_velocity;           // 上一次的速度
    std::deque<Eigen::Vector3d> position_history;  // 位置历史记录
    std::deque<double> time_history;         // 时间历史记录
    int valid_pose_count;                    // 有效位姿计数
    double last_valid_time;                  // 最后一次有效时间
    Eigen::Vector3d last_valid_position;     // 最后一次有效位置
    
    // **运动约束阈值**
    static constexpr double MIN_DISTANCE = 0.5;
    static constexpr double MAX_YAW_CHANGE = M_PI / 8;          // 22.5度
    static constexpr double YAW_FILTER_ALPHA = 0.3;
    static constexpr double MAX_SPEED = 10.0;                   // 30 m/s 最大速度
    static constexpr double MAX_ACCELERATION = 8.0;            // 8 m/s² 最大加速度
    static constexpr double MAX_LATERAL_ACCELERATION = 5.0;    // 5 m/s² 最大横向加速度
    static constexpr double MAX_YAW_RATE = M_PI / 4;          // 45度/秒 最大角速度
    static constexpr double OUTLIER_DISTANCE_THRESHOLD = 10.0; // 10米异常距离阈值
    static constexpr int HISTORY_SIZE = 5;                    // 历史记录大小

    // **角度标准化函数**
    double normalizeAngle(double angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
        return angle;
    }

    double angleDifference(double a, double b) {
        double diff = a - b;
        return normalizeAngle(diff);
    }

    // **简化的运动一致性检查：基于相邻位置向量方向**
    bool checkMotionConsistency( Eigen::Vector3d current_direction, 
                                 Eigen::Vector3d prev_direction) {
                                    
        // **核心：相邻位移向量方向一致性检查**
        auto v1 = current_direction.normalized();
        auto v2 = prev_direction.normalized();

        // 计算两个方向向量的夹角
        double dot_product = v1.dot(v2);
        double angle_diff = std::acos(dot_product);
        
        // **方向变化阈值：45度**
        const double MAX_DIRECTION_CHANGE = M_PI / 2;  // 45度
        ROS_WARN("  Angle difference: %.1f deg (max: %.1f deg)", 
                     angle_diff * 180.0 / M_PI, MAX_DIRECTION_CHANGE * 180.0 / M_PI);
        if (angle_diff > MAX_DIRECTION_CHANGE) {
            return false;  // 方向变化过大，拒绝此位置
        }
        
        return true;
    }

    void GNSSCB(const sensor_msgs::NavSatFixConstPtr &msg) {
        // std::cout << "gps status: " <<  int(msg->status.status) << " " << msg->status.service << std::endl;
        if (std::isnan(msg->latitude + msg->longitude + msg->altitude)) {
            return;
        }
        if (int(msg->status.status) != 2) {
            std::cout << " NOT RTK FIX --------------- return: "   << std::endl;
            return;
        }
        
        Eigen::Vector3d lla(msg->latitude, msg->longitude, msg->altitude);
        // std::cout << "LLA: " << lla.transpose() << std::endl;

        if (!initENU) {
            ROS_INFO("Init Origin GPS LLA  %f, %f, %f", msg->latitude, msg->longitude, msg->altitude);
            geo_converter.Reset(lla[0], lla[1], lla[2]);
            initENU = true;

            nav_msgs::Odometry init_msg;
            init_msg.header.stamp = msg->header.stamp;
            init_msg.header.frame_id = odometryFrame;
            init_msg.child_frame_id = "gps";
            init_msg.pose.pose.position.x = lla[0];
            init_msg.pose.pose.position.y = lla[1];
            init_msg.pose.pose.position.z = lla[2];
            init_msg.pose.covariance[0] = msg->position_covariance[0];
            init_msg.pose.covariance[7] = msg->position_covariance[4];
            init_msg.pose.covariance[14] = msg->position_covariance[8];
            init_msg.pose.pose.orientation = yaw_quat_left;
            init_origin_pub.publish(init_msg);
            
            // **初始化历史记录**
            prev_pose_left.setZero();
            position_history.clear();
            time_history.clear();
            return;
        }

        int status = int(msg->status.status);
        int satell_num = -1;
        double x, y, z;
        geo_converter.Forward(lla[0], lla[1], lla[2], x, y, z);
        Eigen::Vector3d raw_enu(x, y, z);

        // 距离太近则不更新 去除位置的微小跳动
        static Eigen::Vector3d first_enu(x, y, z);
        static Eigen::Vector3d direct_1 ;
        static bool ok = false;

        if (  !ok && (first_enu - raw_enu).norm() < 0.50 )
        {
            return;
        }

        if (!ok)
        {
            direct_1 = raw_enu - first_enu;
            first_enu = raw_enu;
            ok = true;
            return;
        }
        Eigen::Vector3d direct_2 = raw_enu - first_enu;

        double v1_deg = std::atan2(direct_1(1), direct_1(0)) * 180.0 / M_PI;
        double v2_deg = std::atan2(direct_2(1), direct_2(0)) * 180.0 / M_PI;
            std::cout << " v1_deg: " << v1_deg << std::endl;
            std::cout << " v2_deg: " << v2_deg << std::endl;

        if (checkMotionConsistency(direct_1, direct_2))
        {
            first_enu = raw_enu;
            direct_1 = direct_2;
        }
        else
        {
            std::cout << " GNSS ODOM rejected due to motion inconsistency. " << std::endl;
            return;
        }

        // 位置检查通过，直接使用原始GPS位置
        Eigen::Vector3d enu = raw_enu;

        // **发布消息**
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = msg->header.stamp;
        odom_msg.header.frame_id = odometryFrame;
        odom_msg.child_frame_id = "gps";

        odom_msg.pose.pose.position.x = enu(0);
        odom_msg.pose.pose.position.y = enu(1);
        odom_msg.pose.pose.position.z = enu(2);
        odom_msg.pose.covariance[0] = msg->position_covariance[0];
        odom_msg.pose.covariance[7] = msg->position_covariance[4];
        odom_msg.pose.covariance[14] = msg->position_covariance[8];
        odom_msg.pose.covariance[1] = lla[0];
        odom_msg.pose.covariance[2] = lla[1];
        odom_msg.pose.covariance[3] = lla[2];
        odom_msg.pose.covariance[4] = status;
        odom_msg.pose.covariance[5] = satell_num;
        
        odom_msg.pose.pose.orientation = yaw_quat_left;
        
        left_odom_pub.publish(odom_msg);

        // 路径发布
        left_path.header.frame_id = odometryFrame;
        left_path.header.stamp = msg->header.stamp;
        geometry_msgs::PoseStamped pose;
        pose.header = left_path.header;
        pose.pose.position.x = enu(0);
        pose.pose.position.y = enu(1);
        pose.pose.position.z = enu(2);
        pose.pose.orientation = yaw_quat_left;
        left_path.poses.push_back(pose);
        
        left_path_pub.publish(left_path);
    }


    ros::NodeHandle nh;
    ros::Publisher left_odom_pub, left_path_pub, init_origin_pub;
    ros::Subscriber gpsSub;

    std::mutex mutexLock;
    std::deque<sensor_msgs::NavSatFixConstPtr> gpsBuf;

    bool initENU = false;
    nav_msgs::Path left_path;
    GeographicLib::LocalCartesian geo_converter;
    Eigen::Vector3d prev_pose_left, prev_pose_right;
    geometry_msgs::Quaternion yaw_quat_left;
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "lio_sam_6axis");
    ros::NodeHandle nh;
    ROS_INFO("\033[1;32m---->   139 Simple GPS Odmetry Started.\033[0m");
    GNSSOdom gps(nh);
    ROS_INFO("\033[1;32m----> Simple GPS Odmetry Started.\033[0m");
    ros::spin();
    return 1;
}